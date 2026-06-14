"""Job matching engine — scores jobs against a user profile using semantic embeddings.

This matcher is domain-agnostic: it scores based on the user's `profile.yaml`
(skills/titles/keywords/locations) + optional negative keywords.
"""

import re
import math
import logging
from datetime import datetime, timedelta
from collections import Counter
from pathlib import Path

from models import Job

# Project root
_PROJECT_ROOT = Path(__file__).parent
LIFE_STORY_PATH = _PROJECT_ROOT / "life-story.md"

logger = logging.getLogger(__name__)

# Lazy-loaded sentence transformer model
_model = None
_model_name = "all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the sentence transformer model (first call takes a few seconds)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model '{_model_name}'...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_name)
        logger.info("Embedding model loaded.")
    return _model



SENIORITY_LEVELS = {
    "intern": 0,
    "junior": 1,
    "mid": 2,
    "senior": 3,
    "staff": 4,
    "principal": 5,
}


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization, strip non-alphanumeric."""
    return re.findall(r"[a-z0-9#+\-\.]+", text.lower())


def tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency (normalized by document length)."""
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {t: c / total for t, c in counts.items()}


def cosine_sim(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def load_life_story() -> str:
    """Load life-story.md if available, return empty string otherwise."""
    if LIFE_STORY_PATH.exists():
        try:
            return LIFE_STORY_PATH.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


class JobMatcher:
    """Score and rank jobs against a user profile using semantic embeddings."""

    def __init__(self, profile: dict):
        """
        profile should contain:
          - skills: list of skill strings
          - titles: list of desired job title strings
          - keywords: list of important keyword strings
          - preferred_locations: list of location strings (optional)
          - remote_preferred: bool (optional)
                    - seniority_level: one of intern/junior/mid/senior/staff/principal (optional)
                    - weights: dict with keys 'skills', 'title', 'semantic', 'location',
                        'experience', 'seniority', 'recency' (optional)
        """
        self.profile = profile
        self.weights = profile.get("weights", {
            "title": 0.20,
            "skills": 0.25,
            "semantic": 0.30,
            "location": 0.10,
            "experience": 0.05,
            "seniority": 0.10,
            "specialty": 0.10,
            "recency": 0.0,
        })
        # Ensure semantic weight exists for profiles with old-style weights
        if "semantic" not in self.weights:
            old_kw = self.weights.pop("keywords", 0.15)
            old_exp = self.weights.get("experience", 0.15)
            self.weights["semantic"] = old_kw + old_exp * 0.5
            self.weights["experience"] = old_exp * 0.5

        # Pre-tokenize profile components for token-based matching
        self._title_tokens = tokenize(" ".join(profile.get("titles", [])))
        self._locations = [loc.lower() for loc in profile.get("preferred_locations", [])]
        self._skill_tokens = tf(tokenize(" ".join(profile.get("skills", []))))
        self._preferred_seniority = self._normalize_preferred_seniority(
            profile.get("seniority_level", "mid")
        )
        self._specialty_keywords = [kw.lower() for kw in profile.get("keywords", [])]
        self._negative_keywords = [kw.lower() for kw in profile.get("negative_keywords", [])]
        self._strict_specialty = bool(profile.get("strict_specialty_filter", False))
        self._preferred_regions = [r.lower() for r in profile.get("preferred_regions", [])]

        # Load life-story for experience matching
        life_story_text = load_life_story()
        self._life_story_tokens = tokenize(life_story_text) if life_story_text else []
        self._life_story_tf = tf(self._life_story_tokens) if self._life_story_tokens else {}
        # Build the profile text for semantic embedding
        self._profile_text = self._build_profile_text(life_story_text)
        self._profile_embedding = None  # lazy computed

    def _build_profile_text(self, life_story: str) -> str:
        """Build a rich text representation of the profile for embedding."""
        parts = []

        titles = self.profile.get("titles", [])
        if titles:
            parts.append("Desired roles: " + ", ".join(titles))

        skills = self.profile.get("skills", [])
        if skills:
            parts.append("Skills: " + ", ".join(skills))

        keywords = self.profile.get("keywords", [])
        if keywords:
            parts.append("Expertise in: " + ", ".join(keywords))

        if life_story:
            # Use first ~1500 chars of life story for context
            parts.append("Background: " + life_story[:1500])

        return " ".join(parts)

    def is_relevant(self, job: Job) -> bool:
        """Blocklist + optional strict specialty filter."""
        text = f"{job.title} {job.description}".lower()
        if self._negative_keywords:
            for kw in self._negative_keywords:
                if kw and kw in text:
                    return False
        if self._strict_specialty and self._specialty_keywords:
            if not any(kw in text for kw in self._specialty_keywords):
                return False
        return True

    def _get_profile_embedding(self):
        """Compute and cache profile embedding."""
        if self._profile_embedding is None:
            model = _get_model()
            self._profile_embedding = model.encode(self._profile_text, normalize_embeddings=True)
        return self._profile_embedding

    def _semantic_score(self, job: Job) -> float:
        """Compute semantic similarity between profile and job using embeddings."""
        profile_emb = self._get_profile_embedding()

        # Use cached embedding from batch encoding if available
        if hasattr(job, '_cached_embedding'):
            job_emb = job._cached_embedding
        else:
            model = _get_model()
            job_text = f"{job.title}. {job.company}. {job.description[:2000]}"
            job_emb = model.encode(job_text, normalize_embeddings=True)

        # Dot product of normalized vectors = cosine similarity
        sim = float(profile_emb @ job_emb)
        # Clamp and rescale: raw cosine similarity for text is usually 0.1-0.7
        # Floor raised to 0.25 so only clearly relevant jobs score above 0
        sim = max(0.0, min(1.0, (sim - 0.25) / 0.45))
        return sim

    def score(self, job: Job) -> tuple[float, dict]:
        """
        Score a job from 0.0 to 1.0.
        Returns (score, details_dict).
        """
        job_text = f"{job.title} {job.description}".lower()
        job_tokens = tokenize(job_text)
        job_tf = tf(job_tokens)

        # 1. Title similarity — cosine similarity between desired titles and job title
        title_tokens = tokenize(job.title)
        title_tf = tf(title_tokens)
        profile_title_tf = tf(self._title_tokens)
        title_score = cosine_sim(title_tf, profile_title_tf)

        # 2. Skill keyword overlap — TF-IDF cosine against profile skills
        skill_score = cosine_sim(job_tf, self._skill_tokens) if self._skill_tokens else 0.0
        # Rescale: overlap is typically 0.0–0.15, map to 0–1
        skill_score = min(1.0, skill_score / 0.10)

        # 3. Semantic similarity — deep embedding-based matching
        semantic_score = self._semantic_score(job)

        # 4. Location match
        location_score = self._location_score(job)

        # 5. Experience match — life-story TF-IDF (lightweight supplement)
        experience_score = 0.0
        if self._life_story_tf:
            experience_score = cosine_sim(job_tf, self._life_story_tf)

        # 6. Specialty boost — CV/3D/robotics/perception keyword density
        specialty_score = 0.0
        if self._specialty_keywords:
            hits = sum(1 for kw in self._specialty_keywords if kw in job_text)
            specialty_score = min(1.0, hits / 5.0)

        # 7. Seniority fit — penalize jobs requiring more seniority than preferred
        seniority_score = self._seniority_score(job)

        w = self.weights
        total = (
            w.get("title", 0.20) * title_score
            + w.get("skills", 0.25) * skill_score
            + w.get("semantic", 0.30) * semantic_score
            + w.get("location", 0.10) * location_score
            + w.get("experience", 0.05) * experience_score
            + w.get("seniority", 0.10) * seniority_score
            + w.get("specialty", 0.10) * specialty_score
        )
        # Final block: irrelevant jobs score 0
        if not self.is_relevant(job):
            total = 0.0
        total = min(max(total, 0.0), 1.0)

        details = {
            "title_score": round(title_score, 3),
            "skill_score": round(skill_score, 3),
            "semantic_score": round(semantic_score, 3),
            "location_score": round(location_score, 3),
            "experience_score": round(experience_score, 3),
            "seniority_score": round(seniority_score, 3),
            "specialty_score": round(specialty_score, 3),
            "weighted_total": round(total, 3),
        }

        return round(total, 3), details

    def _location_score(self, job: Job) -> float:
        """Score location with strong boost for Remote + preferred regions/countries."""
        text = f"{job.location} {job.title} {job.description[:500]}".lower()

        # Remote preference
        if self.profile.get("remote_preferred") and any(k in text for k in ["remote", "work from home", "wfh"]):
            return 1.0

        # Preferred explicit locations
        if self._locations:
            for pref in self._locations:
                if pref and pref in text:
                    return 1.0

        # Preferred regions (e.g. Middle East / MENA)
        if self._preferred_regions:
            for region in self._preferred_regions:
                if region and region in text:
                    return 0.9

        return 0.0

    def _recency_score(self, job: Job) -> float:
        """Score from 0-1 based on how recently the job was posted. 1.0 = today."""
        if not job.date_posted:
            return 0.3  # Unknown date gets a small default
        try:
            date_str = job.date_posted[:10]
            posted = datetime.fromisoformat(date_str)
            days_ago = (datetime.now() - posted).days
            if days_ago < 0:
                days_ago = 0
            return max(0.0, 1.0 - days_ago / 30.0)
        except (ValueError, TypeError):
            return 0.3

    def _normalize_preferred_seniority(self, value: str) -> int:
        text = (value or "mid").strip().lower()
        if text in SENIORITY_LEVELS:
            return SENIORITY_LEVELS[text]
        if text in {"entry", "entry-level", "associate", "new grad", "graduate"}:
            return SENIORITY_LEVELS["junior"]
        if text in {"mid-level", "intermediate"}:
            return SENIORITY_LEVELS["mid"]
        if text in {"sr", "lead"}:
            return SENIORITY_LEVELS["senior"]
        return SENIORITY_LEVELS["mid"]

    def _extract_job_seniority_level(self, job: Job) -> int | None:
        text = f"{job.title} {job.description}".lower()
        patterns = [
            (SENIORITY_LEVELS["principal"], ["principal", "distinguished", "fellow"]),
            (SENIORITY_LEVELS["staff"], ["staff", "architect"]),
            (SENIORITY_LEVELS["senior"], [" senior ", " sr ", "lead", "manager", "head of"]),
            (SENIORITY_LEVELS["mid"], ["mid-level", "mid level", "intermediate"]),
            (SENIORITY_LEVELS["junior"], ["junior", "associate", "entry-level", "entry level", "graduate", "new grad"]),
            (SENIORITY_LEVELS["intern"], ["intern", "internship", "trainee"]),
        ]

        padded = f" {text} "
        for level, tokens in patterns:
            for token in tokens:
                if token in padded:
                    return level
        return None

    def _seniority_score(self, job: Job) -> float:
        level = self._extract_job_seniority_level(job)
        if level is None:
            # Unknown seniority: neutral-ish
            return 0.45

        # If user wants internships/entry-level, be more strict against senior roles.
        if self._preferred_seniority <= SENIORITY_LEVELS["junior"]:
            if level == SENIORITY_LEVELS["intern"]:
                return 1.0
            if level == SENIORITY_LEVELS["junior"]:
                return 0.75
            if level == SENIORITY_LEVELS["mid"]:
                return 0.25
            return 0.0

        # General case
        delta = level - self._preferred_seniority
        if delta <= -1:
            return 0.9
        if delta == 0:
            return 1.0
        if delta == 1:
            return 0.35
        return 0.0

    def rank(self, jobs: list[Job], min_score: float = 0.0) -> list[Job]:
        """Score and sort jobs."""
        model = _get_model()
        job_texts = [f"{j.title}. {j.company}. {j.description[:2000]}" for j in jobs]
        if job_texts:
            logger.info(f"Computing embeddings for {len(job_texts)} jobs...")
            job_embeddings = model.encode(job_texts, normalize_embeddings=True,
                                          batch_size=64, show_progress_bar=len(job_texts) > 50)
            # Cache embeddings on jobs for the scoring step
            for job, emb in zip(jobs, job_embeddings):
                job._cached_embedding = emb

        for job in jobs:
            score, details = self.score(job)
            job.match_score = score
            job.match_details = details
            # Clean up cached embedding
            if hasattr(job, '_cached_embedding'):
                del job._cached_embedding

        # Sort by score first, then by date (newer first) as tiebreaker
        def _date_key(j):
            d = j.date_posted
            if not d or not isinstance(d, str):
                return "0000"
            return d if d.lower() not in ("nan", "none", "nat") else "0000"
        ranked = sorted(jobs, key=lambda j: (j.match_score, _date_key(j)), reverse=True)
        if min_score > 0:
            ranked = [j for j in ranked if j.match_score >= min_score]

        return ranked
