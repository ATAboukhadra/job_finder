"""Data models for the job application pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List


# --- Remote / home-office classification ---------------------------------

# Phrases that clearly indicate a remote / home-office work arrangement.
_REMOTE_PHRASES = (
    "fully remote", "100% remote", "100 % remote", "fully-remote",
    "work from home", "work-from-home", "work from anywhere",
    "work remotely", "remote work", "remote-first", "remote first",
    "remote position", "remote role", "remote job", "remote opportunity",
    "remote possible", "remote möglich", "remote-friendly", "remote friendly",
    "home office", "home-office", "homeoffice", "telecommute", "telework",
    "wfh",
)

# Phrases indicating a hybrid arrangement (partial remote).
_HYBRID_PHRASES = (
    "hybrid", "partially remote", "partly remote", "remote/onsite",
    "remote / onsite", "office and remote", "flexible work arrangement",
    "days in office", "days per week in the office",
)

# Substrings where "remote" does NOT mean a work arrangement (domain terms
# common in computer-vision / robotics roles). Stripped before phrase matching.
_REMOTE_FALSE_POSITIVES = (
    "remote sensing", "remote control", "remotely operated",
    "remote operation", "remote server", "remote desktop",
    "remote procedure", "remote vehicle", "teleoperation", "tele-operation",
)

# Tokens that, when present in the structured location/job_type fields,
# reliably mark a job as remote.
_REMOTE_LOCATION_TOKENS = (
    "remote", "anywhere", "home office", "homeoffice", "home-office",
    "work from home", "telework", "telecommute",
)


def classify_remote(title: str = "", description: str = "",
                    location: str = "", job_type: str = "") -> str:
    """Classify a job's work arrangement as 'remote', 'hybrid', or 'onsite'.

    Structured fields (location / job_type) are trusted first; otherwise the
    title + description are scanned for arrangement phrases, with common
    domain false-positives (e.g. "remote sensing") removed beforehand.
    """
    loc = (location or "").lower()
    jt = (job_type or "").lower()

    if any(tok in loc for tok in _REMOTE_LOCATION_TOKENS) or "remote" in jt:
        return "remote"
    if "hybrid" in loc or "hybrid" in jt:
        return "hybrid"

    blob = f" {title} {description} ".lower()
    for fp in _REMOTE_FALSE_POSITIVES:
        blob = blob.replace(fp, " ")

    if any(p in blob for p in _REMOTE_PHRASES):
        return "remote"
    if any(p in blob for p in _HYBRID_PHRASES):
        return "hybrid"
    return "onsite"


def detect_remote(title: str = "", description: str = "",
                  location: str = "", job_type: str = "") -> bool:
    """True when the job is fully remote / home-office (not hybrid/onsite)."""
    return classify_remote(title, description, location, job_type) == "remote"


class JobBoard(Enum):
    INDEED = "indeed"
    LINKEDIN = "linkedin"
    GLASSDOOR = "glassdoor"
    STEPSTONE = "stepstone"
    REMOTIVE = "remotive"
    ADZUNA = "adzuna"
    JSEARCH = "jsearch"
    ARBEITNOW = "arbeitnow"
    THEMUSE = "themuse"
    HIMALAYAS = "himalayas"
    GOOGLE = "google"
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    LINKEDIN_POSTS = "linkedin_posts"
    INTERNET = "internet"
    BAYT = "bayt"
    GULFTALENT = "gulftalent"
    WUZZUF = "wuzzuf"


@dataclass
class Job:
    title: str
    company: str
    location: str
    url: str
    board: JobBoard
    description: str = ""
    salary: str = ""
    date_posted: str = ""
    job_type: str = ""  # full-time, part-time, contract
    is_remote: bool = False  # fully remote / home-office
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    match_score: float = 0.0
    match_details: Dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Unique identifier based on URL."""
        return self.url

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "url": self.url,
            "board": self.board.value,
            "description": self.description,
            "salary": self.salary,
            "date_posted": self.date_posted,
            "job_type": self.job_type,
            "is_remote": self.is_remote,
            "scraped_at": self.scraped_at,
            "match_score": self.match_score,
            "match_details": self.match_details,
        }


@dataclass
class SearchQuery:
    keywords: str
    location: str = ""
    remote: bool = False
    job_type: str = ""  # full-time, part-time, contract
    max_age_days: int = 30
    boards: List[JobBoard] = field(default_factory=lambda: list(JobBoard))
