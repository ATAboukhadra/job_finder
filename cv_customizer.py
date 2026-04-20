"""Automated CV customization engine.

Reads life-story.md + job description, generates tailored LaTeX files
(employment.tex, skills.tex, projects.tex), compiles to PDF.
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from llm import generate_latex, generate_structured, check_ollama_available
from user_profile import load_person, split_name

logger = logging.getLogger(__name__)

# Project root — life-story.md lives here by default
_PROJECT_ROOT = Path(__file__).parent

# Default CV directory: ./cv/ inside the project.
# Override via `cv_dir` in profile.yaml (supports ~ expansion and absolute paths).
_DEFAULT_CV_DIR = _PROJECT_ROOT / "cv"

# Stable files to symlink from cv/ into each per-application directory
SYMLINK_FILES = [
    "education.tex", "teaching.tex", "publications.tex", "misc.tex",
    "referee.tex", "referee-full.tex",
    "own-bib.bib", "photo.png", "photo.jpg", "settings.sty",
]

# Optional few-shot examples for specific domains. Not required for web/dev CVs.
# Keep empty unless you add example files and mappings.
EXAMPLE_EMPLOYMENT: Dict[str, Path] = {}
EXAMPLE_SKILLS: Dict[str, Path] = {}
EXAMPLE_PROJECTS: Dict[str, Path] = {}

def ensure_miktex_auto_install() -> None:
    """Best-effort: disable MiKTeX package install popups.

    MiKTeX can prompt with GUI dialogs for missing packages, which will stall
    automated compilation and cause timeouts. We try to configure MiKTeX to
    auto-install packages without asking.
    """
    try:
        # `initexmf` is the MiKTeX config tool. If it isn't present, ignore.
        subprocess.run(
            ["initexmf", "--set-config-value=[MPM]AutoInstall=1"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        subprocess.run(
            ["initexmf", "--set-config-value=[MPM]AskInstall=0"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        pass

def _is_valid_pdf(path: Path) -> bool:
    """Cheap PDF integrity check: header + EOF marker + minimum size."""
    try:
        if not path.exists() or path.stat().st_size < 10_000:
            return False
        data = path.read_bytes()
        if not data.startswith(b"%PDF"):
            return False
        return b"%%EOF" in data[-2048:]
    except Exception:
        return False


def _latex_escape(text: str) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def personalize_cv_header(app_dir: Path) -> None:
    """Replace YOUR_* placeholders in app_dir/cv-llt.tex using life-story/profile."""
    cv_tex = app_dir / "cv-llt.tex"
    if not cv_tex.exists():
        return

    person = load_person()
    first, last = split_name(person.full_name or "")

    linkedin_handle = (person.linkedin or "").strip()
    if linkedin_handle.startswith("http"):
        linkedin_handle = linkedin_handle.rstrip("/").split("/")[-1]

    github_handle = (person.github or "").strip()
    if github_handle.startswith("http"):
        github_handle = github_handle.rstrip("/").split("/")[-1]

    replacements = {
        "YOUR_FIRST_NAME": _latex_escape(first or person.full_name),
        "YOUR_LAST_NAME": _latex_escape(last),
        "YOUR_EMAIL": _latex_escape(person.email),
        "YOUR_LINKEDIN_HANDLE": _latex_escape(linkedin_handle),
        "YOUR_GITHUB": _latex_escape(github_handle),
        "YOUR_LASTNAME": _latex_escape(last),
        "YOUR_FIRSTNAME": _latex_escape(first),
        "YOUR_MIDDLENAME_OR_INITIAL": "",
    }

    content = cv_tex.read_text(encoding="utf-8", errors="replace")
    for k, v in replacements.items():
        content = content.replace(k, v)

    # If the user didn't provide a photo in this application folder, hide the photo block.
    has_photo = (app_dir / "photo.png").exists() or (app_dir / "photo.jpg").exists()
    if not has_photo:
        content = content.replace(r"\includecomment{fullonly}", r"\excludecomment{fullonly}")

    cv_tex.write_text(content, encoding="utf-8")


def _looks_like_placeholder(tex: str) -> bool:
    markers = [
        "Your Most Recent Job Title",
        "Ph.D. in YOUR FIELD",
        "YOUR FIELD",
        "Project Name",
        "YOUR_GITHUB/PROJECT",
        "Your domain-specific technical skills here",
    ]
    t = tex or ""
    if any(m in t for m in markers):
        return True
    # Also treat known-bad curve entry formatting as "needs regen"
    # (LLM sometimes emits \entry*[DATE]{...} which breaks curve template expectations).
    if re.search(r"\\entry\*\[[^\]]*\]\s*\{", t):
        return True
    return False


def _generate_base_rubric(*, rubric_name: str, life_story: str, model: str) -> str:
    system = (
        "You generate LaTeX rubric content for the curve CV template.\n"
        "Return LaTeX only (no markdown, no explanations).\n"
        "Must compile inside a file like employment.tex/education.tex/skills.tex/projects.tex.\n"
        "Use this exact structure:\n"
        "\\begin{rubric}{<Title>} ... \\end{rubric}\n"
        "Use \\entry*[DATE] ... for items.\n"
        "Do NOT invent degrees, companies, dates, or achievements. Only use what is explicitly in LIFE STORY.\n"
        "Escape LaTeX special characters when needed (e.g., %, &, _).\n"
    )

    prompt = (
        f"LIFE STORY:\n{life_story}\n\n"
        f"Task: Generate the '{rubric_name}' rubric file content.\n"
        "If the life story does not contain enough info for a section, keep it minimal but valid.\n"
    )
    return generate_latex(prompt=prompt, system=system, model=model, temperature=0.2, max_tokens=1800, timeout=600)


def _extract_section(text: str, header: str) -> str:
    # Extract markdown section content between "## Header" and next "## ".
    pattern = rf"^##\s+{re.escape(header)}\s*$"
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(pattern, ln.strip(), flags=re.IGNORECASE):
            start = i + 1
            break
    if start is None:
        return ""
    out: list[str] = []
    for ln in lines[start:]:
        if ln.strip().startswith("## "):
            break
        out.append(ln)
    return "\n".join(out).strip()


def _md_bullets(block: str) -> list[str]:
    bullets: list[str] = []
    for ln in (block or "").splitlines():
        s = ln.strip()
        if s.startswith("- "):
            bullets.append(s[2:].strip())
    return bullets


def _parse_work_experience(text: str) -> list[dict]:
    # Expect entries like:
    # ### Role — Org, Location
    # **July 2025 – August 2025**
    # paragraph(s)
    # - bullets...
    # **Technologies:** ...
    block = _extract_section(text, "Work Experience")
    if not block:
        return []
    lines = block.splitlines()
    entries: list[dict] = []
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("### "):
            title = ln[4:].strip()
            i += 1
            date = ""
            if i < len(lines) and lines[i].strip().startswith("**") and lines[i].strip().endswith("**"):
                date = lines[i].strip().strip("*").strip()
                i += 1
            body_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("### "):
                body_lines.append(lines[i])
                i += 1
            body = "\n".join(body_lines).strip()
            tech = ""
            m = re.search(r"\*\*Technologies:\*\*\s*(.+)", body)
            if m:
                tech = m.group(1).strip()
            bullets = _md_bullets(body)
            # Remove the technologies line from bullets if present
            bullets = [b for b in bullets if not b.lower().startswith("technologies:")]
            # First non-empty paragraph sentence as context (optional)
            paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            context = ""
            if paras:
                context = re.sub(r"\*\*Technologies:\*\*.*", "", paras[0]).strip()
            entries.append({"title": title, "date": date, "context": context, "bullets": bullets, "tech": tech})
        else:
            i += 1
    return entries


def _parse_education(text: str) -> list[dict]:
    block = _extract_section(text, "Education")
    if not block:
        return []
    lines = block.splitlines()
    entries: list[dict] = []
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("### "):
            title = ln[4:].strip()
            i += 1
            date = ""
            if i < len(lines) and lines[i].strip().startswith("**") and lines[i].strip().endswith("**"):
                date = lines[i].strip().strip("*").strip()
                i += 1
            body_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("### "):
                body_lines.append(lines[i])
                i += 1
            body = "\n".join(body_lines).strip()
            bullets = _md_bullets(body)
            entries.append({"title": title, "date": date, "bullets": bullets})
        else:
            i += 1
    return entries


def _parse_projects(text: str) -> list[dict]:
    block = _extract_section(text, "Projects")
    if not block:
        return []
    lines = block.splitlines()
    entries: list[dict] = []
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("### "):
            name = ln[4:].strip()
            i += 1
            body_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("### "):
                body_lines.append(lines[i])
                i += 1
            body = "\n".join(body_lines).strip()
            bullets = _md_bullets(body)
            code = ""
            tech = ""
            what = ""
            for b in bullets:
                if b.lower().startswith("code:"):
                    code = b.split(":", 1)[1].strip()
                elif b.lower().startswith("technologies:"):
                    tech = b.split(":", 1)[1].strip()
                elif b.lower().startswith("what it does:"):
                    what = b.split(":", 1)[1].strip()
            # Key achievements line (optional)
            achievements = ""
            for b in bullets:
                if b.lower().startswith("key achievements:"):
                    achievements = b.split(":", 1)[1].strip()
            entries.append({"name": name, "what": what, "achievements": achievements, "tech": tech, "code": code})
        else:
            i += 1
    return entries


def _parse_skills(text: str) -> list[tuple[str, str]]:
    block = _extract_section(text, "Skills")
    if not block:
        return []
    lines = block.splitlines()
    out: list[tuple[str, str]] = []
    current = ""
    items: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("### "):
            if current and items:
                out.append((current, ", ".join(items)))
            current = s[4:].strip()
            items = []
        elif s.startswith("- "):
            # skill line sometimes has "X — note"
            items.append(s[2:].split("—", 1)[0].strip())
    if current and items:
        out.append((current, ", ".join(items)))
    return out


def _entry(date: str, body_lines: list[str]) -> str:
    # Curve rubric entry format expected by template:
    # \entry*[DATE]%
    #     line
    #     \par line
    date = (date or "").replace("–", "--")
    out = [rf"\entry*[{_latex_escape(date)}]%"]
    for idx, ln in enumerate(body_lines):
        if idx == 0:
            out.append(f"    {ln}")
        else:
            out.append(f"    \\par {ln}")
    return "\n".join(out)


def render_employment_from_life_story(text: str) -> str:
    entries = _parse_work_experience(text)
    lines = [r"\begin{rubric}{Experience}", ""]
    if not entries:
        lines += [r"\end{rubric}", ""]
        return "\n".join(lines).strip()
    for e in entries:
        title = e["title"]
        # Split "Role — Org, ..." if present
        role = title
        org = ""
        if "—" in title:
            role, org = [p.strip() for p in title.split("—", 1)]
        head = rf"\textbf{{{_latex_escape(role)},}} {_latex_escape(org)}.".strip()
        body: list[str] = [head]
        if e.get("context"):
            body.append(_latex_escape(e["context"]))
        for b in e.get("bullets", [])[:6]:
            body.append(rf"- {_latex_escape(b)}")
        if e.get("tech"):
            body.append(rf"Technologies: {_latex_escape(e['tech'])}.")
        lines.append(_entry(e.get("date", ""), body))
        lines.append("")
    lines.append(r"\end{rubric}")
    return "\n".join(lines).strip() + "\n"


def render_education_from_life_story(text: str) -> str:
    entries = _parse_education(text)
    lines = [r"\begin{rubric}{Education}", ""]
    for e in entries:
        title = _latex_escape(e["title"])
        body = [rf"\textbf{{{title}}}"]
        for b in e.get("bullets", [])[:6]:
            body.append(rf"- {_latex_escape(b)}")
        lines.append(_entry(e.get("date", ""), body))
        lines.append("")
    lines.append(r"\end{rubric}")
    return "\n".join(lines).strip() + "\n"


def render_skills_from_life_story(text: str) -> str:
    cats = _parse_skills(text)
    lines = [r"\begin{rubric}{Skills}", ""]
    for cat, items in cats:
        lines.append(rf"\entry*[{_latex_escape(cat)}]%")
        lines.append(f"    {_latex_escape(items)}.")
        lines.append("")
    lines.append(r"\end{rubric}")
    return "\n".join(lines).strip() + "\n"


def render_projects_from_life_story(text: str) -> str:
    projs = _parse_projects(text)
    lines = [r"\begin{rubric}{Projects}", ""]
    for p in projs:
        body: list[str] = [rf"\textbf{{{_latex_escape(p['name'])}}} — {_latex_escape(p.get('what') or p.get('achievements') or '')}"]
        if p.get("tech"):
            body.append(rf"Technologies: {_latex_escape(p['tech'])}.")
        if p.get("code"):
            body.append(rf"\href{{{p['code']}}}{{\faGithub}}")
        lines.append(_entry("2025", body))
        lines.append("")
    lines.append(r"\end{rubric}")
    return "\n".join(lines).strip() + "\n"


def ensure_base_cv_content(cv_dir: Path, *, model: str = "qwen2.5:3b") -> None:
    """If cv_dir contains template placeholders, regenerate from life-story.md."""
    life_story_path = resolve_life_story_path(cv_dir)
    if not life_story_path.exists():
        return
    life_story = life_story_path.read_text(encoding="utf-8", errors="replace")

    renderers = {
        "employment.tex": render_employment_from_life_story,
        "education.tex": render_education_from_life_story,
        "skills.tex": render_skills_from_life_story,
        "projects.tex": render_projects_from_life_story,
    }

    for filename, renderer in renderers.items():
        path = cv_dir / filename
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8", errors="replace")
        if not _looks_like_placeholder(current):
            continue
        try:
            path.write_text(renderer(life_story), encoding="utf-8")
            logger.info("Regenerated base %s from life-story.md", filename)
        except Exception as e:
            logger.warning("Failed to regenerate base %s: %s", filename, e)

def ensure_cv_scaffold(cv_dir: Path) -> None:
    """Ensure cv_dir contains the minimum required template files.

    The repo ships templates in ./cv_templates, but the working cv/ folder is user-owned
    (often gitignored). If cv/ is missing, copy the templates once.
    """
    templates_dir = _PROJECT_ROOT / "cv_templates"
    if not templates_dir.exists():
        return

    cv_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "cv-llt-template.tex": "cv-llt.tex",
        "employment-template.tex": "employment.tex",
        "skills-template.tex": "skills.tex",
        "projects-template.tex": "projects.tex",
        "education-template.tex": "education.tex",
        "publications-template.tex": "publications.tex",
        "own-bib.bib": "own-bib.bib",
        "settings.sty": "settings.sty",
    }

    for src_name, dest_name in mapping.items():
        src = templates_dir / src_name
        dest = cv_dir / dest_name
        if src.exists() and not dest.exists():
            shutil.copy2(str(src), str(dest))

    # Ensure optional referenced files exist so compilation doesn't fail.
    for optional in ["publications.tex", "own-bib.bib"]:
        dest = cv_dir / optional
        if not dest.exists():
            try:
                dest.write_text("", encoding="utf-8")
            except Exception:
                pass

    # Life story template (only if user doesn't already have one in project root)
    project_life = _PROJECT_ROOT / "life-story.md"
    if not project_life.exists():
        src = templates_dir / "life_story_template.md"
        dest = cv_dir / "life-story.md"
        if src.exists() and not dest.exists():
            shutil.copy2(str(src), str(dest))


def resolve_cv_dir(profile: Optional[dict] = None) -> Path:
    """Return the CV directory from profile config, falling back to ./cv."""
    raw = (profile or {}).get("pipeline", {}).get("cv_dir", "")
    if raw:
        return Path(os.path.expanduser(raw))
    return _DEFAULT_CV_DIR


def resolve_life_story_path(cv_dir: Path) -> Path:
    """Return the life-story.md path — project root takes priority over cv_dir."""
    project = _PROJECT_ROOT / "life-story.md"
    return project if project.exists() else cv_dir / "life-story.md"


# ---------------------------------------------------------------------------
# Back-compat module-level accessors (used by cmd_customize in main.py)
# ---------------------------------------------------------------------------
CV_DIR = _DEFAULT_CV_DIR
LIFE_STORY_PATH = resolve_life_story_path(CV_DIR)


def _read_file(path: Path) -> str:
    """Read a file, return empty string if not found."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("File not found: %s", path)
        return ""


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s]+', '-', text)
    return text[:60]


def _extract_user_name(life_story: str) -> str:
    """Extract the user's full name from the life story."""
    for line in life_story.splitlines()[:20]:
        m = re.search(r'\*\*Full Name:\*\*\s*(.+)', line)
        if m:
            return m.group(1).strip()
    # Fallback: first H1 heading
    m = re.search(r'^#\s+Life Story\s*[—–-]\s*(.+)', life_story, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return "the candidate"


def analyze_job(description: str, title: str = "", company: str = "", model: str = "qwen3.5:9b") -> Dict:
    """Analyze a job description to extract domain, key skills, and keywords.

    Returns dict with: domain, key_technologies, keywords, focus_areas
    """
    prompt = f"""Analyze this job posting and return a JSON object.

Job Title: {title}
Company: {company}

Job Description:
{description[:3000]}

Return JSON with these fields:
- "domain": one of ["3d_vision", "robotics_perception", "vlm_multimodal", "generative_ai", "nlp", "general_ml", "autonomous_driving"]
- "key_technologies": list of 5-10 specific technologies/frameworks mentioned or implied
- "keywords": list of 5-10 important keywords for this role
- "focus_areas": list of 3-5 main focus areas of the role
- "company_mission": one sentence about what the company does (if inferrable)
"""
    raw = generate_structured(prompt, model=model)
    # Some models may return a JSON list; normalize to dict.
    if isinstance(raw, list) and raw:
        raw = raw[0] if isinstance(raw[0], dict) else {}
    if not isinstance(raw, dict) or not raw:
        raw = {}
    return {
        "domain": raw.get("domain", "general_ml"),
        "key_technologies": raw.get("key_technologies", []) or [],
        "keywords": raw.get("keywords", []) or [],
        "focus_areas": raw.get("focus_areas", []) or [],
        "company_mission": raw.get("company_mission", "") or "",
    }


def generate_employment_tex(
    life_story: str,
    job_analysis: Dict,
    base_template: str,
    model: str = "qwen3.5:9b",
) -> str:
    """Generate a customized employment.tex for a specific job."""

    # Pick the closest few-shot example
    domain = job_analysis.get("domain", "general_ml")
    example_path = EXAMPLE_EMPLOYMENT.get(domain)
    if not example_path:
        example_path = EXAMPLE_EMPLOYMENT.get("3d_vision")
    example = _read_file(example_path) if example_path and example_path.exists() else ""

    user_name = _extract_user_name(life_story)

    system = f"""You are an expert CV writer helping {user_name} apply for jobs.
You produce LaTeX code using the 'curve' document class rubric format.
You MUST output ONLY valid LaTeX — no markdown, no explanations, no code fences.
The output must compile with pdflatex without errors."""

    prompt = f"""Customize the employment section of {user_name}'s CV for this specific job.

TARGET JOB ANALYSIS:
- Domain: {job_analysis.get('domain', 'general_ml')}
- Key Technologies: {', '.join(job_analysis.get('key_technologies', []))}
- Focus Areas: {', '.join(job_analysis.get('focus_areas', []))}
- Keywords: {', '.join(job_analysis.get('keywords', []))}

MASTER SOURCE OF TRUTH ({user_name}'s full background):
{life_story[:6000]}

BASE TEMPLATE (default employment.tex — keep ALL entries present here):
{base_template}

{"EXAMPLE of a customized version for a similar domain:" if example else ""}
{example}

RULES:
1. Keep ALL job entries that appear in the base template — do not add or remove entries
2. Keep the exact same dates and job titles from the base template
3. HEAVILY customize the most recent/relevant entry to emphasize technologies relevant to the target job
4. Lightly adjust other entries to emphasize relevant aspects
5. Use \\textbf{{}} for key achievements and metrics
6. Use \\par to start description paragraphs
7. Escape special LaTeX characters: & → \\&, % → \\%, $ → use math mode
8. Use the EXACT format: \\begin{{rubric}}{{Experience}} ... \\entry*[dates]% ... \\end{{rubric}}
9. Do NOT add any text outside the rubric environment

Output the complete employment.tex content:"""

    result = generate_latex(prompt, system=system, model=model, max_tokens=3000)

    # Validate basic structure
    if "\\begin{rubric}" not in result or "\\end{rubric}" not in result:
        logger.warning("Generated employment.tex missing rubric structure, using base template")
        return base_template

    return result


def generate_skills_tex(
    life_story: str,
    job_analysis: Dict,
    base_template: str,
    model: str = "qwen3.5:9b",
) -> str:
    """Generate a customized skills.tex for a specific job."""

    domain = job_analysis.get("domain", "general_ml")
    example_path = EXAMPLE_SKILLS.get(domain)
    if not example_path:
        example_path = EXAMPLE_SKILLS.get("perception")
    example = _read_file(example_path) if example_path and example_path.exists() else ""

    user_name = _extract_user_name(life_story)

    system = f"""You are an expert CV writer helping {user_name} apply for jobs.
You produce LaTeX code using the 'curve' document class rubric format.
Output ONLY valid LaTeX — no markdown, no explanations, no code fences."""

    prompt = f"""Customize the skills section of {user_name}'s CV for this specific job.

TARGET JOB ANALYSIS:
- Domain: {job_analysis.get('domain', 'general_ml')}
- Key Technologies: {', '.join(job_analysis.get('key_technologies', []))}
- Focus Areas: {', '.join(job_analysis.get('focus_areas', []))}

AHMED'S FULL SKILLS (from life-story):
{life_story[:4000]}

BASE TEMPLATE:
{base_template}

{"EXAMPLE of a customized version for a similar domain:" if example else ""}
{example}

RULES:
1. Keep 4-5 skill categories using \\entry*[Category Name]%
2. REORDER categories so the most job-relevant one comes first (after Programming)
3. RENAME categories to match the job domain (e.g., "Computer Vision & 3D" for vision roles, "Vision & Language Models" for VLM roles)
4. Within each category, put the most job-relevant skills FIRST
5. Always keep Programming as the first entry
6. Always keep Awards as the last entry (keep awards content unchanged)
7. Use \\& for ampersand, \\LaTeX for LaTeX
8. Use the EXACT format: \\begin{{rubric}}{{Skills}} ... \\entry*[...]% ... \\end{{rubric}}

Output the complete skills.tex content:"""

    result = generate_latex(prompt, system=system, model=model, max_tokens=2000)

    if "\\begin{rubric}" not in result or "\\end{rubric}" not in result:
        logger.warning("Generated skills.tex missing rubric structure, using base template")
        return base_template

    return result


def generate_projects_tex(
    life_story: str,
    job_analysis: Dict,
    base_template: str,
    model: str = "qwen3.5:9b",
) -> str:
    """Generate a customized projects.tex for a specific job."""

    user_name = _extract_user_name(life_story)

    system = f"""You are an expert CV writer helping {user_name} apply for jobs.
You produce LaTeX code using the 'curve' document class rubric format.
Output ONLY valid LaTeX — no markdown, no explanations, no code fences."""

    prompt = f"""Customize the projects section of {user_name}'s CV for this specific job.

TARGET JOB:
- Domain: {job_analysis.get('domain', 'general_ml')}
- Key Technologies: {', '.join(job_analysis.get('key_technologies', []))}
- Focus Areas: {', '.join(job_analysis.get('focus_areas', []))}

ALL AVAILABLE PROJECTS (from life-story):
{life_story[:4000]}

BASE TEMPLATE (current projects.tex):
{base_template}

RULES:
1. Select the 3-4 MOST RELEVANT projects for the target job
2. Order them by relevance (most relevant first)
3. Lightly adjust descriptions to emphasize job-relevant aspects
4. Keep \\href{{}}{{\\faGithub}} links intact
5. Use \\entry*[year]% format
6. Use \\begin{{rubric}}{{Projects}} ... \\end{{rubric}}

Output the complete projects.tex content:"""

    result = generate_latex(prompt, system=system, model=model, max_tokens=2000)

    if "\\begin{rubric}" not in result or "\\end{rubric}" not in result:
        logger.warning("Generated projects.tex missing rubric structure, using base template")
        return base_template

    return result


def validate_latex(content: str) -> bool:
    """Basic validation of LaTeX content."""
    # Check matching rubric environment
    opens = content.count("\\begin{rubric}")
    closes = content.count("\\end{rubric}")
    if opens != closes or opens == 0:
        return False

    # Check for common issues
    if content.count("{") != content.count("}"):
        # Allow small mismatches from escaped braces
        diff = abs(content.count("{") - content.count("}"))
        if diff > 2:
            return False

    return True


def create_application_dir(slug: str, cv_dir: Path) -> Path:
    """Create an application directory with copies and symlinks."""
    applications_dir = cv_dir / "applications"
    dest = applications_dir / slug
    if dest.exists():
        logger.info("Application dir already exists: %s", dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    # Copy the main CV tex file
    for f in ["cv-llt.tex"]:
        src = cv_dir / f
        if src.exists():
            shutil.copy2(str(src), str(dest / f))

    # Create symlinks back to cv_dir for stable (non-customized) files
    for f in SYMLINK_FILES:
        link = dest / f
        target = cv_dir / f
        if not link.exists() and target.exists():
            try:
                link.symlink_to(target)
            except OSError as e:
                # Windows often blocks symlinks without admin/developer-mode.
                # Fall back to copying to keep the build working.
                logger.warning("Failed to create symlink %s: %s (copying instead)", f, e)
                try:
                    shutil.copy2(str(target), str(link))
                except Exception as copy_err:
                    logger.warning("Failed to copy %s: %s", f, copy_err)

    return dest


def compile_latex(directory: Path) -> Optional[str]:
    """Compile LaTeX to PDF in the given directory. Returns PDF path or None."""
    tex_file = directory / "cv-llt.tex"
    if not tex_file.exists():
        logger.error("No cv-llt.tex found in %s", directory)
        return None

    def _cleanup():
        for ext in [".aux", ".bbl", ".bcf", ".blg", ".fdb_latexmk", ".fls",
                    ".log", ".out", ".run.xml", ".synctex.gz", ".toc"]:
            aux = directory / ("cv-llt" + ext)
            if aux.exists():
                try:
                    aux.unlink()
                except Exception:
                    pass

    pdf_path = directory / "cv-llt.pdf"

    ensure_miktex_auto_install()

    # Prefer latexmk if available (fast, handles refs), but on Windows MiKTeX it may require Perl.
    try:
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "cv-llt.tex"],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if _is_valid_pdf(pdf_path):
            _cleanup()
            return str(pdf_path)
        stderr = (result.stderr or "")[-2000:]
        if "script engine 'perl'" not in stderr.lower():
            logger.error("PDF not generated. LaTeX output:\n%s", stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("latexmk unavailable/failed (%s). Falling back to pdflatex.", e)

    # Fallback: run pdflatex directly (no Perl).
    # Try MiKTeX common path first; else rely on PATH.
    miktex_bin = Path(r"C:\Users\Admin\AppData\Local\Programs\MiKTeX\miktex\bin\x64")
    pdflatex = miktex_bin / "pdflatex.exe"
    pdflatex_cmd = str(pdflatex) if pdflatex.exists() else "pdflatex"
    try:
        # Run twice for references.
        last = None
        for _ in range(2):
            last = subprocess.run(
                [pdflatex_cmd, "-interaction=nonstopmode", "cv-llt.tex"],
                cwd=str(directory),
                capture_output=True,
                text=True,
                timeout=900,
            )
        if _is_valid_pdf(pdf_path):
            _cleanup()
            return str(pdf_path)
        # Remove corrupt/partial PDF if it exists
        if pdf_path.exists() and not _is_valid_pdf(pdf_path):
            try:
                pdf_path.unlink()
            except Exception:
                pass
        if last is not None:
            try:
                (directory / "cv-compile.stdout.txt").write_text(last.stdout or "", encoding="utf-8", errors="replace")
                (directory / "cv-compile.stderr.txt").write_text(last.stderr or "", encoding="utf-8", errors="replace")
            except Exception:
                pass
            tail = ((last.stderr or last.stdout or "")[-2500:]).strip()
            if tail:
                logger.error("pdflatex output (tail):\n%s", tail)
        logger.error("PDF not generated via pdflatex either.")
        return None
    except Exception as e:
        logger.error("pdflatex failed: %s", e)
        if pdf_path.exists() and not _is_valid_pdf(pdf_path):
            try:
                pdf_path.unlink()
            except Exception:
                pass
        return None


def customize_cv_for_job(
    job_url: str,
    title: str,
    company: str,
    location: str,
    description: str,
    model: str = "qwen3.5:9b",
    profile: Optional[dict] = None,
) -> Optional[Dict]:
    """Full CV customization pipeline for a job.

    Returns dict with: slug, cv_pdf_path, app_dir, or None on failure.
    """
    if not check_ollama_available():
        logger.error(
            "Ollama is not running. CV customization requires a local LLM.\n"
            "Install and start it: bash setup_ollama.sh"
        )
        return None

    cv_dir = resolve_cv_dir(profile)
    ensure_cv_scaffold(cv_dir)
    # Ensure base files are real (not placeholders) before tailoring per-job.
    ensure_base_cv_content(cv_dir, model=model)
    life_story_path = resolve_life_story_path(cv_dir)

    # Load master content
    life_story = _read_file(life_story_path)
    if not life_story:
        logger.error(
            "life-story.md not found. Expected at: %s\n"
            "Copy cv_templates/life_story_template.md to that path and fill it in.",
            life_story_path,
        )
        return None

    base_employment = _read_file(cv_dir / "employment.tex")
    base_skills = _read_file(cv_dir / "skills.tex")
    base_projects = _read_file(cv_dir / "projects.tex")

    # Create slug
    slug = _slugify(f"{company}-{title}")
    if not slug:
        slug = _slugify(company or "unknown")

    logger.info("Customizing CV for: %s at %s (slug: %s)", title, company, slug)

    # Step 1: Analyze job
    logger.info("Step 1: Analyzing job description...")
    job_analysis = analyze_job(description, title, company, model=model)
    logger.info("Job domain: %s", job_analysis.get("domain"))

    # Step 2: Generate customized LaTeX files with retry
    for attempt in range(3):
        logger.info("Step 2: Generating LaTeX (attempt %d)...", attempt + 1)

        employment_tex = generate_employment_tex(life_story, job_analysis, base_employment, model=model)
        skills_tex = generate_skills_tex(life_story, job_analysis, base_skills, model=model)
        projects_tex = generate_projects_tex(life_story, job_analysis, base_projects, model=model)

        # Validate
        valid = all([
            validate_latex(employment_tex),
            validate_latex(skills_tex),
            validate_latex(projects_tex),
        ])

        if valid:
            break
        logger.warning("LaTeX validation failed on attempt %d", attempt + 1)

        if attempt == 2:
            logger.warning("Using base templates as fallback")
            employment_tex = base_employment
            skills_tex = base_skills
            projects_tex = base_projects

    # Step 3: Create application directory
    logger.info("Step 3: Creating application directory...")
    app_dir = create_application_dir(slug, cv_dir)

    # Write customized files
    (app_dir / "employment.tex").write_text(employment_tex, encoding="utf-8")
    (app_dir / "skills.tex").write_text(skills_tex, encoding="utf-8")
    (app_dir / "projects.tex").write_text(projects_tex, encoding="utf-8")
    personalize_cv_header(app_dir)

    # Write job description for reference
    jd_content = f"# {title} at {company}\n\n**Location:** {location}\n**URL:** {job_url}\n\n---\n\n{description}"
    (app_dir / "job-description.md").write_text(jd_content, encoding="utf-8")

    # Step 4: Compile to PDF
    logger.info("Step 4: Compiling LaTeX to PDF...")
    pdf_path = compile_latex(app_dir)

    if pdf_path:
        logger.info("CV generated: %s", pdf_path)
        return {
            "slug": slug,
            "cv_pdf_path": pdf_path,
            "app_dir": str(app_dir),
        }
    else:
        logger.error("PDF compilation failed for %s", slug)
        # Try with base templates
        logger.info("Retrying with base templates...")
        (app_dir / "employment.tex").write_text(base_employment, encoding="utf-8")
        (app_dir / "skills.tex").write_text(base_skills, encoding="utf-8")
        (app_dir / "projects.tex").write_text(base_projects, encoding="utf-8")
        pdf_path = compile_latex(app_dir)
        if pdf_path:
            logger.warning("Used base templates as fallback. PDF at: %s", pdf_path)
            return {
                "slug": slug,
                "cv_pdf_path": pdf_path,
                "app_dir": str(app_dir),
            }
        return None
