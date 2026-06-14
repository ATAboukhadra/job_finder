"""User profile extraction utilities.

Single source of truth:
- Prefer parsing `life-story.md` (human-authored, stable).
- Fall back to `profile.yaml` for any missing fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


_PROJECT_ROOT = Path(__file__).parent
DEFAULT_LIFE_STORY_PATH = _PROJECT_ROOT / "life-story.md"
DEFAULT_PROFILE_YAML_PATH = _PROJECT_ROOT / "profile.yaml"


@dataclass(frozen=True)
class Person:
    full_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    website: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _first_nonempty(*values: str) -> str:
    for v in values:
        v = (v or "").strip()
        if v:
            return v
    return ""


def _parse_life_story(text: str) -> Person:
    # Matches lines like: "- **Email:** someone@example.com"
    def get_field(label: str) -> str:
        pattern = rf"^\s*-\s*\*\*{re.escape(label)}:\*\*\s*(.+?)\s*$"
        m = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    full_name = get_field("Full Name")
    email = get_field("Email")
    phone = get_field("Phone")
    location = get_field("Location")
    linkedin = get_field("LinkedIn")
    github = get_field("GitHub")
    website = get_field("Website")

    # If phone line is missing but there is a likely phone number in the header block.
    if not phone:
        m = re.search(r"(\+\d{1,3}\s?\d[\d\s\-]{7,}\d)", text)
        if m:
            phone = m.group(1).strip()

    return Person(
        full_name=full_name,
        email=email,
        phone=phone,
        location=location,
        linkedin=linkedin,
        github=github,
        website=website,
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _load_profile_yaml(path: Path) -> dict:
    """Very small YAML reader for a few scalar fields.

    Avoids requiring PyYAML at runtime (the project may be run without it).
    This is NOT a general YAML parser; it only extracts simple `key: value` lines.
    """
    if not path.exists():
        return {}
    text = _read_text(path)
    if not text:
        return {}

    def find_top_level_scalar(key: str) -> str:
        m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, flags=re.MULTILINE)
        if not m:
            return ""
        v = m.group(1).strip().strip('"').strip("'")
        # ignore YAML comments if present
        v = v.split(" #", 1)[0].strip()
        return v

    # pipeline.email_recipient (very naive: just find the first email_recipient)
    email_recipient = ""
    m = re.search(r"^\s*email_recipient\s*:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if m:
        email_recipient = m.group(1).strip().strip('"').strip("'")
        email_recipient = email_recipient.split(" #", 1)[0].strip()

    return {
        "name": find_top_level_scalar("name"),
        "email": find_top_level_scalar("email"),
        "location": find_top_level_scalar("location"),
        "phone": find_top_level_scalar("phone"),
        "linkedin": find_top_level_scalar("linkedin"),
        "github": find_top_level_scalar("github"),
        "website": find_top_level_scalar("website"),
        "pipeline": {"email_recipient": email_recipient},
    }


def load_person(
    *,
    life_story_path: Path = DEFAULT_LIFE_STORY_PATH,
    profile_yaml_path: Path = DEFAULT_PROFILE_YAML_PATH,
) -> Person:
    """Load user identity/contact info."""
    life_story = _read_text(life_story_path)
    from_life = _parse_life_story(life_story) if life_story else Person()

    profile = _load_profile_yaml(profile_yaml_path)
    # Some profile.yaml files may not contain these keys; keep optional.
    from_profile = Person(
        full_name=str(profile.get("name", "") or "").strip(),
        email=str(profile.get("pipeline", {}).get("email_recipient", "") or profile.get("email", "") or "").strip(),
        phone=str(profile.get("phone", "") or "").strip(),
        location=str(profile.get("location", "") or "").strip(),
        linkedin=str(profile.get("linkedin", "") or "").strip(),
        github=str(profile.get("github", "") or "").strip(),
        website=str(profile.get("website", "") or "").strip(),
    )

    return Person(
        full_name=_first_nonempty(from_life.full_name, from_profile.full_name),
        email=_first_nonempty(from_life.email, from_profile.email),
        phone=_first_nonempty(from_life.phone, from_profile.phone),
        location=_first_nonempty(from_life.location, from_profile.location),
        linkedin=_first_nonempty(from_life.linkedin, from_profile.linkedin),
        github=_first_nonempty(from_life.github, from_profile.github),
        website=_first_nonempty(from_life.website, from_profile.website),
    )


def split_name(full_name: str) -> tuple[str, str]:
    """Best-effort split into first and last name."""
    parts = [p for p in (full_name or "").strip().split() if p]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])

