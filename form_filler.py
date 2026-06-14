"""Application form auto-filler instruction builder.

This module prepares form-filling instructions:
- static fields from your `life-story.md` (preferred) / `profile.yaml` (fallback)
- dynamic fields from pre-generated answers stored in SQLite

It does NOT submit anything by itself.
"""

import json
import logging
from typing import Dict, List, Optional

from storage import get_application_by_job
from user_profile import load_person, split_name

logger = logging.getLogger(__name__)

# Common field name patterns and what answer to map them to
def _static_field_mappings() -> Dict[str, str]:
    person = load_person()
    first, last = split_name(person.full_name)

    # Best-effort split of location into city/country.
    city, country = "", ""
    if person.location:
        parts = [p.strip() for p in person.location.split(",") if p.strip()]
        if len(parts) >= 2:
            city, country = parts[0], parts[-1]
        else:
            country = parts[0]

    # Keep a few common aliases that forms use.
    return {
        # Name
        "first_name": first,
        "last_name": last,
        "full_name": person.full_name,
        "name": person.full_name,

        # Contact
        "email": person.email,
        "phone": person.phone,
        "linkedin": person.linkedin,
        "github": person.github,
        "website": person.website,
        "portfolio": person.website,

        # Location
        "city": city,
        "country": country,
        "address": person.location,
        "location": person.location,
    }


def get_fill_instructions(job_url: str) -> Optional[Dict]:
    """Get auto-fill instructions for a job application.

    Returns a dict with:
    - static_fields: field name -> value (always the same)
    - dynamic_fields: question -> answer (from pre-generated form answers)
    - cv_pdf_path: path to customized CV for upload
    - cover_letter_pdf_path: path to cover letter for upload
    - application_url: the job URL to navigate to
    """
    app = get_application_by_job(job_url)
    if not app:
        logger.warning("No application found for %s", job_url)
        return None

    # Parse form answers
    answers = {}
    try:
        answers = json.loads(app.get("form_answers_json", "{}"))
    except json.JSONDecodeError:
        pass

    return {
        "application_url": job_url,
        "static_fields": _static_field_mappings(),
        "dynamic_fields": answers,
        "cv_pdf_path": app.get("cv_pdf_path", ""),
        "cover_letter_pdf_path": app.get("cover_letter_pdf_path", ""),
        "status": app.get("status", "unknown"),
        "slug": app.get("slug", ""),
    }


def format_fill_guide(instructions: Dict) -> str:
    """Format fill instructions as a human-readable guide for copy-paste."""
    lines = []
    lines.append("=" * 60)
    lines.append("APPLICATION FILL GUIDE")
    lines.append("=" * 60)
    lines.append(f"\nJob URL: {instructions['application_url']}")
    lines.append(f"CV PDF: {instructions['cv_pdf_path']}")
    lines.append(f"Cover Letter: {instructions['cover_letter_pdf_path']}")

    lines.append("\n--- STATIC FIELDS ---")
    for field, value in instructions["static_fields"].items():
        if value:
            lines.append(f"  {field}: {value}")

    lines.append("\n--- APPLICATION QUESTIONS ---")
    for question, answer in instructions["dynamic_fields"].items():
        lines.append(f"\n  Q: {question}")
        lines.append(f"  A: {answer}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
