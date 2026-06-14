"""Arbeitnow API — free, no API key needed. EU + remote jobs."""

import logging
import re
from datetime import datetime, timedelta

import requests

from models import Job, JobBoard, SearchQuery

logger = logging.getLogger(__name__)

API_URL = "https://www.arbeitnow.com/api/job-board-api"


class ArbeitnowScraper:
    """Fetch jobs from Arbeitnow (free, no key). Strong EU coverage."""

    def scrape(self, query: SearchQuery, max_results: int = 50) -> list[Job]:
        jobs = []
        page = 1
        query_lower = query.keywords.lower()

        while len(jobs) < max_results:
            try:
                resp = requests.get(API_URL, params={"page": page}, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Arbeitnow API error: {e}")
                break

            items = data.get("data", [])
            if not items:
                break

            for item in items:
                if len(jobs) >= max_results:
                    break

                title = item.get("title", "")
                desc = _strip_html(item.get("description", ""))
                tags = " ".join(item.get("tags", []))
                searchable = f"{title} {desc} {tags}".lower()

                # Filter: must match at least one keyword from the query
                keywords = [w for w in query_lower.split() if len(w) >= 3]
                if not any(kw in searchable for kw in keywords):
                    continue

                # Filter by age
                created = item.get("created_at", "")
                if query.max_age_days and created:
                    try:
                        posted = datetime.fromtimestamp(int(created))
                        if datetime.now() - posted > timedelta(days=query.max_age_days):
                            continue
                    except (ValueError, TypeError):
                        pass

                # Location filter
                location = item.get("location", "")
                remote = item.get("remote", False)
                if query.location and not remote:
                    if query.location.lower() not in location.lower():
                        # Loose match — skip only if clearly wrong country
                        pass

                job = Job(
                    title=title,
                    company=item.get("company_name", "Unknown"),
                    location=location if location else ("Remote" if remote else ""),
                    url=item.get("url", ""),
                    board=JobBoard.ARBEITNOW,
                    description=desc,
                    salary="",
                    date_posted=datetime.fromtimestamp(int(created)).isoformat() if created else "",
                    job_type=", ".join(item.get("job_types", [])),
                    is_remote=bool(remote),
                )
                if job.url and job.title:
                    jobs.append(job)

            # Arbeitnow pagination: check for next link
            links = data.get("links", {})
            if not links.get("next"):
                break
            page += 1

        logger.info(f"  Arbeitnow: {len(jobs)} jobs found")
        return jobs[:max_results]

    def get_job_details(self, job: Job) -> Job:
        # API returns full descriptions
        return job


def _strip_html(text: str) -> str:
    """Basic HTML tag stripping."""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean
