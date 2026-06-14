"""Remotive API — free, no API key needed. Remote jobs only."""

import logging

import requests

from models import Job, JobBoard, SearchQuery

logger = logging.getLogger(__name__)

API_URL = "https://remotive.com/api/remote-jobs"

# Remotive category slugs relevant to our search
CATEGORY_MAP = {
    "software-dev": "software-dev",
    "data": "data",
    "machine-learning": "machine-learning",
    "devops": "devops",
}


class RemotiveScraper:
    """Fetch remote jobs from the Remotive API (free, no key)."""

    def scrape(self, query: SearchQuery, max_results: int = 50) -> list[Job]:
        jobs = []
        seen_urls: set[str] = set()

        # Remotive has a small index — try the full query first, then
        # fall back to individual broader keywords so we don't miss results.
        search_terms = [query.keywords]
        for word in query.keywords.lower().split():
            if len(word) >= 4 and word not in {"with", "that", "from", "this", "senior", "junior"}:
                search_terms.append(word)

        for term in search_terms:
            if len(jobs) >= max_results:
                break

            params = {
                "search": term,
                "limit": max_results,
            }

            try:
                resp = requests.get(API_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Remotive API error: {e}")
                continue

            for item in data.get("jobs", []):
                if len(jobs) >= max_results:
                    break
                url = item.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                job = Job(
                    title=item.get("title", ""),
                    company=item.get("company_name", "Unknown"),
                    location=item.get("candidate_required_location", "Remote"),
                    url=url,
                    board=JobBoard.REMOTIVE,
                    description=_strip_html(item.get("description", "")),
                    salary=item.get("salary", ""),
                    date_posted=item.get("publication_date", ""),
                    job_type=item.get("job_type", ""),
                    is_remote=True,  # Remotive lists remote jobs only
                )
                if job.title:
                    jobs.append(job)

        logger.info(f"  Remotive: {len(jobs)} jobs found")
        return jobs

    def get_job_details(self, job: Job) -> Job:
        # Remotive API already returns full descriptions
        return job


def _strip_html(text: str) -> str:
    """Basic HTML tag stripping."""
    import re
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean
