"""Himalayas API — free, no API key needed. Remote-first tech jobs."""

import logging

import requests

from models import Job, JobBoard, SearchQuery

logger = logging.getLogger(__name__)

API_URL = "https://himalayas.app/jobs/api"


class HimalayasScraper:
    """Fetch remote jobs from Himalayas (free, no key)."""

    def scrape(self, query: SearchQuery, max_results: int = 50) -> list[Job]:
        jobs = []
        offset = 0
        per_page = 20  # Himalayas API max per page
        max_pages = 15  # Scan enough pages to find matches
        pages_fetched = 0
        query_lower = query.keywords.lower()

        while len(jobs) < max_results:
            params = {
                "limit": per_page,
                "offset": offset,
            }

            try:
                resp = requests.get(API_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Himalayas API error: {e}")
                break

            items = data.get("jobs", [])
            if not items:
                break

            for item in items:
                if len(jobs) >= max_results:
                    break

                title = item.get("title", "")
                desc = item.get("description", "") or item.get("excerpt", "")
                categories = " ".join(item.get("categories", []) + item.get("parentCategories", []))
                # Replace hyphens so "Machine-Learning-Engineer" matches "machine learning engineer"
                searchable = f"{title} {desc} {categories}".lower().replace("-", " ")

                # Filter: must match at least one keyword from the query
                keywords = [w for w in query_lower.split() if len(w) >= 3]
                if not any(kw in searchable for kw in keywords):
                    continue

                # Location restrictions check
                loc_restrictions = item.get("locationRestrictions", [])
                location_str = ", ".join(loc_restrictions) if loc_restrictions else "Remote"

                # Salary
                salary = ""
                min_sal = item.get("minSalary")
                max_sal = item.get("maxSalary")
                currency = item.get("currency", "")
                if min_sal and max_sal:
                    salary = f"{currency} {int(min_sal):,} - {int(max_sal):,}".strip()
                elif min_sal:
                    salary = f"From {currency} {int(min_sal):,}".strip()

                # Build URL from applicationLink or guid
                company = item.get("companyName", "Unknown")
                url = item.get("applicationLink", "")
                if not url:
                    guid = item.get("guid", "")
                    url = f"https://himalayas.app/jobs/{guid}" if guid else ""

                job = Job(
                    title=title,
                    company=company,
                    location=location_str,
                    url=url,
                    board=JobBoard.HIMALAYAS,
                    description=_strip_html(desc),
                    salary=salary,
                    date_posted=item.get("pubDate", ""),
                    job_type=item.get("employmentType", ""),
                    is_remote=True,  # Himalayas is remote-first / remote-only
                )
                if job.url and job.title:
                    jobs.append(job)

            offset += per_page
            pages_fetched += 1
            if len(items) < per_page or pages_fetched >= max_pages:
                break

        logger.info(f"  Himalayas: {len(jobs)} jobs found")
        return jobs[:max_results]

    def get_job_details(self, job: Job) -> Job:
        # API returns full descriptions
        return job


def _strip_html(text: str) -> str:
    """Basic HTML tag stripping."""
    import re
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean
