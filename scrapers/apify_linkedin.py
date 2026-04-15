"""Apify-based LinkedIn job scraper — used as fallback when JobSpy/guest API is blocked.

Requires APIFY_API_TOKEN in environment. Uses the free "LinkedIn Jobs Scraper" actor
(bebity/linkedin-jobs-scraper). Free tier: 100 actor compute units/month.

Sign up: https://apify.com  (free tier available)
Actor: https://apify.com/bebity/linkedin-jobs-scraper
"""

import logging
import os
import time
from typing import Optional

import requests

from models import Job, JobBoard, SearchQuery

logger = logging.getLogger(__name__)

APIFY_BASE = "https://api.apify.com/v2"
ACTOR_ID = "bebity/linkedin-jobs-scraper"
POLL_INTERVAL = 5   # seconds between status checks
MAX_WAIT = 300      # max seconds to wait for actor run


class ApifyLinkedInScraper:
    """LinkedIn scraper backed by Apify cloud actors.

    Falls back gracefully (returns []) when APIFY_API_TOKEN is not set or
    the actor run fails, so the main pipeline is never blocked.
    """

    def __init__(self):
        self.token = os.getenv("APIFY_API_TOKEN", "")
        self.session = requests.Session()

    def _available(self) -> bool:
        if not self.token:
            logger.debug("APIFY_API_TOKEN not set — Apify LinkedIn scraper disabled")
            return False
        return True

    def scrape(self, query: SearchQuery, max_results: int = 50) -> list[Job]:
        if not self._available():
            return []

        run_input = self._build_input(query, max_results)

        # Start actor run
        try:
            resp = self.session.post(
                f"{APIFY_BASE}/acts/{ACTOR_ID}/runs",
                params={"token": self.token},
                json=run_input,
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Apify LinkedIn: failed to start actor run: {e}")
            return []

        run_id = resp.json().get("data", {}).get("id")
        if not run_id:
            logger.error("Apify LinkedIn: no run ID in response")
            return []

        logger.info(f"Apify LinkedIn: actor run started ({run_id}), waiting for results...")

        # Poll until finished or timeout
        dataset_id = self._wait_for_run(run_id)
        if not dataset_id:
            return []

        # Fetch results
        return self._fetch_results(dataset_id, max_results)

    def _build_input(self, query: SearchQuery, max_results: int) -> dict:
        inp: dict = {
            "title": query.keywords,
            "location": query.location or "",
            "rows": min(max_results, 100),
            "scrapeCompany": False,
        }
        if query.remote:
            inp["workType"] = "remote"
        if query.job_type == "full-time":
            inp["contractType"] = "full_time"
        elif query.job_type == "contract":
            inp["contractType"] = "contract"
        if query.max_age_days <= 1:
            inp["publishedAt"] = "past-24h"
        elif query.max_age_days <= 7:
            inp["publishedAt"] = "past-week"
        elif query.max_age_days <= 30:
            inp["publishedAt"] = "past-month"
        return inp

    def _wait_for_run(self, run_id: str) -> Optional[str]:
        deadline = time.time() + MAX_WAIT
        while time.time() < deadline:
            try:
                resp = self.session.get(
                    f"{APIFY_BASE}/actor-runs/{run_id}",
                    params={"token": self.token},
                    timeout=15,
                )
                data = resp.json().get("data", {})
                status = data.get("status", "")
                if status == "SUCCEEDED":
                    return data.get("defaultDatasetId")
                if status in ("FAILED", "ABORTED", "TIMED-OUT"):
                    logger.error(f"Apify LinkedIn: run {status}")
                    return None
            except Exception as e:
                logger.warning(f"Apify LinkedIn: poll error: {e}")
            time.sleep(POLL_INTERVAL)

        logger.error(f"Apify LinkedIn: run timed out after {MAX_WAIT}s")
        return None

    def _fetch_results(self, dataset_id: str, max_results: int) -> list[Job]:
        try:
            resp = self.session.get(
                f"{APIFY_BASE}/datasets/{dataset_id}/items",
                params={"token": self.token, "limit": max_results, "format": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            items = resp.json()
        except Exception as e:
            logger.error(f"Apify LinkedIn: failed to fetch dataset: {e}")
            return []

        jobs = []
        for item in items:
            job = self._parse_item(item)
            if job:
                jobs.append(job)

        logger.info(f"  Apify LinkedIn: {len(jobs)} jobs found")
        return jobs

    def _parse_item(self, item: dict) -> Optional[Job]:
        url = item.get("jobUrl") or item.get("url") or ""
        title = item.get("title") or item.get("positionName") or ""
        if not url or not title:
            return None

        company = item.get("company") or item.get("companyName") or "Unknown"
        location = item.get("location") or item.get("place") or ""
        description = item.get("description") or item.get("descriptionHtml") or ""
        salary = item.get("salary") or ""
        date_posted = item.get("postedAt") or item.get("publishedAt") or ""

        return Job(
            title=title,
            company=company,
            location=location,
            url=url.split("?")[0],
            board=JobBoard.LINKEDIN,
            description=description,
            salary=salary,
            date_posted=date_posted,
        )

    def get_job_details(self, job: Job) -> Job:
        return job  # Apify actor returns full descriptions
