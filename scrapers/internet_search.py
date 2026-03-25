"""Internet-wide hiring scraper via DuckDuckGo search.

Finds job postings and hiring announcements across the web using broad hiring
signals and ATS/domain patterns, not limited to one platform.
"""

import logging
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from models import Job, JobBoard, SearchQuery

logger = logging.getLogger(__name__)

_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

FAMOUS_HIRING_SIGNALS = [
    '"we\'re hiring"',
    '"hiring"',
    '"job opening"',
    '"open role"',
    '"careers"',
    '"join our team"',
    '"apply now"',
    '"vacancy"',
    '"work with us"',
    '"greenhouse.io"',
    '"jobs.lever.co"',
    '"ashbyhq.com"',
    '"smartrecruiters.com"',
    '"workdayjobs.com"',
    '"boards.greenhouse.io"',
]

PORTAL_DOMAIN_TO_BOARD = {
    "stepstone": JobBoard.STEPSTONE,
    "indeed": JobBoard.INDEED,
    "glassdoor": JobBoard.GLASSDOOR,
    "linkedin": JobBoard.LINKEDIN,
    "remotive": JobBoard.REMOTIVE,
    "arbeitnow": JobBoard.ARBEITNOW,
    "themuse": JobBoard.THEMUSE,
    "himalayas": JobBoard.HIMALAYAS,
    "adzuna": JobBoard.ADZUNA,
    "greenhouse": JobBoard.GREENHOUSE,
    "lever": JobBoard.LEVER,
}


def _infer_board_from_url(url: str) -> JobBoard:
    host = urlparse(url).netloc.lower()
    for token, board in PORTAL_DOMAIN_TO_BOARD.items():
        if token in host:
            return board
    if "jobs.lever.co" in host:
        return JobBoard.LEVER
    if "boards.greenhouse.io" in host or "greenhouse.io" in host:
        return JobBoard.GREENHOUSE
    return JobBoard.INTERNET


def _is_probably_listing_page(url: str) -> bool:
    path = urlparse(url).path.lower()
    if not path or path in {"/", ""}:
        return False
    listing_markers = ["/jobs", "search", "stellenangebote", "careers", "vacancies", "positions"]
    return any(marker in path for marker in listing_markers)


def _looks_like_job_link(url: str, board: JobBoard) -> bool:
    parsed = urlparse(url)
    if not parsed.scheme.startswith("http"):
        return False
    path = parsed.path.lower()

    board_patterns = {
        JobBoard.STEPSTONE: ["/job/", "/stellenangebot", "/jobs/"],
        JobBoard.INDEED: ["/viewjob", "/rc/clk", "/jobs"],
        JobBoard.GLASSDOOR: ["joblisting", "-job.htm"],
        JobBoard.LINKEDIN: ["/jobs/view/", "/jobs/collections/"],
        JobBoard.GREENHOUSE: ["/jobs/", "/boards/"],
        JobBoard.LEVER: ["/jobs", "/lever.co/"],
    }
    patterns = board_patterns.get(board, ["/job", "/jobs", "/careers", "/position", "/vacancy"])
    return any(p in path for p in patterns)


def _extract_subjob_links(url: str, board: JobBoard, max_links: int = 8) -> list[str]:
    try:
        resp = requests.get(url, headers=_SEARCH_HEADERS, timeout=10, allow_redirects=True)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        return []

    links: list[str] = []
    seen: set[str] = set()
    base_host = urlparse(url).netloc.lower()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        absolute = urljoin(url, href)
        parsed = urlparse(absolute)
        if not parsed.scheme.startswith("http"):
            continue
        if parsed.netloc.lower() != base_host:
            continue
        if absolute in seen:
            continue
        if not _looks_like_job_link(absolute, board):
            continue
        seen.add(absolute)
        links.append(absolute)
        if len(links) >= max_links:
            break

    return links


def _extract_company_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    parts = [p for p in host.split(".") if p and p not in {"com", "io", "ai", "co", "org", "net"}]
    if not parts:
        return "Web"
    candidate = parts[0].replace("-", " ").strip()
    return candidate.title() if candidate else "Web"


def _clean_title(title: str, fallback: str) -> str:
    t = (title or "").strip()
    if not t:
        return fallback
    for sep in [" | ", " - ", " · ", " — "]:
        if sep in t:
            t = t.split(sep)[0].strip()
            break
    return t or fallback


def _extract_location(text: str) -> str:
    m = re.search(r"\b(remote|hybrid|on[\s-]?site)\b", text.lower())
    if m:
        return m.group(1)
    return ""


def _fetch_page_details(url: str) -> dict:
    try:
        resp = requests.get(url, headers=_SEARCH_HEADERS, timeout=10, allow_redirects=True)
        if resp.status_code != 200:
            return {}
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        logger.debug("Could not fetch page %s: %s", url, e)
        return {}

    result: dict = {}

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if title:
        result["title"] = title

    desc_tag = soup.find("meta", attrs={"name": "description"})
    if not desc_tag:
        desc_tag = soup.find("meta", attrs={"property": "og:description"})
    if desc_tag and desc_tag.get("content"):
        result["description"] = desc_tag.get("content", "").strip()

    date_tag = (
        soup.find("meta", attrs={"property": "article:published_time"})
        or soup.find("meta", attrs={"name": "date"})
        or soup.find("time")
    )
    if date_tag:
        content = date_tag.get("content") if hasattr(date_tag, "get") else None
        result["date_posted"] = (content or date_tag.get_text(" ", strip=True) or "").strip()

    return result


class InternetSearchScraper:
    """Scrape internet-wide job pages and hiring posts using DuckDuckGo."""

    def scrape(self, query: SearchQuery, max_results: int = 30) -> list[Job]:
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                logger.warning("ddgs not installed — run: pip install ddgs")
                return []

        jobs: list[Job] = []
        seen_urls: set[str] = set()

        base_terms = [query.keywords]
        if query.location:
            base_terms.append(query.location)

        signal_groups = [
            FAMOUS_HIRING_SIGNALS[:5],
            FAMOUS_HIRING_SIGNALS[5:10],
            FAMOUS_HIRING_SIGNALS[10:],
        ]

        for signals in signal_groups:
            if len(jobs) >= max_results:
                break
            signal_text = " OR ".join(signals)
            search_query = f"({signal_text}) {' '.join(base_terms)}"

            try:
                results = DDGS().text(search_query, max_results=max_results)
            except Exception as e:
                logger.warning("Internet search failed for '%s': %s", search_query, e)
                continue

            for r in results or []:
                if len(jobs) >= max_results:
                    break
                url = (r.get("href") or "").strip()
                if not url or url in seen_urls:
                    continue
                if not url.startswith("http"):
                    continue
                seen_urls.add(url)

                board = _infer_board_from_url(url)

                raw_title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                title = _clean_title(raw_title, query.keywords.title())
                company = _extract_company_from_url(url)
                location = _extract_location(f"{raw_title} {body}")

                job = Job(
                    title=title,
                    company=company,
                    location=location,
                    url=url,
                    board=board,
                    description=f"{raw_title}\n\n{body}".strip(),
                )
                jobs.append(job)

                if len(jobs) >= max_results:
                    continue

                if board != JobBoard.INTERNET and _is_probably_listing_page(url):
                    sub_links = _extract_subjob_links(url, board=board, max_links=6)
                    for sub_url in sub_links:
                        if len(jobs) >= max_results:
                            break
                        if sub_url in seen_urls:
                            continue
                        seen_urls.add(sub_url)
                        jobs.append(Job(
                            title=query.keywords.title(),
                            company=company,
                            location=location,
                            url=sub_url,
                            board=board,
                            description=f"Discovered from listing page: {url}",
                        ))

        logger.info("  internet: %d results for '%s'", len(jobs), query.keywords)
        return jobs

    def get_job_details(self, job: Job) -> Job:
        details = _fetch_page_details(job.url)
        if details.get("title"):
            job.title = _clean_title(details["title"], job.title)
        if details.get("description"):
            if job.description:
                job.description = f"{job.description}\n\n{details['description']}"
            else:
                job.description = details["description"]
        if details.get("date_posted"):
            job.date_posted = details["date_posted"]
        if not job.location:
            job.location = _extract_location(job.description or "")
        return job
