"""Graph-search crawler.

Given a (region, term) seed — e.g. UAE + "AI, computer vision" — this performs
a breadth-first crawl of the open web, the way a search engine indexes pages:

    seed search (news + web)        -> page nodes
    fetch page -> LLM/regex extract -> company & person nodes
    follow relevant outbound links  -> more page nodes (depth + 1)
    per company: resolve career page (Greenhouse / Lever / Ashby / generic)
                                    -> job nodes, scored against the profile

All results stream into the graph_* tables (see graph_storage) so the UI can
render progress live. Runs inside a background thread; see app.py.
"""

import logging
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import graph_storage as gs
import llm
from matcher import JobMatcher
from models import Job, JobBoard, classify_remote

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Portals / aggregators we don't want to treat as "companies" or follow deeply.
_SKIP_HOST_TOKENS = (
    "linkedin.", "facebook.", "twitter.", "x.com", "instagram.", "youtube.",
    "wikipedia.", "google.", "bing.", "duckduckgo.", "reddit.", "medium.",
    "glassdoor.", "indeed.", "crunchbase.", "bloomberg.", "reuters.",
    "gov.", "wikimedia.", "amazon.com", "apple.com/newsroom",
)

# Words that are never company names (filters regex-fallback noise).
_STOPWORD_NAMES = {
    "the", "and", "for", "with", "our", "your", "their", "this", "that",
    "ai", "artificial intelligence", "computer vision", "machine learning",
    "united arab emirates", "saudi arabia", "qatar", "uae", "middle east",
    "news", "home", "about", "contact", "careers", "jobs", "read more",
}


# --- Search -------------------------------------------------------------

def _ddgs():
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
            return DDGS
        except ImportError:
            logger.warning("ddgs not installed — run: pip install ddgs")
            return None


def search_text(query: str, max_results: int = 12) -> list[dict]:
    DDGS = _ddgs()
    if not DDGS:
        return []
    try:
        return list(DDGS().text(query, max_results=max_results) or [])
    except Exception as e:
        logger.warning("text search failed for '%s': %s", query, e)
        return []


def search_news(query: str, max_results: int = 10) -> list[dict]:
    DDGS = _ddgs()
    if not DDGS:
        return []
    try:
        return list(DDGS().news(query, max_results=max_results) or [])
    except Exception as e:
        logger.warning("news search failed for '%s': %s", query, e)
        return []


def seed_queries(region: str, term: str) -> list[tuple[str, str]]:
    """Return (kind, query) seed searches for a region + term."""
    return [
        ("web", f"{term} companies in {region}"),
        ("web", f"{region} {term} startups"),
        ("web", f"leading {term} companies {region}"),
        ("news", f"{term} {region}"),
        ("news", f"{region} artificial intelligence funding"),
    ]


# --- Page fetch ---------------------------------------------------------

def fetch_page(url: str, timeout: int = 12) -> dict:
    """Fetch a URL. Returns {title, text, links[]} (links = same-ish outbound http)."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", "text/html"):
            return {}
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        logger.debug("fetch failed %s: %s", url, e)
        return {}

    title = soup.title.get_text(" ", strip=True) if soup.title else url
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)

    links: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        absolute = urljoin(url, (a.get("href") or "").strip())
        p = urlparse(absolute)
        if not p.scheme.startswith("http"):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)

    return {"title": title, "text": text, "links": links}


# --- Entity extraction --------------------------------------------------

def _keyword_set(term: str) -> list[str]:
    """Expand a term like 'AI, computer vision' into match keywords."""
    base = [t.strip().lower() for t in re.split(r"[,/;]", term) if t.strip()]
    extra = []
    blob = term.lower()
    if "ai" in blob or "artificial" in blob:
        extra += ["ai", "artificial intelligence", "machine learning", "ml", "deep learning"]
    if "vision" in blob or "cv" in blob:
        extra += ["computer vision", "vision", "perception", "image", "detection"]
    out, seen = [], set()
    for kw in base + extra:
        if kw and kw not in seen:
            seen.add(kw)
            out.append(kw)
    return out or [term.lower()]


def extract_entities(text: str, region: str, term: str) -> dict:
    """Extract {companies:[{name,reason}], people:[{name,role,company}]} from page text.

    Uses the local LLM when Ollama is available, else a regex heuristic.
    """
    text = (text or "")[:3500]
    if len(text) < 80:
        return {"companies": [], "people": []}

    if llm.check_ollama_available():
        try:
            return _extract_llm(text, region, term)
        except Exception as e:
            logger.warning("LLM extraction failed, falling back to regex: %s", e)
    return _extract_regex(text)


def _extract_llm(text: str, region: str, term: str) -> dict:
    system = (
        "You extract organizations and notable people from text. "
        "Only return real, specific company/organization names and person names "
        "that actually appear in the text."
    )
    prompt = (
        f"From the text below, list companies/organizations working on or associated "
        f"with '{term}' in or near {region}, and any notable people mentioned "
        f"(founders, executives, researchers).\n\n"
        f'Return JSON: {{"companies": [{{"name": "...", "reason": "..."}}], '
        f'"people": [{{"name": "...", "role": "...", "company": "..."}}]}}\n\n'
        f"TEXT:\n{text}"
    )
    data = llm.generate_structured(prompt, system=system, model=llm.recommend_model(),
                                   max_tokens=1200)
    companies, people = [], []
    for c in (data.get("companies") or [])[:15]:
        name = (c.get("name") or "").strip() if isinstance(c, dict) else str(c).strip()
        if _valid_company(name):
            companies.append({"name": name, "reason": (c.get("reason") or "") if isinstance(c, dict) else ""})
    for p in (data.get("people") or [])[:10]:
        if isinstance(p, dict):
            name = (p.get("name") or "").strip()
        else:
            name = str(p).strip()
        if name and len(name) > 2 and name.lower() not in _STOPWORD_NAMES:
            people.append({
                "name": name,
                "role": (p.get("role") or "") if isinstance(p, dict) else "",
                "company": (p.get("company") or "") if isinstance(p, dict) else "",
            })
    return {"companies": companies, "people": people}


_COMPANY_SUFFIX = re.compile(
    r"\b([A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+){0,3}"
    r"\s+(?:AI|Labs|Technologies|Technology|Systems|Robotics|Solutions|"
    r"Inc|LLC|Ltd|Group|Analytics|Networks|Corp|Company))\b"
)


def _extract_regex(text: str) -> dict:
    """Heuristic company extraction: multi-word Capitalized phrases with company suffixes."""
    found, seen = [], set()
    for m in _COMPANY_SUFFIX.finditer(text):
        name = m.group(1).strip()
        key = name.lower()
        if _valid_company(name) and key not in seen:
            seen.add(key)
            found.append({"name": name, "reason": "pattern match"})
        if len(found) >= 12:
            break
    return {"companies": found, "people": []}


def _valid_company(name: str) -> bool:
    if not name or len(name) < 2 or len(name) > 60:
        return False
    low = name.lower().strip()
    if low in _STOPWORD_NAMES:
        return False
    if not re.search(r"[A-Za-z]", name):
        return False
    return True


# --- Career page resolution --------------------------------------------

def _first_path_seg(url: str) -> str:
    segs = [s for s in urlparse(url).path.split("/") if s]
    return segs[0] if segs else ""


def fetch_greenhouse(slug: str, keywords: list[str], limit: int = 15) -> list[dict]:
    try:
        r = requests.get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
                         params={"content": "true"}, headers=_HEADERS, timeout=12)
        r.raise_for_status()
        items = r.json().get("jobs", [])
    except Exception:
        return []
    return _filter_jobs([
        {
            "title": it.get("title", ""),
            "location": ((it.get("location") or {}).get("name")
                         or (it.get("offices") or [{}])[0].get("name", "")),
            "url": it.get("absolute_url", ""),
            "description": it.get("content", "") or "",
            "source": "greenhouse",
        }
        for it in items
    ], keywords, limit)


def fetch_lever(slug: str, keywords: list[str], limit: int = 15) -> list[dict]:
    try:
        r = requests.get(f"https://api.lever.co/v0/postings/{slug}",
                         params={"mode": "json"}, headers=_HEADERS, timeout=12)
        r.raise_for_status()
        items = r.json()
    except Exception:
        return []
    out = []
    for it in items:
        cats = it.get("categories") or {}
        out.append({
            "title": it.get("text", ""),
            "location": cats.get("location", "") or (cats.get("allLocations") or [""])[0],
            "url": it.get("hostedUrl", ""),
            "description": it.get("descriptionPlain") or it.get("description") or "",
            "source": "lever",
        })
    return _filter_jobs(out, keywords, limit)


def fetch_ashby(slug: str, keywords: list[str], limit: int = 15) -> list[dict]:
    try:
        r = requests.get(f"https://api.ashbyhq.com/posting-api/job-board/{slug}",
                         headers=_HEADERS, timeout=12)
        r.raise_for_status()
        items = r.json().get("jobs", [])
    except Exception:
        return []
    return _filter_jobs([
        {
            "title": it.get("title", ""),
            "location": it.get("location", "") or it.get("locationName", ""),
            "url": it.get("jobUrl") or it.get("applyUrl", ""),
            "description": it.get("descriptionPlain") or "",
            "source": "ashby",
        }
        for it in items
    ], keywords, limit)


def _filter_jobs(jobs: list[dict], keywords: list[str], limit: int) -> list[dict]:
    out, seen = [], set()
    for j in jobs:
        if not j.get("title") or not j.get("url") or j["url"] in seen:
            continue
        blob = f"{j['title']} {j.get('description','')}".lower()
        if keywords and not any(kw in blob for kw in keywords):
            continue
        seen.add(j["url"])
        out.append(j)
        if len(out) >= limit:
            break
    return out


def resolve_career_jobs(company: str, keywords: list[str]) -> list[dict]:
    """Find and scrape a company's open roles filtered by keywords."""
    results = search_text(f"{company} careers jobs", max_results=8)
    jobs: list[dict] = []
    checked_ats: set[str] = set()

    for r in results:
        url = (r.get("href") or r.get("url") or "").strip()
        if not url:
            continue
        host = urlparse(url).netloc.lower()

        if "greenhouse.io" in host:
            slug = _greenhouse_slug(url)
            if slug and slug not in checked_ats:
                checked_ats.add(slug)
                jobs += fetch_greenhouse(slug, keywords)
        elif "lever.co" in host:
            slug = _first_path_seg(url)
            if slug and slug not in checked_ats:
                checked_ats.add(slug)
                jobs += fetch_lever(slug, keywords)
        elif "ashbyhq.com" in host:
            slug = _first_path_seg(url)
            if slug and slug not in checked_ats:
                checked_ats.add(slug)
                jobs += fetch_ashby(slug, keywords)

        if len(jobs) >= 15:
            break

    # Generic careers-page fallback when no ATS matched
    if not jobs:
        for r in results[:3]:
            url = (r.get("href") or r.get("url") or "").strip()
            if url and re.search(r"career|jobs|join|vacan", url.lower()):
                jobs += _scrape_generic_careers(url, keywords)
                if jobs:
                    break

    # Dedup by url
    out, seen = [], set()
    for j in jobs:
        if j["url"] not in seen:
            seen.add(j["url"])
            out.append(j)
    return out[:15]


def _greenhouse_slug(url: str) -> str:
    m = re.search(r"[?&]for=([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    segs = [s for s in urlparse(url).path.split("/") if s and s not in ("embed", "boards")]
    return segs[0] if segs else ""


def _scrape_generic_careers(url: str, keywords: list[str], limit: int = 10) -> list[dict]:
    page = fetch_page(url)
    if not page:
        return []
    out, seen = [], set()
    base_host = urlparse(url).netloc.lower()
    for link in page.get("links", []):
        if urlparse(link).netloc.lower() != base_host:
            continue
        if not re.search(r"/job|/career|/position|/vacan|/opening|/roles?/", link.lower()):
            continue
        if link in seen:
            continue
        seen.add(link)
        out.append({"title": "", "location": "", "url": link,
                    "description": page.get("text", "")[:400], "source": "careers-page"})
        if len(out) >= limit:
            break
    return out


# --- Crawler ------------------------------------------------------------

class GraphCrawler:
    """Breadth-first regional opportunity crawler."""

    MAX_PAGES = 120
    MAX_LINKS_PER_PAGE = 5
    DELAY = 0.4  # politeness between fetches

    def __init__(self, run_id: int, region: str, term: str, profile: dict,
                 max_depth: int = 3, max_companies: int = 100):
        self.run_id = run_id
        self.region = region
        self.term = term
        self.max_depth = max_depth
        self.max_companies = max_companies
        self.keywords = _keyword_set(term)
        self.matcher = JobMatcher(profile)

        self.visited_urls: set[str] = set()
        self.seen_companies: set[str] = set()
        self.pages_fetched = 0
        self.n_companies = 0
        self.n_people = 0
        self.n_jobs = 0

    def _log(self, msg: str):
        logger.info("[graph %s] %s", self.run_id, msg)
        gs.append_log(self.run_id, msg)

    def _stats(self):
        gs.update_run(self.run_id, stats={
            "pages": self.pages_fetched, "companies": self.n_companies,
            "people": self.n_people, "jobs": self.n_jobs,
        })

    def run(self):
        try:
            self._log(f"Seeding crawl: {self.region} + '{self.term}' "
                      f"(depth={self.max_depth}, max_companies={self.max_companies})")
            region_id = gs.add_node(self.run_id, "region", self.region, depth=0)
            term_id = gs.add_node(self.run_id, "term", self.term, depth=0)
            gs.add_edge(self.run_id, region_id, term_id, "affiliated")

            frontier: deque = deque()

            # Seed searches -> page tasks at depth 1
            for kind, q in seed_queries(self.region, self.term):
                results = search_news(q) if kind == "news" else search_text(q)
                self._log(f"Search ({kind}): '{q}' -> {len(results)} results")
                for r in results:
                    url = (r.get("href") or r.get("url") or "").strip()
                    title = (r.get("title") or url).strip()
                    if not url or url in self.visited_urls or self._skip_host(url):
                        continue
                    self.visited_urls.add(url)
                    page_id = gs.add_node(self.run_id, "page", title, url=url, depth=1,
                                          discovered_from=term_id)
                    gs.add_edge(self.run_id, term_id, page_id, "mentions")
                    frontier.append(("page", url, title, 1, page_id))
                self._stats()

            # BFS
            while frontier:
                if self.pages_fetched >= self.MAX_PAGES:
                    self._log(f"Reached page cap ({self.MAX_PAGES}); stopping crawl.")
                    break
                kind, url, name, depth, node_id = frontier.popleft()
                if kind == "page":
                    self._process_page(url, name, depth, node_id, frontier)
                elif kind == "company":
                    self._process_company(name, node_id)
                self._stats()

            gs.update_run(self.run_id, status="completed",
                          finished_at=_now(), )
            self._stats()
            self._log(f"Done. {self.n_companies} companies, {self.n_people} people, "
                      f"{self.n_jobs} jobs across {self.pages_fetched} pages.")
        except Exception as e:
            logger.exception("Graph crawl failed")
            gs.append_log(self.run_id, f"ERROR: {e}")
            gs.update_run(self.run_id, status="failed", finished_at=_now())

    def _skip_host(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return any(tok in host for tok in _SKIP_HOST_TOKENS)

    def _process_page(self, url, name, depth, node_id, frontier):
        time.sleep(self.DELAY)
        page = fetch_page(url)
        self.pages_fetched += 1
        if not page:
            return
        self._log(f"Page [{depth}] {name[:60]}")

        ents = extract_entities(page.get("text", ""), self.region, self.term)
        for c in ents.get("companies", []):
            self._add_company(c.get("name", ""), c.get("reason", ""), depth, node_id, frontier)
        for p in ents.get("people", []):
            self._add_person(p, node_id)

        # Follow a few relevant outbound links one hop deeper
        if depth < self.max_depth:
            followed = 0
            for link in page.get("links", []):
                if followed >= self.MAX_LINKS_PER_PAGE:
                    break
                if link in self.visited_urls or self._skip_host(link):
                    continue
                if not re.search(r"news|company|companies|startup|about|ai|tech|career",
                                 link.lower()):
                    continue
                self.visited_urls.add(link)
                child = gs.add_node(self.run_id, "page", link, url=link, depth=depth + 1,
                                    discovered_from=node_id)
                gs.add_edge(self.run_id, node_id, child, "links_to")
                frontier.append(("page", link, link, depth + 1, child))
                followed += 1

    def _add_company(self, name, reason, depth, page_node_id, frontier):
        name = (name or "").strip()
        key = re.sub(r"[^a-z0-9]+", "", name.lower())
        if not name or not key or key in self.seen_companies:
            # Still connect the page to an already-known company
            return
        if self.n_companies >= self.max_companies:
            return
        self.seen_companies.add(key)
        self.n_companies += 1
        cid = gs.add_node(self.run_id, "company", name, depth=depth,
                          discovered_from=page_node_id, meta={"reason": reason})
        gs.add_edge(self.run_id, page_node_id, cid, "mentions")
        self._log(f"  + company: {name}")
        frontier.append(("company", name, name, depth, cid))

    def _add_person(self, person: dict, page_node_id):
        name = (person.get("name") or "").strip()
        if not name:
            return
        self.n_people += 1
        pid = gs.add_node(self.run_id, "person", name, depth=0,
                          discovered_from=page_node_id,
                          meta={"role": person.get("role", ""), "company": person.get("company", "")})
        gs.add_edge(self.run_id, page_node_id, pid, "mentions")

    def _process_company(self, company, company_node_id):
        time.sleep(self.DELAY)
        try:
            jobs = resolve_career_jobs(company, self.keywords)
        except Exception as e:
            logger.debug("career resolution failed for %s: %s", company, e)
            return
        if not jobs:
            return
        for j in jobs:
            title = j.get("title") or self.term.title()
            url = j.get("url")
            if not url:
                continue
            score = self._score(title, company, j.get("location", ""), j.get("description", ""), url)
            job_id = gs.add_job(
                self.run_id, company_node_id, title, company,
                j.get("location", ""), url, j.get("description", "")[:4000],
                j.get("source", ""), score,
            )
            if job_id > 0:
                self.n_jobs += 1
                jnode = gs.add_node(self.run_id, "job", title, url=url, depth=0,
                                    discovered_from=company_node_id,
                                    meta={"score": round(score, 3), "graph_job_id": job_id})
                gs.add_edge(self.run_id, company_node_id, jnode, "hiring")
        self._log(f"  {company}: {len(jobs)} roles")

    def _score(self, title, company, location, description, url) -> float:
        try:
            job = Job(title=title, company=company, location=location, url=url,
                      board=JobBoard.INTERNET, description=description or "",
                      is_remote=classify_remote(title, description, location) == "remote")
            score, _ = self.matcher.score(job)
            return float(score)
        except Exception:
            return 0.0


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat()
