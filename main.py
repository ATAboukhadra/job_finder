#!/usr/bin/env python3
"""
AI Apply — Automated job application pipeline.

Usage:
    python main.py scrape                    # Scrape all boards with default config
    python main.py scrape --boards remotive adzuna --max 30
    python main.py match                     # Re-score all stored jobs
    python main.py top --limit 20            # Show top matches
    python main.py export -o jobs.json       # Export to JSON
    python main.py ui                        # Launch web UI
    python main.py pipeline                  # Run full automation cycle once
    python main.py pipeline --dry-run        # Preview what would happen
    python main.py daemon                    # Start background automation (every 2 days)
    python main.py customize --url <URL>     # Generate custom CV for a specific job
    python main.py answers --url <URL>       # Show pre-generated form answers
    python main.py init-profile              # Generate profile.yaml from life-story.md (requires Ollama)
"""

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Avoid Windows console Unicode crashes when printing job titles.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from models import Job, JobBoard, SearchQuery
from scrapers import SCRAPERS
from matcher import JobMatcher
from storage import save_jobs, update_scores, get_top_jobs, get_db

CONFIG_PATH = Path(__file__).parent / "profile.yaml"

# Scrapers that ignore the location parameter — only need to run once per keyword
LOCATION_AGNOSTIC_BOARDS = {"remotive", "arbeitnow", "himalayas", "greenhouse", "lever", "linkedin_posts", "internet"}

ALL_BOARDS = list(SCRAPERS.keys())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aiapply")


def load_profile() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(f"Profile config not found: {CONFIG_PATH}")
        logger.error("Copy profile.yaml.example to profile.yaml and edit it.")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_queries(profile: dict) -> List[SearchQuery]:
    """Build search queries from profile config."""
    search = profile.get("search", {})
    board_names = search.get("boards", ["remotive", "adzuna", "linkedin", "jsearch"])
    boards = []
    for b in board_names:
        try:
            boards.append(JobBoard(b))
        except ValueError:
            logger.warning(f"Unknown board: {b}")
    queries = []
    for kw in search.get("queries", ["machine learning engineer"]):
        for loc in search.get("locations", [search.get("location", "")]):
            queries.append(SearchQuery(
                keywords=kw,
                location=loc,
                remote=search.get("remote", False),
                job_type=search.get("job_type", ""),
                max_age_days=search.get("max_age_days", 14),
                boards=boards,
            ))
    return queries


def _scrape_one(board_name: str, query: SearchQuery, max_results: int,
                fetch_details: bool) -> list[Job]:
    """Scrape a single board+query (runs inside a thread)."""
    scraper_cls = SCRAPERS.get(board_name)
    if not scraper_cls:
        return []
    scraper = scraper_cls()
    jobs = scraper.scrape(query, max_results=max_results)
    if fetch_details:
        for job in jobs[:10]:
            scraper.get_job_details(job)
    return jobs


def _filter_old_jobs(jobs: List[Job], max_age_days: int = 180) -> List[Job]:
    """Drop jobs whose date_posted is older than max_age_days. Jobs with no
    parseable date are kept (we can't confirm they're old)."""
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_days * 86400
    kept, dropped = [], 0
    for job in jobs:
        raw = str(job.date_posted or "").strip()
        if not raw:
            kept.append(job)
            continue
        ts = None
        # Unix ms timestamp (Lever uses this)
        if raw.isdigit() and len(raw) == 13:
            ts = int(raw) / 1000
        else:
            # Strip microseconds before parsing
            normalized = re.sub(r"\.\d+", "", raw)
            for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
                        "%B %d, %Y", "%b %d, %Y"):
                try:
                    dt = datetime.strptime(normalized, fmt)
                    ts = dt.replace(tzinfo=timezone.utc).timestamp()
                    break
                except ValueError:
                    continue
        if ts is None or ts >= cutoff:
            kept.append(job)
        else:
            dropped += 1
    if dropped:
        logger.info(f"  Filtered out {dropped} jobs older than {max_age_days} days")
    return kept


def cmd_scrape(args):
    """Scrape jobs from all configured boards."""
    profile = load_profile()
    queries = build_queries(profile)

    if getattr(args, "all_boards", False):
        override_boards = [JobBoard(b) for b in ALL_BOARDS]
        for q in queries:
            q.boards = override_boards
    elif args.boards:
        override_boards = [JobBoard(b) for b in args.boards]
        for q in queries:
            q.boards = override_boards

    matcher = JobMatcher(profile)
    all_jobs: List[Job] = []

    # Track (board, keyword) combos already submitted so location-agnostic
    # boards don't get called repeatedly for every location.
    seen_combos: set[tuple[str, str]] = set()
    futures = []

    max_workers = min(8, len(queries) * 3)  # reasonable thread count
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for query in queries:
            logger.info(f"Searching: '{query.keywords}' in '{query.location or 'anywhere'}'")

            for board in query.boards:
                board_name = board.value

                # Skip duplicate calls for location-agnostic boards
                if board_name in LOCATION_AGNOSTIC_BOARDS:
                    combo = (board_name, query.keywords)
                    if combo in seen_combos:
                        logger.debug(f"  Skipping {board_name} (already queried for '{query.keywords}')")
                        continue
                    seen_combos.add(combo)

                logger.info(f"  Queuing {board_name}...")
                fut = pool.submit(
                    _scrape_one, board_name, query, args.max, args.fetch_details
                )
                fut.board_name = board_name  # type: ignore[attr-defined]
                fut.query = query  # type: ignore[attr-defined]
                futures.append(fut)

        for fut in as_completed(futures):
            board_name = fut.board_name  # type: ignore[attr-defined]
            query = fut.query  # type: ignore[attr-defined]
            try:
                jobs = fut.result()
                logger.info(f"  {board_name} ({query.keywords[:30]}): {len(jobs)} jobs")
                all_jobs.extend(jobs)
            except Exception as e:
                logger.error(f"  {board_name} failed: {e}")

    # Deduplicate by URL and by title+company fingerprint
    seen_urls: set[str] = set()
    seen_fingerprints: set[str] = set()
    unique_jobs = []
    for j in all_jobs:
        fingerprint = f"{j.title.lower().strip()}|{j.company.lower().strip()}"
        if j.url not in seen_urls and fingerprint not in seen_fingerprints:
            seen_urls.add(j.url)
            seen_fingerprints.add(fingerprint)
            unique_jobs.append(j)

    unique_jobs = _filter_old_jobs(unique_jobs, max_age_days=180)

    ranked = matcher.rank(unique_jobs)
    n_saved = save_jobs(ranked)
    logger.info(f"\nTotal: {len(ranked)} unique jobs scraped, {n_saved} new saved to DB")
    _print_jobs(ranked[:15])


def cmd_match(args):
    """Re-score all stored jobs with current profile."""
    profile = load_profile()
    matcher = JobMatcher(profile)

    conn = get_db()
    rows = conn.execute("SELECT * FROM jobs WHERE hidden = 0").fetchall()
    conn.close()

    jobs = []
    for r in rows:
        jobs.append(Job(
            title=r["title"],
            company=r["company"],
            location=r["location"],
            url=r["url"],
            board=JobBoard(r["board"]),
            description=r["description"] or "",
            salary=r["salary"] or "",
            date_posted=r["date_posted"] or "",
            job_type=r["job_type"] or "",
            scraped_at=r["scraped_at"] or "",
        ))

    ranked = matcher.rank(jobs, min_score=args.min_score)
    update_scores(ranked)
    logger.info(f"Re-scored {len(ranked)} jobs")
    _print_jobs(ranked[:20])


def cmd_top(args):
    """Show top matching jobs from DB."""
    jobs = get_top_jobs(limit=args.limit, min_score=args.min_score)
    if not jobs:
        print("No jobs found. Run 'scrape' first.")
        return
    for i, j in enumerate(jobs, 1):
        score = j["match_score"]
        details = json.loads(j.get("match_details", "{}"))
        skills = ", ".join(details.get("skills_matched", []))
        print(
            f"{i:3d}. [{score:.2f}] {j['title']}\n"
            f"     {j['company']} | {j['location']} | {j['board']}\n"
            f"     Skills: {skills or 'N/A'}\n"
            f"     {j['url']}\n"
        )


def cmd_export(args):
    """Export top jobs to JSON."""
    jobs = get_top_jobs(limit=args.limit, min_score=args.min_score)
    output = Path(args.output)
    with open(output, "w") as f:
        json.dump(jobs, f, indent=2, default=str)
    print(f"Exported {len(jobs)} jobs to {output}")


def cmd_ui(args):
    """Launch the web UI."""
    from app import create_app
    app = create_app()
    print(f"Starting AI Apply UI at http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


def _print_jobs(jobs: List[Job]):
    """Pretty-print job list to console."""
    if not jobs:
        print("No jobs found.")
        return
    print(f"\n{'#':>3} {'Score':>5}  {'Title':<45} {'Company':<25} {'Board':<10}")
    print("-" * 95)
    for i, j in enumerate(jobs, 1):
        title = j.title[:44] if len(j.title) > 44 else j.title
        company = j.company[:24] if len(j.company) > 24 else j.company
        matched = ", ".join(j.match_details.get("skills_matched", [])[:5])
        print(f"{i:3d} {j.match_score:5.2f}  {title:<45} {company:<25} {j.board.value:<10}")
        if matched:
            print(f"{'':>10} Skills: {matched}")
    print()


def cmd_pipeline(args):
    """Run the full automation pipeline once."""
    from pipeline import run_pipeline
    profile = load_profile()
    stats = run_pipeline(
        profile=profile,
        dry_run=args.dry_run,
        max_applications=args.max,
        threshold=args.threshold,
    )
    print(f"\nPipeline complete: {stats}")


def cmd_daemon(args):
    """Start the background automation daemon."""
    from pipeline import run_daemon
    run_daemon(interval_hours=args.interval)


def cmd_customize(args):
    """Generate a customized CV for a specific job."""
    from cv_customizer import customize_cv_for_job
    from cover_letter import create_cover_letter
    from form_answers import generate_form_answers
    from cv_customizer import analyze_job, LIFE_STORY_PATH

    conn = get_db()
    row = conn.execute("SELECT * FROM jobs WHERE url = ?", (args.url,)).fetchone()
    conn.close()

    if not row:
        print(f"Job not found in DB: {args.url}")
        sys.exit(1)

    job = dict(row)
    profile = load_profile()
    model = profile.get("pipeline", {}).get("ollama_model", "qwen3.5:9b")

    print(f"Customizing CV for: {job['title']} at {job['company']}")

    result = customize_cv_for_job(
        job_url=job["url"],
        title=job["title"],
        company=job["company"],
        location=job.get("location", ""),
        description=job.get("description", ""),
        model=model,
        profile=profile,
    )

    if result:
        print(f"CV generated: {result['cv_pdf_path']}")

        # Also generate cover letter and form answers
        life_story = LIFE_STORY_PATH.read_text(encoding="utf-8") if LIFE_STORY_PATH.exists() else ""
        job_analysis = analyze_job(job.get("description", ""), job["title"], job["company"], model=model)

        cl_path = create_cover_letter(
            app_dir=result["app_dir"],
            title=job["title"],
            company=job["company"],
            location=job.get("location", ""),
            description=job.get("description", ""),
            life_story=life_story,
            job_analysis=job_analysis,
            model=model,
        )
        if cl_path:
            print(f"Cover letter generated: {cl_path}")

        answers = generate_form_answers(
            life_story=life_story,
            title=job["title"],
            company=job["company"],
            description=job.get("description", ""),
            job_analysis=job_analysis,
            model=model,
        )
        if answers:
            print(f"\nForm Answers ({len(answers)} questions):")
            for q, a in answers.items():
                print(f"  Q: {q}")
                print(f"  A: {a}\n")

        # Create application record
        from storage import create_application, update_application
        app_id = create_application(job["url"], result["slug"])
        update_application(
            app_id,
            status="ready",
            cv_pdf_path=result["cv_pdf_path"],
            cover_letter_pdf_path=cl_path or "",
            form_answers_json=json.dumps(answers),
        )
        print(f"\nApplication saved (ID: {app_id})")
    else:
        print("Failed to generate CV")
        sys.exit(1)


def cmd_answers(args):
    """Show pre-generated form answers for a job."""
    from form_filler import get_fill_instructions, format_fill_guide

    instructions = get_fill_instructions(args.url)
    if not instructions:
        print(f"No application found for: {args.url}")
        print("Run 'customize --url <URL>' first.")
        sys.exit(1)

    print(format_fill_guide(instructions))


# --- Job selectors tried in order when extracting description from a page ---
_DESC_SELECTORS = [
    "div[class*='job-description']",
    "div[id*='job-description']",
    "div[class*='description']",
    "div[id*='description']",
    "section[class*='description']",
    "div[class*='job-detail']",
    "div[class*='jobdetail']",
    "div[class*='job_detail']",
    "article",
    "main",
]


def _fetch_job_page(url: str) -> tuple[str, str]:
    """Fetch a job URL and return (title, description_text)."""
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error fetching URL: {e}")
        sys.exit(1)

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    og_title = soup.find("meta", property="og:title")
    title = (
        og_title["content"].strip()
        if og_title and og_title.get("content")
        else (soup.find("title").get_text(strip=True) if soup.find("title") else url)
    )

    # Description — try targeted selectors, fall back to <main> or full body
    desc = ""
    for sel in _DESC_SELECTORS:
        el = soup.select_one(sel)
        if el and len(el.get_text(strip=True)) > 200:
            desc = el.get_text(separator="\n", strip=True)
            break

    if not desc:
        # Last resort: strip nav/header/footer and take body text
        for tag in soup(["nav", "header", "footer", "script", "style"]):
            tag.decompose()
        desc = soup.get_text(separator="\n", strip=True)

    return title, desc[:5000]


def cmd_score(args):
    """Fetch a job URL and compute its similarity score against your profile."""
    profile = load_profile()
    matcher = JobMatcher(profile)

    print(f"Fetching: {args.url}")
    title, desc = _fetch_job_page(args.url)

    if not desc:
        print("Could not extract job description from the page.")
        sys.exit(1)

    print(f"Title   : {title}")
    print(f"Desc    : {len(desc)} chars extracted\n")

    job = Job(
        title=title,
        company="",
        location=args.location or "",
        url=args.url,
        description=desc,
        board=JobBoard.LINKEDIN,
    )

    score, details = matcher.score(job)

    bar_width = 40
    filled = int(score * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"Match Score : {score:.1%}  [{bar}]")
    print()
    print(f"  Title score    : {details.get('title_score', 0):.3f}")
    print(f"  Skill score    : {details.get('skill_score', 0):.3f}")
    print(f"  Semantic score : {details.get('semantic_score', 0):.3f}")
    print(f"  Location score : {details.get('location_score', 0):.3f}")
    print(f"  Experience     : {details.get('experience_score', 0):.3f}")
    print(f"  Recency        : {details.get('recency_score', 0):.3f}")
    print(f"  Weighted total : {details.get('weighted_total', 0):.3f}")

    if args.save:
        jobs_saved = save_jobs([job])
        matcher.rank([job])
        print(f"\nSaved to DB ({jobs_saved} new).")


def cmd_init_profile(args):
    """Generate profile.yaml from a life-story.md using a local LLM."""
    from profile_generator import generate_profile_from_life_story
    life_story_path = Path(args.life_story).expanduser()
    output_path = Path(args.output).expanduser()

    if not life_story_path.exists():
        print(f"Life story not found: {life_story_path}")
        print(f"Create it first — a template is at: cv_templates/life_story_template.md")
        sys.exit(1)

    ok = generate_profile_from_life_story(
        life_story_path=life_story_path,
        output_path=output_path,
        model=getattr(args, "model", None),
    )
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AI Apply — Automated job application pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scrape
    p_scrape = subparsers.add_parser("scrape", help="Scrape jobs from boards")
    p_scrape.add_argument("--boards", nargs="+", choices=ALL_BOARDS)
    p_scrape.add_argument("--all", dest="all_boards", action="store_true",
                          help="Run every available scraper (overrides --boards and profile)")
    p_scrape.add_argument("--max", type=int, default=50, help="Max results per board per query")
    p_scrape.add_argument("--fetch-details", action="store_true", help="Fetch full descriptions (slower)")
    p_scrape.set_defaults(func=cmd_scrape)

    # match
    p_match = subparsers.add_parser("match", help="Re-score stored jobs with current profile")
    p_match.add_argument("--min-score", type=float, default=0.0)
    p_match.set_defaults(func=cmd_match)

    # top
    p_top = subparsers.add_parser("top", help="Show top matching jobs")
    p_top.add_argument("--limit", type=int, default=20)
    p_top.add_argument("--min-score", type=float, default=0.0)
    p_top.set_defaults(func=cmd_top)

    # export
    p_export = subparsers.add_parser("export", help="Export top jobs to JSON")
    p_export.add_argument("--output", "-o", default="top_jobs.json")
    p_export.add_argument("--limit", type=int, default=50)
    p_export.add_argument("--min-score", type=float, default=0.0)
    p_export.set_defaults(func=cmd_export)

    # ui
    p_ui = subparsers.add_parser("ui", help="Launch web UI")
    p_ui.add_argument("--port", type=int, default=5000)
    p_ui.add_argument("--debug", action="store_true")
    p_ui.set_defaults(func=cmd_ui)

    # pipeline
    p_pipeline = subparsers.add_parser("pipeline", help="Run full automation cycle")
    p_pipeline.add_argument("--dry-run", action="store_true", help="Preview without executing")
    p_pipeline.add_argument("--max", type=int, default=10, help="Max applications per run")
    p_pipeline.add_argument("--threshold", type=float, default=0.5, help="Min score to process")
    p_pipeline.set_defaults(func=cmd_pipeline)

    # daemon
    p_daemon = subparsers.add_parser("daemon", help="Start background automation loop")
    p_daemon.add_argument("--interval", type=float, default=48.0, help="Hours between cycles")
    p_daemon.set_defaults(func=cmd_daemon)

    # customize
    p_custom = subparsers.add_parser("customize", help="Generate custom CV for a job")
    p_custom.add_argument("--url", required=True, help="Job URL from database")
    p_custom.set_defaults(func=cmd_customize)

    # answers
    p_answers = subparsers.add_parser("answers", help="Show form answers for a job")
    p_answers.add_argument("--url", required=True, help="Job URL")
    p_answers.set_defaults(func=cmd_answers)

    # score
    p_score = subparsers.add_parser("score", help="Fetch a job URL and compute similarity score")
    p_score.add_argument("--url", required=True, help="Job URL to evaluate")
    p_score.add_argument("--location", default="", help="Job location (optional, affects location score)")
    p_score.add_argument("--save", action="store_true", help="Also save the job to DB")
    p_score.set_defaults(func=cmd_score)

    # init-profile
    p_init = subparsers.add_parser(
        "init-profile",
        help="Generate profile.yaml from a life-story.md using a local LLM (requires Ollama)"
    )
    p_init.add_argument(
        "--life-story",
        default=str(Path(__file__).parent / "life-story.md"),
        help="Path to your life-story.md (default: life-story.md in project root)"
    )
    p_init.add_argument(
        "--output",
        default=str(Path(__file__).parent / "profile.yaml"),
        help="Output path for profile.yaml (default: profile.yaml in project root)"
    )
    p_init.add_argument("--model", default=None, help="Ollama model to use (auto-detected if not set)")
    p_init.set_defaults(func=cmd_init_profile)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
