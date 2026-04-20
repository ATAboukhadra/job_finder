"""SQLite storage for scraped jobs and applications."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from models import Job, JobBoard

DB_PATH = Path(__file__).parent / "jobs.db"


def get_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            url TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            location TEXT,
            board TEXT,
            description TEXT,
            salary TEXT,
            date_posted TEXT,
            job_type TEXT,
            scraped_at TEXT,
            match_score REAL DEFAULT 0,
            match_details TEXT DEFAULT '{}',
            applied INTEGER DEFAULT 0,
            hidden INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_url TEXT REFERENCES jobs(url),
            slug TEXT UNIQUE,
            status TEXT DEFAULT 'pending',
            cv_pdf_path TEXT,
            cover_letter_pdf_path TEXT,
            form_answers_json TEXT DEFAULT '{}',
            recruiter_email TEXT DEFAULT '',
            email_subject TEXT DEFAULT '',
            email_body TEXT DEFAULT '',
            approved_at TEXT DEFAULT '',
            sent_at TEXT DEFAULT '',
            created_at TEXT,
            updated_at TEXT
        )
    """)
    # Lightweight schema migration for existing DBs
    _ensure_columns(
        conn,
        "applications",
        {
            "recruiter_email": "TEXT DEFAULT ''",
            "email_subject": "TEXT DEFAULT ''",
            "email_body": "TEXT DEFAULT ''",
            "approved_at": "TEXT DEFAULT ''",
            "sent_at": "TEXT DEFAULT ''",
        },
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT,
            finished_at TEXT,
            jobs_scraped INTEGER DEFAULT 0,
            jobs_matched INTEGER DEFAULT 0,
            applications_created INTEGER DEFAULT 0,
            emails_sent INTEGER DEFAULT 0,
            status TEXT DEFAULT 'running',
            log TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS email_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sent_at TEXT,
            subject TEXT,
            job_count INTEGER DEFAULT 0,
            recipient TEXT
        )
    """)
    conn.commit()
    return conn


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Add columns if missing (SQLite ALTER TABLE ADD COLUMN)."""
    try:
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
    conn.commit()


# --- Application CRUD ---

def create_application(job_url: str, slug: str, db_path: Path = DB_PATH) -> int:
    """Create a new application record. Returns the application ID."""
    conn = get_db(db_path)
    now = datetime.now().isoformat()
    cursor = conn.execute(
        """INSERT INTO applications (job_url, slug, status, created_at, updated_at)
           VALUES (?, ?, 'pending', ?, ?)""",
        (job_url, slug, now, now),
    )
    app_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return app_id


def update_application(app_id: int, db_path: Path = DB_PATH, **kwargs):
    """Update application fields. Pass field=value as keyword args."""
    conn = get_db(db_path)
    kwargs["updated_at"] = datetime.now().isoformat()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [app_id]
    conn.execute(f"UPDATE applications SET {sets} WHERE id = ?", values)
    conn.commit()
    conn.close()


def get_applications(
    status: Optional[str] = None,
    limit: int = 50,
    db_path: Path = DB_PATH,
) -> List[Dict]:
    """Get applications, optionally filtered by status."""
    conn = get_db(db_path)
    if status:
        rows = conn.execute(
            """SELECT a.*, j.title, j.company, j.location, j.match_score, j.board
               FROM applications a JOIN jobs j ON a.job_url = j.url
               WHERE a.status = ? ORDER BY a.created_at DESC LIMIT ?""",
            (status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT a.*, j.title, j.company, j.location, j.match_score, j.board
               FROM applications a JOIN jobs j ON a.job_url = j.url
               ORDER BY a.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_application_by_job(job_url: str, db_path: Path = DB_PATH) -> Optional[Dict]:
    """Get application for a specific job, or None."""
    conn = get_db(db_path)
    row = conn.execute(
        "SELECT * FROM applications WHERE job_url = ?", (job_url,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# --- Pipeline Run Logging ---

def start_pipeline_run(db_path: Path = DB_PATH) -> int:
    """Start a new pipeline run. Returns run ID."""
    conn = get_db(db_path)
    now = datetime.now().isoformat()
    cursor = conn.execute(
        "INSERT INTO pipeline_runs (started_at, status) VALUES (?, 'running')",
        (now,),
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id


def finish_pipeline_run(
    run_id: int,
    jobs_scraped: int = 0,
    jobs_matched: int = 0,
    applications_created: int = 0,
    emails_sent: int = 0,
    status: str = "completed",
    log: str = "",
    db_path: Path = DB_PATH,
):
    """Finish a pipeline run with results."""
    conn = get_db(db_path)
    now = datetime.now().isoformat()
    conn.execute(
        """UPDATE pipeline_runs
           SET finished_at=?, jobs_scraped=?, jobs_matched=?,
               applications_created=?, emails_sent=?, status=?, log=?
           WHERE id=?""",
        (now, jobs_scraped, jobs_matched, applications_created, emails_sent, status, log, run_id),
    )
    conn.commit()
    conn.close()


def get_pipeline_runs(limit: int = 20, db_path: Path = DB_PATH) -> List[Dict]:
    conn = get_db(db_path)
    rows = conn.execute(
        "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Email Log ---

def log_email_sent(subject: str, job_count: int, recipient: str, db_path: Path = DB_PATH):
    conn = get_db(db_path)
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT INTO email_log (sent_at, subject, job_count, recipient) VALUES (?, ?, ?, ?)",
        (now, subject, job_count, recipient),
    )
    conn.commit()
    conn.close()


def get_last_email_sent(db_path: Path = DB_PATH) -> Optional[Dict]:
    conn = get_db(db_path)
    row = conn.execute(
        "SELECT * FROM email_log ORDER BY sent_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_new_jobs_since(since_iso: str, min_score: float = 0.0, db_path: Path = DB_PATH) -> List[Dict]:
    """Get jobs scraped after a given ISO timestamp."""
    conn = get_db(db_path)
    rows = conn.execute(
        """SELECT * FROM jobs
           WHERE scraped_at > ? AND match_score >= ? AND hidden = 0
           ORDER BY match_score DESC""",
        (since_iso, min_score),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_jobs(jobs: List[Job], db_path: Path = DB_PATH) -> int:
    """Save jobs to SQLite. Returns number of new jobs inserted.
    Deduplicates by both URL and title+company fingerprint."""
    conn = get_db(db_path)
    # Load existing fingerprints to skip title+company duplicates
    existing = conn.execute(
        "SELECT LOWER(TRIM(title)) || '|' || LOWER(TRIM(company)) FROM jobs"
    ).fetchall()
    seen_fingerprints = {row[0] for row in existing}

    inserted = 0
    for job in jobs:
        fingerprint = f"{job.title.lower().strip()}|{job.company.lower().strip()}"
        if fingerprint in seen_fingerprints:
            continue
        try:
            conn.execute(
                """INSERT OR IGNORE INTO jobs
                   (url, title, company, location, board, description,
                    salary, date_posted, job_type, scraped_at, match_score, match_details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.url, job.title, job.company, job.location,
                    job.board.value, job.description, job.salary,
                    job.date_posted, job.job_type, job.scraped_at,
                    job.match_score, json.dumps(job.match_details),
                ),
            )
            seen_fingerprints.add(fingerprint)
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    return inserted


def update_scores(jobs: List[Job], db_path: Path = DB_PATH):
    """Update match scores for existing jobs."""
    conn = get_db(db_path)
    for job in jobs:
        conn.execute(
            "UPDATE jobs SET match_score = ?, match_details = ? WHERE url = ?",
            (job.match_score, json.dumps(job.match_details), job.url),
        )
    conn.commit()
    conn.close()


def get_top_jobs(limit: int = 20, min_score: float = 0.0, db_path: Path = DB_PATH) -> list[dict]:
    """Get top-scored jobs from the database."""
    conn = get_db(db_path)
    rows = conn.execute(
        """SELECT * FROM jobs
           WHERE match_score >= ? AND hidden = 0
           ORDER BY match_score DESC, CASE WHEN date_posted IS NULL OR date_posted = '' OR LOWER(date_posted) IN ('nan','nat','none','null') THEN 0 ELSE 1 END DESC, date_posted DESC
           LIMIT ?""",
        (min_score, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_applied(url: str, db_path: Path = DB_PATH):
    conn = get_db(db_path)
    conn.execute("UPDATE jobs SET applied = 1 WHERE url = ?", (url,))
    conn.commit()
    conn.close()


def mark_hidden(url: str, db_path: Path = DB_PATH):
    conn = get_db(db_path)
    conn.execute("UPDATE jobs SET hidden = 1 WHERE url = ?", (url,))
    conn.commit()
    conn.close()
