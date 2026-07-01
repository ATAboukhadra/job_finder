"""SQLite storage for the graph-search crawler.

Independent from the query-based search: the crawler writes its own tables
(graph_runs / graph_nodes / graph_edges / graph_jobs) into the same jobs.db
file, and only touches the main `jobs` table through the explicit
"send to pipeline" action (see storage.save_jobs).
"""

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from storage import DB_PATH


def get_graph_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT,
            term TEXT,
            status TEXT DEFAULT 'running',
            max_depth INTEGER DEFAULT 3,
            max_companies INTEGER DEFAULT 100,
            started_at TEXT,
            finished_at TEXT,
            stats_json TEXT DEFAULT '{}',
            log TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            node_type TEXT,
            name TEXT,
            url TEXT DEFAULT '',
            meta_json TEXT DEFAULT '{}',
            depth INTEGER DEFAULT 0,
            discovered_from INTEGER,
            dedup_key TEXT,
            created_at TEXT,
            UNIQUE(run_id, dedup_key)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            source_id INTEGER,
            target_id INTEGER,
            relation TEXT,
            UNIQUE(run_id, source_id, target_id, relation)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            company_node_id INTEGER,
            title TEXT,
            company TEXT,
            location TEXT,
            url TEXT UNIQUE,
            description TEXT DEFAULT '',
            source TEXT DEFAULT '',
            match_score REAL DEFAULT 0,
            pushed INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


def _norm(text: str) -> str:
    """Normalize a name/url for dedup (lowercase, collapse non-alphanumerics)."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower()).strip()


# --- Runs ---------------------------------------------------------------

def create_run(region: str, term: str, max_depth: int, max_companies: int,
               db_path: Path = DB_PATH) -> int:
    conn = get_graph_db(db_path)
    cur = conn.execute(
        """INSERT INTO graph_runs (region, term, status, max_depth, max_companies, started_at)
           VALUES (?, ?, 'running', ?, ?, ?)""",
        (region, term, max_depth, max_companies, datetime.now().isoformat()),
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def update_run(run_id: int, db_path: Path = DB_PATH, **fields):
    if not fields:
        return
    conn = get_graph_db(db_path)
    if "stats" in fields:
        fields["stats_json"] = json.dumps(fields.pop("stats"))
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE graph_runs SET {sets} WHERE id = ?",
                 list(fields.values()) + [run_id])
    conn.commit()
    conn.close()


def append_log(run_id: int, line: str, db_path: Path = DB_PATH):
    """Append a timestamped line to the run's log."""
    conn = get_graph_db(db_path)
    stamp = datetime.now().strftime("%H:%M:%S")
    conn.execute(
        "UPDATE graph_runs SET log = log || ? WHERE id = ?",
        (f"[{stamp}] {line}\n", run_id),
    )
    conn.commit()
    conn.close()


def get_run(run_id: int, db_path: Path = DB_PATH) -> Optional[Dict]:
    conn = get_graph_db(db_path)
    row = conn.execute("SELECT * FROM graph_runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_runs(limit: int = 25, db_path: Path = DB_PATH) -> List[Dict]:
    conn = get_graph_db(db_path)
    rows = conn.execute(
        "SELECT * FROM graph_runs ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Nodes & edges ------------------------------------------------------

def add_node(run_id: int, node_type: str, name: str, url: str = "",
             depth: int = 0, discovered_from: Optional[int] = None,
             meta: Optional[dict] = None, db_path: Path = DB_PATH) -> int:
    """Insert a node (idempotent within a run). Returns the node id.

    Dedup key is the node type plus the normalized url (if any) or name.
    """
    dedup_key = f"{node_type}:{_norm(url) or _norm(name)}"
    conn = get_graph_db(db_path)
    conn.execute(
        """INSERT OR IGNORE INTO graph_nodes
           (run_id, node_type, name, url, meta_json, depth, discovered_from, dedup_key, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, node_type, name, url, json.dumps(meta or {}), depth,
         discovered_from, dedup_key, datetime.now().isoformat()),
    )
    row = conn.execute(
        "SELECT id FROM graph_nodes WHERE run_id = ? AND dedup_key = ?",
        (run_id, dedup_key),
    ).fetchone()
    conn.commit()
    conn.close()
    return row["id"] if row else -1


def add_edge(run_id: int, source_id: int, target_id: int, relation: str,
             db_path: Path = DB_PATH):
    if source_id < 0 or target_id < 0 or source_id == target_id:
        return
    conn = get_graph_db(db_path)
    conn.execute(
        """INSERT OR IGNORE INTO graph_edges (run_id, source_id, target_id, relation)
           VALUES (?, ?, ?, ?)""",
        (run_id, source_id, target_id, relation),
    )
    conn.commit()
    conn.close()


def count_nodes(run_id: int, node_type: str, db_path: Path = DB_PATH) -> int:
    conn = get_graph_db(db_path)
    n = conn.execute(
        "SELECT COUNT(*) FROM graph_nodes WHERE run_id = ? AND node_type = ?",
        (run_id, node_type),
    ).fetchone()[0]
    conn.close()
    return n


# --- Jobs ---------------------------------------------------------------

def add_job(run_id: int, company_node_id: int, title: str, company: str,
            location: str, url: str, description: str = "", source: str = "",
            match_score: float = 0.0, db_path: Path = DB_PATH) -> int:
    conn = get_graph_db(db_path)
    try:
        cur = conn.execute(
            """INSERT OR IGNORE INTO graph_jobs
               (run_id, company_node_id, title, company, location, url,
                description, source, match_score, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, company_node_id, title, company, location, url,
             description, source, match_score, datetime.now().isoformat()),
        )
        job_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()
    return job_id or -1


def get_job(job_id: int, db_path: Path = DB_PATH) -> Optional[Dict]:
    conn = get_graph_db(db_path)
    row = conn.execute("SELECT * FROM graph_jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def mark_job_pushed(job_id: int, db_path: Path = DB_PATH):
    conn = get_graph_db(db_path)
    conn.execute("UPDATE graph_jobs SET pushed = 1 WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()


# --- Aggregate read for the UI -----------------------------------------

def get_graph(run_id: int, db_path: Path = DB_PATH) -> Dict:
    """Return everything the UI needs to render a run: nodes, edges, companies, jobs."""
    conn = get_graph_db(db_path)
    nodes = [dict(r) for r in conn.execute(
        "SELECT id, node_type, name, url, meta_json, depth FROM graph_nodes WHERE run_id = ?",
        (run_id,),
    ).fetchall()]
    edges = [dict(r) for r in conn.execute(
        "SELECT source_id, target_id, relation FROM graph_edges WHERE run_id = ?",
        (run_id,),
    ).fetchall()]
    jobs = [dict(r) for r in conn.execute(
        "SELECT * FROM graph_jobs WHERE run_id = ? ORDER BY match_score DESC, id DESC",
        (run_id,),
    ).fetchall()]
    # Company summary with job counts
    companies = [dict(r) for r in conn.execute(
        """SELECT n.id, n.name, n.url, n.meta_json,
                  (SELECT COUNT(*) FROM graph_jobs j WHERE j.company_node_id = n.id) AS job_count
           FROM graph_nodes n
           WHERE n.run_id = ? AND n.node_type = 'company'
           ORDER BY job_count DESC, n.name ASC""",
        (run_id,),
    ).fetchall()]
    conn.close()
    for n in nodes:
        n["meta"] = json.loads(n.pop("meta_json", "{}") or "{}")
    for c in companies:
        c["meta"] = json.loads(c.pop("meta_json", "{}") or "{}")
    return {"nodes": nodes, "edges": edges, "companies": companies, "jobs": jobs}
