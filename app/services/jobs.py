"""Job lifecycle + FIFO queue, shared by REST and (later) MCP.

One worker claims one `queued` row at a time via an atomic UPDATE..RETURNING,
so no external queue is needed (docs/design.md §9.1).
"""
from __future__ import annotations

import json
import secrets
import time
from datetime import datetime, timezone
from typing import Optional

from ..config import settings
from ..db import connect
from ..models import JobParams

try:
    from ulid import ULID  # python-ulid

    def _new_id() -> str:
        return str(ULID())
except Exception:  # pragma: no cover - fallback if ulid not installed
    def _new_id() -> str:
        # time-sortable-ish: hex millis + random
        return f"{int(time.time() * 1000):012x}{secrets.token_hex(6)}".upper()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["params"] = json.loads(d["params"]) if d.get("params") else {}
    if d.get("detected_languages"):
        d["detected_languages"] = json.loads(d["detected_languages"])
    return d


def create_job(
    params: JobParams,
    source_filename: Optional[str],
    created_by: Optional[str],
    status: str = "queued",
    with_upload_token: bool = False,
) -> dict:
    params.validate()
    job_id = _new_id()
    now = _now()
    token = None
    token_hash = None
    expires = None
    if with_upload_token:
        from ..auth import hash_key

        token = secrets.token_urlsafe(32)
        token_hash = hash_key(token)
        expires = datetime.fromtimestamp(
            time.time() + settings.upload_link_ttl_s, tz=timezone.utc
        ).isoformat(timespec="seconds")

    with connect() as conn:
        conn.execute(
            """INSERT INTO jobs
               (id, status, progress, params, title, source_filename,
                created_by, created_at, upload_token_hash, upload_expires_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                job_id, status, 0.0, json.dumps(params.to_dict()),
                params.title or source_filename, source_filename,
                created_by, now, token_hash, expires,
            ),
        )
    out = get_job(job_id)
    if token:
        out["_upload_token"] = token  # returned once to the caller
        out["_upload_expires_at"] = expires
    return out


def get_job(job_id: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(status: Optional[str] = None, limit: int = 50, query: Optional[str] = None) -> list[dict]:
    sql = "SELECT * FROM jobs"
    clauses, args = [], []
    if status:
        clauses.append("status=?")
        args.append(status)
    if query:
        clauses.append("title LIKE ?")
        args.append(f"%{query}%")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(max(1, min(limit, 500)))
    with connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_row_to_dict(r) for r in rows]


def position_in_queue(job_id: str) -> Optional[int]:
    with connect() as conn:
        row = conn.execute("SELECT created_at FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        n = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='queued' AND created_at < ?",
            (row["created_at"],),
        ).fetchone()[0]
    return n


def claim_next() -> Optional[dict]:
    """Atomically move the oldest queued job to processing. Returns it or None."""
    now = _now()
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                conn.execute("COMMIT")
                return None
            conn.execute(
                "UPDATE jobs SET status='processing', started_at=?, stage='convert', progress=0 WHERE id=?",
                (now, row["id"]),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return get_job(row["id"])


def update_progress(job_id: str, stage: str, progress: float, eta_s: Optional[int] = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET stage=?, progress=?, eta_s=? WHERE id=? AND status='processing'",
            (stage, max(0.0, min(progress, 1.0)), eta_s, job_id),
        )


def mark_done(job_id: str, duration_s: float, detected_languages: list[str], speakers_count: int) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE jobs SET status='done', stage=NULL, progress=1.0, eta_s=0,
               finished_at=?, duration_s=?, detected_languages=?, speakers_count=?
               WHERE id=?""",
            (_now(), duration_s, json.dumps(detected_languages), speakers_count, job_id),
        )


def mark_failed(job_id: str, error: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
            (_now(), error[:2000], job_id),
        )


def cancel_job(job_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status='cancelled', finished_at=? WHERE id=? AND status IN ('queued','processing','awaiting_upload')",
            (_now(), job_id),
        )
        return cur.rowcount > 0


def delete_job(job_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    # transcripts/work cleanup is the caller's / cron's job
    return cur.rowcount > 0


def enqueue(job_id: str, source_filename: Optional[str] = None) -> Optional[dict]:
    """Flip an awaiting_upload job to queued once its file is on disk."""
    with connect() as conn:
        conn.execute(
            """UPDATE jobs SET status='queued', source_filename=COALESCE(?, source_filename)
               WHERE id=? AND status='awaiting_upload'""",
            (source_filename, job_id),
        )
    return get_job(job_id)


def attach_upload(token: str, source_filename: str) -> Optional[dict]:
    """Validate a one-time upload token, bind filename, move to queued."""
    from ..auth import hash_key

    th = hash_key(token)
    now = _now()
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE upload_token_hash=? AND status='awaiting_upload'",
                (th,),
            ).fetchone()
            if not row:
                conn.execute("COMMIT")
                return None
            if row["upload_expires_at"] and row["upload_expires_at"] < now:
                conn.execute("COMMIT")
                return None
            conn.execute(
                """UPDATE jobs SET status='queued', source_filename=?,
                   upload_token_hash=NULL WHERE id=?""",
                (source_filename, row["id"]),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return get_job(row["id"])
