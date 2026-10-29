#!/usr/bin/env python3
"""Periodic housekeeping (design §8 "Данные", §12 stage 4). Idempotent — run
from cron, e.g. hourly:  */30 * * * *  python /srv/scripts/cleanup.py

- awaiting_upload jobs older than TTL (file never uploaded): delete + work dir.
- failed jobs older than TTL: drop the work dir (stale WAV kept for retry until
  now), keep the job row for history.
- orphan work dirs with no matching job row: remove.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.db import connect, init_db  # noqa: E402

STALE_HOURS = 24


def _rm_work(job_id: str) -> None:
    shutil.rmtree(settings.work_dir / job_id, ignore_errors=True)


def main() -> None:
    init_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=STALE_HOURS)).isoformat(timespec="seconds")
    removed_jobs = removed_work = orphans = 0

    with connect() as conn:
        # 1) never-uploaded jobs past TTL
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status='awaiting_upload' AND created_at < ?",
            (cutoff,),
        ).fetchall()
        for r in rows:
            _rm_work(r["id"])
            conn.execute("DELETE FROM jobs WHERE id=?", (r["id"],))
            removed_jobs += 1

        # 2) failed jobs past TTL: free the work dir, keep the row
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status='failed' AND COALESCE(finished_at, created_at) < ?",
            (cutoff,),
        ).fetchall()
        for r in rows:
            if (settings.work_dir / r["id"]).exists():
                _rm_work(r["id"])
                removed_work += 1

        known = {r["id"] for r in conn.execute("SELECT id FROM jobs").fetchall()}

    # 3) orphan work dirs (no job row), skipping the token temp dirs
    if settings.work_dir.exists():
        for d in settings.work_dir.iterdir():
            if d.is_dir() and not d.name.startswith("_pending_") and d.name not in known:
                shutil.rmtree(d, ignore_errors=True)
                orphans += 1

    print(f"cleanup: removed_jobs={removed_jobs} freed_work={removed_work} orphans={orphans}")


if __name__ == "__main__":
    main()
