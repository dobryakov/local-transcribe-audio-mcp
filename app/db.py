"""SQLite access layer. The `jobs` table doubles as the work queue
(one row claimed at a time), see docs/design.md §10.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                 TEXT PRIMARY KEY,
    status             TEXT NOT NULL,
    stage              TEXT,
    progress           REAL NOT NULL DEFAULT 0,
    eta_s              INTEGER,
    params             TEXT NOT NULL,          -- JSON
    title              TEXT,
    source_filename    TEXT,
    duration_s         REAL,
    detected_languages TEXT,                   -- JSON array
    speakers_count     INTEGER,
    error              TEXT,
    created_by         TEXT,
    created_at         TEXT NOT NULL,
    started_at         TEXT,
    finished_at        TEXT,
    upload_token_hash  TEXT,
    upload_expires_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_token ON jobs(upload_token_hash)
    WHERE upload_token_hash IS NOT NULL;

CREATE TABLE IF NOT EXISTS stats (
    model   TEXT NOT NULL,
    backend TEXT NOT NULL,
    stage   TEXT NOT NULL,
    rtf     REAL NOT NULL,
    at      TEXT NOT NULL
);
"""


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    settings.ensure_dirs()
    conn = _connect(settings.db_path)
    try:
        conn.executescript(SCHEMA)
    finally:
        conn.close()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = _connect(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()
