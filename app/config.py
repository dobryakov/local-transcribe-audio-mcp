"""Centralised configuration, read once from the environment.

Mirrors the .env keys documented in docs/design.md §11.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _parse_api_keys(raw: str) -> dict[str, str]:
    """`label:sha256hex,label:sha256hex` -> {sha256hex: label}."""
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        label, _, h = item.partition(":")
        out[h.strip().lower()] = label.strip()
    return out


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "/data")))
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", "/models")))

    public_base_url: str = _env("PUBLIC_BASE_URL", "http://127.0.0.1:18080")

    default_language: str = _env("DEFAULT_LANGUAGE", "en")
    default_model: str = _env("DEFAULT_MODEL", "large-v3-turbo")
    default_diarization: str = _env("DEFAULT_DIARIZATION", "pyannote")

    max_upload_mb: int = int(_env("MAX_UPLOAD_MB", "2048"))
    max_duration_min: int = int(_env("MAX_DURATION_MIN", "240"))
    part_tokens: int = int(_env("PART_TOKENS", "15000"))
    pause_split_s: float = float(_env("PAUSE_SPLIT_S", "2"))

    cpu_threads: int = int(_env("CPU_THREADS", "3"))
    upload_link_ttl_s: int = int(_env("UPLOAD_LINK_TTL_S", "3600"))

    # Only needed to download gated pyannote models; unused at runtime (offline).
    hf_token: str = _env("HF_TOKEN", "")
    pyannote_pipeline: str = _env(
        "PYANNOTE_PIPELINE", "pyannote/speaker-diarization-community-1"
    )

    # {sha256hex: label}
    api_keys: dict[str, str] = field(default_factory=lambda: _parse_api_keys(_env("API_KEYS", "")))
    # Secrets embedded in the MCP URL path (/mcp/k/<secret>) for remote
    # connectors (Claude.ai) that cannot set an Authorization header. Same
    # label:sha256 format as API_KEYS. {sha256hex: label}
    mcp_url_secrets: dict[str, str] = field(
        default_factory=lambda: _parse_api_keys(_env("MCP_URL_SECRETS", ""))
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db.sqlite"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.work_dir, self.transcripts_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
