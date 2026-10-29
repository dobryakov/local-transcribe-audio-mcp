"""API-key auth. Keys are compared by SHA-256 hash; only hashes live in env.

A key looks like `tsb_<random>`; the stored form is its hex sha256 with a label,
see docs/design.md §8. OAuth / secret-in-URL for remote connectors comes later.
"""
from __future__ import annotations

import hashlib
import hmac

from .config import settings


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def label_for_key(raw: str) -> str | None:
    """Return the key's label if it matches a configured key, else None."""
    candidate = hash_key(raw)
    for stored_hash, label in settings.api_keys.items():
        if hmac.compare_digest(candidate, stored_hash):
            return label
    return None


def label_for_secret(secret: str) -> str | None:
    """Return the label for a URL-path MCP secret if it matches, else None."""
    candidate = hash_key(secret)
    for stored_hash, label in settings.mcp_url_secrets.items():
        if hmac.compare_digest(candidate, stored_hash):
            return label
    return None


def extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None
