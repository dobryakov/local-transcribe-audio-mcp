#!/usr/bin/env python3
"""Generate a new API key, print it once, append its label:sha256 to .env.

Usage: python scripts/new_key.py <label>
"""
import hashlib
import re
import secrets
import sys
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent / ".env"


def main() -> None:
    if len(sys.argv) != 2 or not re.fullmatch(r"[a-zA-Z0-9_-]+", sys.argv[1]):
        sys.exit("usage: new_key.py <label>  (label: letters/digits/_/-)")
    label = sys.argv[1]

    raw = "tsb_" + secrets.token_urlsafe(32)
    h = hashlib.sha256(raw.encode()).hexdigest()
    entry = f"{label}:{h}"

    lines = ENV.read_text().splitlines() if ENV.exists() else []
    out, found = [], False
    for ln in lines:
        if ln.startswith("API_KEYS="):
            found = True
            existing = ln[len("API_KEYS="):].strip()
            existing = [e for e in existing.split(",") if e and not e.startswith(f"{label}:")]
            existing.append(entry)
            out.append("API_KEYS=" + ",".join(existing))
        else:
            out.append(ln)
    if not found:
        out.append(f"API_KEYS={entry}")
    ENV.write_text("\n".join(out) + "\n")

    print(f"label : {label}")
    print(f"key   : {raw}")
    print("Saved its hash to .env (API_KEYS). The key above is shown once — store it now.")


if __name__ == "__main__":
    main()
