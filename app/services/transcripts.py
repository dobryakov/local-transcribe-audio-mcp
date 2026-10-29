"""Read transcripts from disk and split Markdown into token-bounded parts
on utterance boundaries (design §6.2 — why the MCP tool paginates)."""
from __future__ import annotations

import math
from pathlib import Path

from ..config import settings


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def md_path(job_id: str) -> Path:
    return settings.transcripts_dir / f"{job_id}.md"


def exists(job_id: str) -> bool:
    return md_path(job_id).exists()


def _split_blocks(md: str) -> tuple[str, list[str]]:
    """Return (header, [utterance blocks]). Header = everything before the
    first timestamped line, kept and repeated on each part."""
    blocks = md.split("\n\n")
    header_parts, body = [], []
    in_body = False
    for b in blocks:
        if not in_body and b.lstrip().startswith("["):
            in_body = True
        (body if in_body else header_parts).append(b)
    return "\n\n".join(header_parts).rstrip(), [b for b in body if b.strip()]


def get_part(job_id: str, part: int = 1) -> tuple[str, int]:
    """Return (markdown_of_part, parts_total). Parts are 1-indexed."""
    md = md_path(job_id).read_text(encoding="utf-8")
    header, body = _split_blocks(md)
    budget = settings.part_tokens

    pages: list[list[str]] = [[]]
    used = _approx_tokens(header)
    for block in body:
        t = _approx_tokens(block)
        if pages[-1] and used + t > budget:
            pages.append([])
            used = _approx_tokens(header)
        pages[-1].append(block)
        used += t

    total = len(pages)
    part = max(1, min(part, total))
    chunk = pages[part - 1]
    head = header
    if total > 1:
        head = f"{header}\n\n_(часть {part} из {total})_"
    return (head + "\n\n" + "\n\n".join(chunk)).strip() + "\n", total
