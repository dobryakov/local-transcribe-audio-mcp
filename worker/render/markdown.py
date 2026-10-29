"""Render the canonical transcript.json into the Markdown dialogue (design §5.5)."""
from __future__ import annotations

from datetime import datetime, timezone


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def render(transcript: dict, job: dict) -> str:
    params = job.get("params", {})
    title = job.get("title") or params.get("title") or f"Transcript {job['id']}"
    dur = job.get("duration_s") or (
        transcript["utterances"][-1]["end"] if transcript["utterances"] else 0
    )
    langs = ", ".join(transcript.get("languages") or ["?"])
    model = params.get("model", "?")
    diar = params.get("diarization", "?")
    n_spk = len(transcript.get("speakers") or [])
    processed = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# {title}",
        "",
        f"- Длительность: {_hms(dur)} · Языки: {langs} · Модель: {model} · "
        f"Диаризация: {diar} · Спикеров: {n_spk}",
        f"- Обработано: {processed} · job: {job['id']}",
        "",
    ]
    for u in transcript["utterances"]:
        ts = f"[{_hms(u['start'])}–{_hms(u['end'])}]"
        lines.append(f"{ts} **{u['speaker']}:** {u['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
