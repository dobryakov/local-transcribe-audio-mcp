"""Stage 4: assign each ASR word to a speaker turn, then group into utterances.

Writes transcript.json (canonical internal format, design §5.5).
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import settings


def _assign_speaker(w_start: float, w_end: float, turns: list[dict]) -> str:
    if not turns:
        return "SPEAKER_1"
    best, best_ov = None, -1.0
    for t in turns:
        ov = min(w_end, t["end"]) - max(w_start, t["start"])
        if ov > best_ov:
            best_ov, best = ov, t
    if best_ov > 0:
        return best["speaker"]
    # no overlap -> nearest turn by gap
    nearest = min(turns, key=lambda t: min(abs(t["start"] - w_end), abs(w_start - t["end"])))
    return nearest["speaker"]


def run(work_dir: Path, job_id: str) -> dict:
    words = json.loads((work_dir / "words.json").read_text())["words"]
    turns = json.loads((work_dir / "turns.json").read_text())["turns"]
    langs = json.loads((work_dir / "words.json").read_text()).get("languages", [])

    pause = settings.pause_split_s
    utterances: list[dict] = []
    cur = None
    for w in words:
        spk = _assign_speaker(w["start"], w["end"], turns)
        gap = (w["start"] - cur["end"]) if cur else 0.0
        too_long = cur and (w["end"] - cur["start"] > 60)
        if cur and spk == cur["speaker"] and gap <= pause and not too_long:
            cur["end"] = w["end"]
            cur["text"] += w["text"]
            if w.get("lang") and w["lang"] not in cur["langs"]:
                cur["langs"].append(w["lang"])
        else:
            if cur:
                utterances.append(cur)
            cur = {
                "speaker": spk, "start": w["start"], "end": w["end"],
                "text": w["text"], "langs": [w["lang"]] if w.get("lang") else [],
            }
    if cur:
        utterances.append(cur)

    for u in utterances:
        u["text"] = u["text"].strip()

    speakers = sorted({u["speaker"] for u in utterances})
    transcript = {
        "job_id": job_id,
        "languages": langs,
        "speakers": speakers,
        "utterances": utterances,
    }
    (work_dir / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False))
    return transcript
