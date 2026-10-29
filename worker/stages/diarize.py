"""Stage 2: diarization. Writes turns.json (list of {start,end,speaker}).

Backends:
  none     -> no turns; merge assigns everything to SPEAKER_1.
  pyannote -> not wired yet (needs torch + pyannote.audio + HF token).
  sherpa   -> not wired yet (needs sherpa-onnx + models).

Speaker labels are normalised to SPEAKER_1..N by order of first appearance.
"""
from __future__ import annotations

import json
from pathlib import Path


def _normalise(turns: list[dict]) -> list[dict]:
    order: dict[str, str] = {}
    out = []
    for t in sorted(turns, key=lambda x: x["start"]):
        raw = t["speaker"]
        if raw not in order:
            order[raw] = f"SPEAKER_{len(order) + 1}"
        out.append({"start": t["start"], "end": t["end"], "speaker": order[raw]})
    return out


def run(job_id: str, wav: str, duration_s: float, params: dict) -> None:
    backend = params.get("diarization") or "none"
    work = Path(wav).parent

    if backend == "none":
        turns: list[dict] = []
    elif backend == "pyannote":
        turns = _diarize_pyannote(job_id, wav, duration_s, params)
    elif backend == "sherpa":
        turns = _diarize_sherpa(job_id, wav, duration_s, params)
    else:
        raise ValueError(f"unknown diarization backend: {backend}")

    (work / "turns.json").write_text(json.dumps({"turns": _normalise(turns)}))


def _diarize_pyannote(job_id, wav, duration_s, params) -> list[dict]:
    raise NotImplementedError(
        "pyannote backend not installed yet (needs torch + pyannote.audio + HF token). "
        "Use diarization=none or sherpa for now."
    )


def _diarize_sherpa(job_id, wav, duration_s, params) -> list[dict]:
    raise NotImplementedError(
        "sherpa backend not installed yet (needs sherpa-onnx + segmentation/embedding models)."
    )
