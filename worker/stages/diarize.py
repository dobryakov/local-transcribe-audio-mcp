"""Stage 2: diarization. Writes turns.json (list of {start,end,speaker}).

Backends:
  none     -> no turns; merge assigns everything to SPEAKER_1.
  pyannote -> not wired yet (needs torch + pyannote.audio + HF token).
  sherpa   -> not wired yet (needs sherpa-onnx + models).

Speaker labels are normalised to SPEAKER_1..N by order of first appearance.
"""
from __future__ import annotations

import json
import os
import wave
from pathlib import Path

from app.config import settings


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
    # Keep model/HF caches under MODELS_DIR so they persist in the volume and
    # can be used offline at runtime (HF_HUB_OFFLINE=1).
    os.environ.setdefault("HF_HOME", str(settings.models_dir / "hf"))

    import numpy as np
    import torch
    from pyannote.audio import Pipeline

    from app.services import jobs

    token = settings.hf_token or None
    try:
        pipe = Pipeline.from_pretrained(settings.pyannote_pipeline, token=token)
    except Exception:
        # fall back to the older public pipeline if community-1 is unavailable
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    pipe.to(torch.device("cpu"))

    jobs.update_progress(job_id, "diarize", 0.05)

    data, sr = _read_wav_f32(wav)
    audio = {"waveform": torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0),
             "sample_rate": sr}

    kwargs: dict = {}
    if params.get("num_speakers"):
        kwargs["num_speakers"] = int(params["num_speakers"])
    else:
        if params.get("min_speakers"):
            kwargs["min_speakers"] = int(params["min_speakers"])
        if params.get("max_speakers"):
            kwargs["max_speakers"] = int(params["max_speakers"])

    result = pipe(audio, **kwargs)
    turns = result.serialize()["diarization"]
    return [{"start": float(t["start"]), "end": float(t["end"]),
             "speaker": t["speaker"]} for t in turns]


def _read_wav_f32(path: str):
    """Read a 16 kHz mono PCM WAV (as produced by the convert stage) into a
    float32 numpy array in [-1, 1]."""
    import numpy as np

    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data, sr


def _diarize_sherpa(job_id, wav, duration_s, params) -> list[dict]:
    import sherpa_onnx as so

    from app.services import jobs

    seg = settings.models_dir / "sherpa" / "segmentation.onnx"
    emb = settings.models_dir / "sherpa" / "embedding.onnx"
    if not seg.exists() or not emb.exists():
        raise FileNotFoundError(
            f"sherpa models missing under {settings.models_dir}/sherpa; "
            "run: MODELS_DIR=... python scripts/download_models.py"
        )

    num = params.get("num_speakers")
    threshold = float(os.environ.get("SHERPA_CLUSTER_THRESHOLD", "0.5"))
    clustering = so.FastClusteringConfig(
        num_clusters=int(num) if num else -1, threshold=threshold
    )
    config = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)),
        ),
        embedding=so.SpeakerEmbeddingExtractorConfig(
            model=str(emb), num_threads=settings.cpu_threads
        ),
        clustering=clustering,
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    sd = so.OfflineSpeakerDiarization(config)

    samples, sr = _read_wav_f32(wav)
    if sr != sd.sample_rate:
        raise ValueError(f"expected {sd.sample_rate} Hz wav, got {sr}")

    def _progress(processed: int, total: int, *_) -> int:
        try:
            jobs.update_progress(job_id, "diarize", min(processed / max(total, 1), 0.99))
        except Exception:
            pass
        return 0

    result = sd.process(samples, callback=_progress)
    return [
        {"start": float(s.start), "end": float(s.end), "speaker": f"spk{int(s.speaker)}"}
        for s in result.sort_by_start_time()
    ]
