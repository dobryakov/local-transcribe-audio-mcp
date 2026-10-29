"""Stage 1: normalise any input to 16 kHz mono WAV, delete the original.

Kept light enough to run in-process (no ML models loaded here).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def has_audio_stream(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return bool(out.stdout.strip())


def convert(original: Path, work_dir: Path) -> dict:
    """original -> work_dir/audio.wav (16k mono). Returns {wav, duration_s}."""
    wav = work_dir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(original),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )
    duration = probe_duration(wav)
    # original is removed immediately after successful conversion (design §5.1)
    try:
        original.unlink()
    except FileNotFoundError:
        pass
    meta = {"wav": str(wav), "duration_s": duration}
    (work_dir / "convert.json").write_text(json.dumps(meta))
    return meta
