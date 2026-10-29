#!/usr/bin/env python3
"""Download model weights into MODELS_DIR.

Currently fetches the sherpa-onnx diarization models (open, no token):
  - pyannote segmentation 3.0 (ONNX)
  - NeMo TitaNet-small speaker embeddings (English)

faster-whisper ASR models are fetched lazily on first use into the HF cache.
pyannote.audio (gated) will be added here once a backend is wired in; it needs
HF_TOKEN and a one-time acceptance of the model terms.

Usage: MODELS_DIR=./models python scripts/download_models.py
"""
from __future__ import annotations

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEG_URL = f"{BASE}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_URL = f"{BASE}/speaker-recongition-models/nemo_en_titanet_small.onnx"  # tag is misspelled upstream

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
SHERPA_DIR = MODELS_DIR / "sherpa"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, tempfile.NamedTemporaryFile(
        delete=False, dir=dest.parent
    ) as tmp:
        tmp.write(r.read())
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)


def fetch_sherpa() -> None:
    seg = SHERPA_DIR / "segmentation.onnx"
    emb = SHERPA_DIR / "embedding.onnx"

    if seg.exists():
        print(f"segmentation: present ({seg})")
    else:
        print("segmentation: fetching pyannote-segmentation-3.0")
        with tempfile.TemporaryDirectory() as td:
            tarball = Path(td) / "seg.tar.bz2"
            _download(SEG_URL, tarball)
            with tarfile.open(tarball, "r:bz2") as tf:
                tf.extractall(td)  # noqa: S202 - trusted upstream release
            model = next(Path(td).rglob("model.onnx"))
            seg.parent.mkdir(parents=True, exist_ok=True)
            seg.write_bytes(model.read_bytes())
        print(f"  -> {seg}")

    if emb.exists():
        print(f"embedding: present ({emb})")
    else:
        print("embedding: fetching nemo_en_titanet_small")
        _download(EMB_URL, emb)
        print(f"  -> {emb}")


def fetch_pyannote() -> None:
    """Pre-download the gated pyannote pipeline into the HF cache under
    MODELS_DIR so the worker can run offline. Needs HF_TOKEN and a one-time
    acceptance of the model terms on huggingface.co."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("pyannote: skipped (no HF_TOKEN set)")
        return
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    name = os.environ.get("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-community-1")
    print(f"pyannote: fetching {name} into {os.environ['HF_HOME']}")
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("pyannote: pyannote.audio not installed (pip install -r requirements-pyannote.txt)")
        return
    Pipeline.from_pretrained(name, token=token)
    print("  -> cached")


def main() -> None:
    print(f"MODELS_DIR = {MODELS_DIR}")
    fetch_sherpa()
    fetch_pyannote()
    print("done.")


if __name__ == "__main__":
    main()
