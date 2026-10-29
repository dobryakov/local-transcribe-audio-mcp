"""Stage 3: ASR with faster-whisper. Writes words.json.

Runs in its own (spawned) process so CTranslate2 memory is freed on exit.
Progress is written straight to SQLite (WAL is multi-process safe).
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import settings


def run(job_id: str, wav: str, duration_s: float, params: dict) -> None:
    from faster_whisper import WhisperModel

    from app.services import jobs

    language = params.get("language") or "en"
    model_name = params.get("model") or settings.default_model
    prompt = params.get("prompt")

    lang_arg = None if language in ("auto", "multi") else language
    multilingual = language == "multi"
    beam_size = 1 if model_name == "small" else 5

    model = WhisperModel(
        model_name, device="cpu", compute_type="int8",
        cpu_threads=settings.cpu_threads,
        download_root=str(settings.models_dir) if settings.models_dir.exists() else None,
    )
    segments, info = model.transcribe(
        wav,
        language=lang_arg,
        multilingual=multilingual,
        beam_size=beam_size,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=True,
        initial_prompt=prompt,
        hotwords=prompt if prompt else None,
    )

    words: list[dict] = []
    langs: set[str] = set()
    dur = duration_s or info.duration or 1.0
    for seg in segments:
        seg_lang = getattr(seg, "language", None) or info.language
        if seg_lang:
            langs.add(seg_lang)
        if seg.words:
            for w in seg.words:
                words.append({"start": w.start, "end": w.end,
                              "text": w.word, "lang": seg_lang})
        else:  # VAD/model returned no word timing for this segment
            words.append({"start": seg.start, "end": seg.end,
                          "text": seg.text, "lang": seg_lang})
        jobs.update_progress(job_id, "transcribe", min(seg.end / dur, 0.99))

    work = Path(wav).parent
    (work / "words.json").write_text(
        json.dumps({"words": words, "languages": sorted(langs)}, ensure_ascii=False)
    )
