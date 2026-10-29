"""Run one job end-to-end. Heavy stages (diarize, asr) run in spawned
subprocesses so model memory is returned to the OS between them (design §4).
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from app.config import settings
from app.services import jobs
from .render import markdown
from .stages import convert, merge

_ctx = mp.get_context("spawn")


def _child(queue, dotted: str, args: tuple) -> None:
    """Top-level (picklable) entry point for a spawned stage process."""
    import importlib
    try:
        mod_name, _, fn_name = dotted.rpartition(".")
        fn = getattr(importlib.import_module(mod_name), fn_name)
        fn(*args)
        queue.put(None)
    except Exception as exc:  # pragma: no cover - surfaced to parent
        import traceback
        queue.put(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def _isolated(dotted: str, *args) -> None:
    """Run the function named by `dotted` in a fresh process; raise if it fails."""
    q: mp.Queue = _ctx.Queue()
    p = _ctx.Process(target=_child, args=(q, dotted, args))
    p.start()
    err = q.get()          # blocks until child reports
    p.join()
    if err is not None:
        raise RuntimeError(err)
    if p.exitcode not in (0, None):
        raise RuntimeError(f"stage process exited with code {p.exitcode}")


def run_job(job: dict) -> None:
    job_id = job["id"]
    params = job["params"]
    work_dir = settings.work_dir / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    originals = list(work_dir.glob("original.*"))
    if not originals:
        raise FileNotFoundError(f"no uploaded file in {work_dir}")
    original = originals[0]

    # 1) convert (light, in-process)
    jobs.update_progress(job_id, "convert", 0.0)
    meta = convert.convert(original, work_dir)
    wav, duration = meta["wav"], meta["duration_s"]

    # 2) diarize (isolated)
    jobs.update_progress(job_id, "diarize", 0.0)
    _isolated("worker.stages.diarize.run", job_id, wav, duration, params)

    # 3) asr (isolated)
    jobs.update_progress(job_id, "transcribe", 0.0)
    _isolated("worker.stages.asr.run", job_id, wav, duration, params)

    # 4) merge + render (light, in-process)
    jobs.update_progress(job_id, "merge", 0.95)
    transcript = merge.run(work_dir, job_id)
    md = markdown.render(transcript, {**job, "duration_s": duration})

    settings.transcripts_dir.mkdir(parents=True, exist_ok=True)
    (settings.transcripts_dir / f"{job_id}.json").write_text(
        (work_dir / "transcript.json").read_text(), encoding="utf-8"
    )
    (settings.transcripts_dir / f"{job_id}.md").write_text(md, encoding="utf-8")

    # 5) cleanup wav, finish
    try:
        Path(wav).unlink()
    except FileNotFoundError:
        pass

    jobs.mark_done(
        job_id,
        duration_s=duration,
        detected_languages=transcript.get("languages", []),
        speakers_count=len(transcript.get("speakers", [])),
    )
