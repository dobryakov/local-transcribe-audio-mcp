"""Worker loop: claim one queued job at a time and run the pipeline.

Usage:
    python -m worker.main          # run forever
    python -m worker.main --once   # process at most one job, then exit (tests)
"""
from __future__ import annotations

import logging
import os
import sys
import time

from app.db import init_db
from app.services import jobs
from . import pipeline

log = logging.getLogger("worker")

POLL_S = float(os.environ.get("WORKER_POLL_S", "2"))


def process_one() -> bool:
    job = jobs.claim_next()
    if not job:
        return False
    jid = job["id"]
    log.info("claimed job %s (model=%s diar=%s)",
             jid, job["params"].get("model"), job["params"].get("diarization"))
    t0 = time.perf_counter()
    try:
        pipeline.run_job(job)
        log.info("job %s done in %.1fs", jid, time.perf_counter() - t0)
    except Exception as exc:
        log.exception("job %s failed", jid)
        jobs.mark_failed(jid, str(exc))
    return True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    init_db()
    once = "--once" in sys.argv
    log.info("worker started (once=%s)", once)
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    while True:
        worked = process_one()
        if once:
            break
        if not worked:
            time.sleep(POLL_S)


if __name__ == "__main__":
    main()
