"""MCP tools exposed over Streamable HTTP at /mcp, mounted into the FastAPI app.

Tool docstrings are written for the model (design §6.2): they tell Claude when
to use language=multi, that num_speakers improves quality, and that processing
takes tens of minutes so it should call wait_for_job rather than poll.

File upload never goes through MCP (JSON args only) — the model gets a job_id
via create_upload_link / list_jobs and works with that.
"""
from __future__ import annotations

import asyncio
import contextvars
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .config import settings
from .services import jobs, transcripts

# Label of the API key behind the current MCP request, set by the auth
# middleware in main.py. Used for jobs.created_by.
current_label: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mcp_label", default="mcp"
)

mcp = FastMCP("transcribator", stateless_http=True, streamable_http_path="/")


@mcp.tool()
def create_upload_link(
    language: str = "en",
    model: str = "large-v3-turbo",
    diarization: str = "pyannote",
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    prompt: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """Create a transcription job and a one-time upload link (valid ~1 hour).

    Give the returned upload_url to the user; they open it and drop the audio
    file in. Processing starts automatically once the file is uploaded. This is
    the main path for Claude.ai / the mobile app, which cannot send a file
    through MCP.

    language: an ISO code like "en" or "ru" for a single known language;
      "auto" to detect once; "multi" for mixed-language recordings (e.g. an
      interview that switches between English and Russian). PREFER "multi" for
      mixed en+ru audio — with a fixed "en" Whisper may TRANSLATE Russian
      speech to English instead of transcribing it.
    model: "small" (fast draft), "medium", or "large-v3-turbo" (best, default).
    diarization: "pyannote" (best speaker separation, default), "sherpa"
      (faster, good when the speaker count is known), or "none" (single voice).
    num_speakers / min_speakers / max_speakers: tell the diarizer how many
      people are talking. Passing num_speakers noticeably improves quality when
      you know it (e.g. a 1-on-1 interview → num_speakers=2).
    prompt: names, terms and company names to bias the ASR (improves spelling).
    title: human-readable title for the transcript (defaults to the filename).
    """
    from .models import JobParams

    params = JobParams(
        language=language, model=model, diarization=diarization,
        num_speakers=num_speakers, min_speakers=min_speakers,
        max_speakers=max_speakers, prompt=prompt, title=title,
    )
    job = jobs.create_job(params, source_filename=None,
                          created_by=current_label.get(),
                          status="awaiting_upload", with_upload_token=True)
    token = job.pop("_upload_token")
    return {
        "job_id": job["id"],
        "upload_url": f"{settings.public_base_url}/u/{token}",
        "expires_at": job.get("_upload_expires_at"),
    }


def _status(job: dict) -> dict:
    return {
        "job_id": job["id"],
        "status": job["status"],
        "stage": job.get("stage"),
        "progress": job.get("progress"),
        "eta_s": job.get("eta_s"),
        "position_in_queue": jobs.position_in_queue(job["id"])
        if job["status"] == "queued" else None,
        "duration_s": job.get("duration_s"),
        "title": job.get("title"),
        "detected_languages": job.get("detected_languages"),
        "speakers_count": job.get("speakers_count"),
        "error": job.get("error"),
    }


@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """Return status, stage, progress (0..1), ETA and queue position for a job."""
    job = jobs.get_job(job_id)
    if not job:
        raise ValueError(f"job not found: {job_id}")
    return _status(job)


@mcp.tool()
async def wait_for_job(job_id: str, timeout_s: int = 50) -> dict:
    """Long-poll: block until the job's status changes or timeout_s elapses.

    Transcription takes tens of minutes. Use this instead of calling
    get_job_status in a loop — it saves round-trips and tokens. Call it
    repeatedly until status is "done" or "failed". timeout_s is capped at 50.
    """
    timeout_s = max(1, min(timeout_s, 50))
    job = jobs.get_job(job_id)
    if not job:
        raise ValueError(f"job not found: {job_id}")
    start_status = job["status"]
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        job = jobs.get_job(job_id)
        if not job:
            raise ValueError(f"job disappeared: {job_id}")
        if job["status"] != start_status or job["status"] in ("done", "failed", "cancelled"):
            break
        await asyncio.sleep(1.0)
    return _status(job)


@mcp.tool()
def get_transcript(job_id: str, part: int = 1) -> dict:
    """Return the Markdown transcript for a finished job, one part at a time.

    A 2-hour conversation is tens of thousands of tokens, so the transcript is
    split into parts on utterance boundaries. Read successive parts (1..parts_total)
    until you have what you need; each part repeats a short header.
    """
    job = jobs.get_job(job_id)
    if not job:
        raise ValueError(f"job not found: {job_id}")
    if not transcripts.exists(job_id):
        raise ValueError(f"transcript not ready (status={job['status']})")
    text, total = transcripts.get_part(job_id, part)
    return {"job_id": job_id, "part": max(1, min(part, total)),
            "parts_total": total, "markdown": text}


@mcp.tool()
def list_jobs(status: Optional[str] = None, limit: int = 20,
             query: Optional[str] = None) -> dict:
    """List recent jobs (newest first) with title, status, date and duration.

    status filters by one of queued/processing/done/failed/cancelled;
    query matches the title substring.
    """
    return {"jobs": [_status(j) for j in jobs.list_jobs(status, limit, query)]}


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Cancel a queued or processing job."""
    if not jobs.cancel_job(job_id):
        raise ValueError("job not cancellable")
    return _status(jobs.get_job(job_id))


@mcp.tool()
def delete_job(job_id: str) -> dict:
    """Delete a job and its transcript permanently."""
    import shutil

    job = jobs.get_job(job_id)
    if not job:
        raise ValueError(f"job not found: {job_id}")
    jobs.delete_job(job_id)
    shutil.rmtree(settings.work_dir / job_id, ignore_errors=True)
    for ext in (".json", ".md"):
        (settings.transcripts_dir / f"{job_id}{ext}").unlink(missing_ok=True)
    return {"deleted": job_id}
