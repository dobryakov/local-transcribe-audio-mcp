"""FastAPI app: REST API, one-time upload page, health. MCP mounts here later."""
from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import (
    Depends, FastAPI, File, Form, Header, HTTPException, UploadFile,
)
from fastapi.responses import HTMLResponse, PlainTextResponse

from .auth import extract_bearer, label_for_key, label_for_secret
from .config import settings
from .db import init_db
from .mcp_server import current_label, mcp
from .models import JobParams
from .services import jobs, transcripts
from worker.stages import convert as convert_stage

CHUNK = 1024 * 1024

# Build the MCP Streamable-HTTP ASGI app once; its session manager is run from
# the FastAPI lifespan below (required when the app is mounted, not standalone).
_mcp_app = mcp.streamable_http_app()


class MCPAuth:
    """ASGI middleware: Bearer-auth every /mcp request and stash the key label.

    Claude Code sends `Authorization: Bearer <key>`; remote connectors
    (Claude.ai) will use the secret-in-URL scheme added in a later stage.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        label = None
        # Path relative to the /mcp mount. Secret-in-URL (Claude.ai connector)
        # hits /mcp/k/<secret>; strip that segment so the inner app sees the
        # same path as the Bearer case.
        root = scope.get("root_path", "")
        full = scope.get("path", "")
        sub = full[len(root):] if full.startswith(root) else full
        if sub.startswith("/k/"):
            secret, _, rest = sub[len("/k/"):].partition("/")
            label = label_for_secret(secret)
            if label:
                # Rewrite to the inner "/" route WITH trailing slash, so the app
                # serves it directly instead of 307-redirecting to /mcp/ (which
                # would drop the secret from the path and fail the next request).
                new_full = root + "/" + rest
                scope = {**scope, "path": new_full, "raw_path": new_full.encode()}
        else:
            headers = dict(scope.get("headers") or [])
            token = extract_bearer(headers.get(b"authorization", b"").decode())
            label = label_for_key(token) if token else None

        if not label:
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body",
                        "body": b'{"error":"unauthorized"}'})
            return
        current_label.set(label)
        return await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="Transcribator", version="1.0", lifespan=lifespan)
app.mount("/mcp", MCPAuth(_mcp_app))


def require_key(authorization: Optional[str] = Header(None)) -> str:
    token = extract_bearer(authorization)
    label = label_for_key(token) if token else None
    if not label:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return label


def _job_params_from_form(
    language: str, model: str, diarization: str,
    num_speakers, min_speakers, max_speakers, prompt, title,
) -> JobParams:
    p = JobParams(
        language=language or settings.default_language,
        model=model or settings.default_model,
        diarization=diarization or settings.default_diarization,
        num_speakers=num_speakers, min_speakers=min_speakers,
        max_speakers=max_speakers, prompt=prompt, title=title,
    )
    try:
        p.validate()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return p


def _stream_to_disk(upload: UploadFile, dest: Path) -> int:
    size = 0
    limit = settings.max_upload_mb * 1024 * 1024
    with dest.open("wb") as fh:
        while True:
            chunk = upload.file.read(CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="file too large")
            fh.write(chunk)
    return size


def _validate_audio(path: Path) -> float:
    try:
        if not convert_stage.has_audio_stream(path):
            raise HTTPException(status_code=400, detail="no audio stream")
        duration = convert_stage.probe_duration(path)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="unreadable media file")
    if duration > settings.max_duration_min * 60:
        raise HTTPException(status_code=400, detail="audio too long")
    return duration


def _public_status(job: dict) -> dict:
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
        "created_at": job.get("created_at"),
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "models_dir": str(settings.models_dir),
            "models_present": settings.models_dir.exists()}


@app.post("/v1/jobs", status_code=202)
def create_job(
    label: str = Depends(require_key),
    file: UploadFile = File(...),
    language: str = Form(""),
    model: str = Form(""),
    diarization: str = Form(""),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    prompt: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
) -> dict:
    params = _job_params_from_form(language, model, diarization,
                                   num_speakers, min_speakers, max_speakers,
                                   prompt, title)
    job = jobs.create_job(params, source_filename=file.filename,
                          created_by=label, status="awaiting_upload")
    work = settings.work_dir / job["id"]
    work.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "audio").suffix or ".bin"
    dest = work / f"original{ext}"
    _stream_to_disk(file, dest)
    _validate_audio(dest)
    job = jobs.enqueue(job["id"], file.filename)
    return _public_status(job)


@app.get("/v1/jobs")
def list_jobs(
    label: str = Depends(require_key),
    status: Optional[str] = None,
    limit: int = 50,
    q: Optional[str] = None,
) -> dict:
    return {"jobs": [_public_status(j) for j in jobs.list_jobs(status, limit, q)]}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, label: str = Depends(require_key)) -> dict:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    out = _public_status(job)
    out["parts_total"] = transcripts.get_part(job_id)[1] if transcripts.exists(job_id) else None
    return out


@app.get("/v1/jobs/{job_id}/transcript", response_class=PlainTextResponse)
def get_transcript(job_id: str, part: int = 1, label: str = Depends(require_key)) -> str:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if not transcripts.exists(job_id):
        raise HTTPException(status_code=409, detail=f"transcript not ready (status={job['status']})")
    text, _ = transcripts.get_part(job_id, part)
    return text


@app.post("/v1/jobs/{job_id}/cancel")
def cancel(job_id: str, label: str = Depends(require_key)) -> dict:
    if not jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="job not cancellable")
    return _public_status(jobs.get_job(job_id))


@app.post("/v1/jobs/{job_id}/retry")
def retry(job_id: str, label: str = Depends(require_key)) -> dict:
    job = jobs.requeue(job_id)
    if not job:
        raise HTTPException(status_code=409, detail="only failed jobs can be retried")
    return _public_status(job)


@app.delete("/v1/jobs/{job_id}")
def delete(job_id: str, label: str = Depends(require_key)) -> dict:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    jobs.delete_job(job_id)
    shutil.rmtree(settings.work_dir / job_id, ignore_errors=True)
    for ext in (".json", ".md"):
        (settings.transcripts_dir / f"{job_id}{ext}").unlink(missing_ok=True)
    return {"deleted": job_id}


@app.post("/v1/upload-links")
def create_upload_link(
    label: str = Depends(require_key),
    language: str = Form(""),
    model: str = Form(""),
    diarization: str = Form(""),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    prompt: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
) -> dict:
    params = _job_params_from_form(language, model, diarization,
                                   num_speakers, min_speakers, max_speakers,
                                   prompt, title)
    job = jobs.create_job(params, source_filename=None, created_by=label,
                          status="awaiting_upload", with_upload_token=True)
    token = job.pop("_upload_token")
    return {
        "job_id": job["id"],
        "upload_url": f"{settings.public_base_url}/u/{token}",
        "expires_at": job.get("_upload_expires_at"),
    }


_UPLOAD_PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Загрузка записи</title>
<style>body{{font-family:system-ui,sans-serif;max-width:32rem;margin:3rem auto;padding:0 1rem}}
input[type=file]{{display:block;margin:1rem 0}}button{{padding:.6rem 1.2rem;font-size:1rem}}
.ok{{color:green}}.err{{color:#c00}}</style></head><body>
<h1>Загрузка записи</h1>
<p>Выберите аудиофайл — обработка начнётся автоматически.</p>
<form id="f"><input type="file" name="file" accept="audio/*,video/*" required>
<button type="submit">Загрузить</button></form>
<p id="msg"></p>
<script>
const f=document.getElementById('f'),m=document.getElementById('msg');
f.onsubmit=async e=>{{e.preventDefault();m.textContent='Загрузка…';m.className='';
const fd=new FormData(f);
try{{const r=await fetch(location.pathname,{{method:'POST',body:fd}});
const j=await r.json();
if(r.ok){{m.textContent='Готово! Задача '+j.job_id+' в очереди.';m.className='ok';f.style.display='none';}}
else{{m.textContent='Ошибка: '+(j.detail||r.status);m.className='err';}}
}}catch(err){{m.textContent='Ошибка сети';m.className='err';}}}};
</script></body></html>"""


@app.get("/u/{token}", response_class=HTMLResponse)
def upload_page(token: str) -> str:
    return _UPLOAD_PAGE


@app.post("/u/{token}")
def upload_via_link(token: str, file: UploadFile = File(...)) -> dict:
    # Reject an invalid/expired/used link BEFORE reading the body, so an
    # unauthenticated caller can't make us stream up to MAX_UPLOAD_MB to disk
    # and run ffprobe on a bogus token. The token is re-checked atomically in
    # attach_upload after the file lands (guards the single-use race).
    if not jobs.token_is_valid(token):
        raise HTTPException(status_code=404, detail="invalid or expired upload link")
    from .auth import hash_key
    th = hash_key(token)
    tmp = settings.work_dir / f"_pending_{th[:16]}"
    tmp.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "audio").suffix or ".bin"
    dest = tmp / f"original{ext}"
    _stream_to_disk(file, dest)
    _validate_audio(dest)
    job = jobs.attach_upload(token, file.filename or "upload")
    if not job:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(status_code=404, detail="invalid or expired upload link")
    final = settings.work_dir / job["id"]
    final.mkdir(parents=True, exist_ok=True)
    dest.rename(final / dest.name)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"job_id": job["id"], "status": job["status"]}
