# Transcribator

Self-hosted speech-to-text with **speaker diarization** and **timestamps**. Turns long meeting / interview recordings into a Markdown dialogue. Runs fully offline on CPU — no paid ASR APIs.

Access is via a **REST API** and an **MCP** server so Claude (Claude Code, Claude.ai / Desktop) can create jobs, wait for them, and read transcripts.

## Features

- **ASR:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, int8). Models: `small`, `medium`, `large-v3-turbo` (default).
- **Languages:** `en`, `ru`, or `multi` (per-segment language detection for mixed en+ru).
- **Diarization:** `sherpa` (open, no token) or `pyannote` (gated HF models). Anonymous labels `SPEAKER_1`, …
- **Output:** Markdown with speaker turns and timestamps; long transcripts are split into parts for MCP clients.
- **Queue:** one job at a time; worker is CPU/RAM-capped so neighbouring services stay healthy.
- **Privacy:** audio is deleted after processing; transcripts stay until you delete them. Nothing leaves the machine.

## Requirements

- Docker + Docker Compose
- ~4+ GB RAM available for the worker (default limit 4 GB)
- CPU with AVX2/AVX-512 preferred
- Optional: an existing reverse proxy (Caddy example included) for HTTPS

## Quick start

```bash
cp .env.example .env
python scripts/new_key.py claude-code   # prints the key once; appends hash to .env

# Download diarization models (sherpa is open). For pyannote, set HF_TOKEN first.
MODELS_DIR=./models python scripts/download_models.py

docker compose up -d --build
```

API listens on `http://127.0.0.1:18080` by default (container port 8080). Set `PUBLIC_BASE_URL` in `.env` to your public origin if you put it behind a proxy.

Adjust the external `caddy` network name in `docker-compose.yml` (or remove that network and publish only on localhost) to match your host.

Operational checklist (keys for Claude.ai, Caddy snippet, cron cleanup, backup): [`deploy/README.md`](deploy/README.md).  
Design and rationale: [`docs/design.md`](docs/design.md).

## REST API

All `/v1/*` routes require `Authorization: Bearer <key>`.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/jobs` | Upload audio and enqueue a job (`202`) |
| `GET` | `/v1/jobs` | List jobs |
| `GET` | `/v1/jobs/{id}` | Status, stage, progress, ETA |
| `GET` | `/v1/jobs/{id}/transcript?part=1` | Transcript Markdown (paged) |
| `POST` | `/v1/jobs/{id}/cancel` | Cancel |
| `POST` | `/v1/jobs/{id}/retry` | Retry a failed job |
| `DELETE` | `/v1/jobs/{id}` | Delete job + transcript |
| `POST` | `/v1/upload-links` | One-time browser upload URL (TTL 1 h) |
| `GET` | `/healthz` | Liveness |

Multipart fields for `POST /v1/jobs`: `file`, optional `language`, `model`, `diarization`, `num_speakers` / `min_speakers` / `max_speakers`, `prompt`, `title`.

```bash
curl -H "Authorization: Bearer $TRANSCRIBATOR_KEY" \
  -F file=@interview.m4a -F language=multi -F num_speakers=2 \
  -F title="Backend interview" \
  http://127.0.0.1:18080/v1/jobs
```

One-time upload pages live at `/u/{token}` (no API key in the browser).

## MCP (Claude)

Streamable HTTP endpoint: `/mcp` (Bearer key) or `/mcp/k/<secret>` for Claude.ai custom connectors that cannot set headers.

**Claude Code:**

```bash
claude mcp add --transport http transcribator https://transcribe.<YOUR-HOST>.net/mcp \
  --header "Authorization: Bearer $TRANSCRIBATOR_KEY"
# on the same host: http://127.0.0.1:18080/mcp
```

**Tools:** `create_upload_link`, `get_job_status`, `wait_for_job`, `get_transcript`, `list_jobs`, `cancel_job`, `delete_job`.

Typical Claude.ai flow: ask Claude to transcribe → it calls `create_upload_link` → you open the link and drop the file → Claude polls with `wait_for_job` → `get_transcript`.

MCP does not accept file bytes (too large for the protocol). Upload via REST or the one-time link; MCP works with `job_id` afterwards.

## Architecture

```
api (FastAPI + MCP)  ←→  SQLite queue + /data
worker (one process)     convert → diarize → ASR → merge → Markdown
```

Two containers share one image and the `./data` / `./models` bind mounts. The worker has no inbound network. See `docker-compose.yml` for CPU/RAM limits.

## License

Private / personal use unless you add a license of your own.
