# Deployment

See [`../docs/design.md`](../docs/design.md) for the full design. This is the
operational checklist.

## 1. Configure

```bash
cp .env.example .env
python scripts/new_key.py claude-code    # prints the key once, saves its hash
python scripts/new_key.py laptop
python scripts/new_key.py claudeai
```

For the Claude.ai MCP connector, generate a URL secret and add it to
`MCP_URL_SECRETS` (same `label:sha256` format as `API_KEYS`):

```bash
python - <<'PY'
import secrets, hashlib
s = "tsb_" + secrets.token_urlsafe(32)
print("secret (put in the connector URL):", s)
print("MCP_URL_SECRETS entry:", "claudeai:" + hashlib.sha256(s.encode()).hexdigest())
PY
```

The connector URL is then `https://transcribe.dobryakov.net/mcp/k/<secret>`.
Claude Code instead uses a Bearer key:

```bash
claude mcp add --transport http transcribator https://transcribe.dobryakov.net/mcp \
  --header "Authorization: Bearer $TRANSCRIBATOR_KEY"
# on this server, use http://127.0.0.1:8080/mcp
```

## 2. Download models

```bash
# sherpa (open). Add HF_TOKEN to .env first for the gated pyannote pipeline.
MODELS_DIR=./models python scripts/download_models.py
```

## 3. Run

```bash
docker compose up -d --build               # sherpa-only image
# with pyannote baked in (heavier, needs HF_TOKEN at build to pre-cache):
# docker compose build --build-arg WITH_PYANNOTE=1 && docker compose up -d
```

Add the `transcribe.dobryakov.net` block from [`Caddyfile`](Caddyfile) to the
existing Caddy stack and reload it.

## 4. Cron housekeeping

Clean up stale jobs/work dirs every 30 min:

```cron
*/30 * * * * docker exec transcribator-worker python /srv/scripts/cleanup.py >> /var/log/transcribator-cleanup.log 2>&1
```

## 5. Backup

Back up the transcripts and the database together:

```bash
tar czf transcribator-backup-$(date +%F).tgz -C ./data transcripts db.sqlite
```

Audio is never retained (original deleted after convert, WAV after success, or
after 24 h on failure). Models in `./models` are re-downloadable and need no
backup.

## Health

`GET /healthz` reports liveness and whether the models dir is present — point
your monitor at it.
