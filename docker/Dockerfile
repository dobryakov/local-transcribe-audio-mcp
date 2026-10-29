# One image for both api and worker (command differs per service).
FROM python:3.12-slim

# ffmpeg/ffprobe for convert + upload validation.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Install deps first for layer caching. No torch here — the pyannote backend
# (which needs it) is added in a later stage; sherpa-onnx is CPU/ONNX only.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY worker ./worker
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    MODELS_DIR=/models

EXPOSE 8080

# Default command runs the API; the worker service overrides it in compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
