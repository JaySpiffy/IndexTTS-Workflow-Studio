#!/bin/sh
set -eu

MODEL_DIR="${INDTEXTS_MODEL_PATH:-/app/shared/models/checkpoints}"
AUTO_DOWNLOAD="${INDTEXTS_AUTO_DOWNLOAD_MODELS:-true}"
MODEL_REPO="${INDTEXTS_MODEL_REPO:-IndexTeam/IndexTTS-2}"

if [ "$AUTO_DOWNLOAD" = "true" ] || [ "$AUTO_DOWNLOAD" = "1" ]; then
    python3.10 /app/docker/download_models.py --model-dir "$MODEL_DIR" --repo-id "$MODEL_REPO"
else
    echo "[startup] Model auto-download disabled. Expecting checkpoints in $MODEL_DIR"
fi

exec python3.10 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
