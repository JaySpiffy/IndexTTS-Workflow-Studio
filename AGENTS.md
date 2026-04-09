# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Direction

This repository is a Docker-first local application built on top of the official IndexTTS2 models.

Prefer:
- Docker Compose for runtime verification
- backend container commands for Python-based tests and maintenance
- the mounted `shared/models/checkpoints` directory for model management

Avoid teaching or reintroducing a host-managed Python setup unless the user explicitly asks for it.

## Runtime Layout

- Backend: `backend/main.py`
- Frontend: `frontend/`
- Docker stack: `docker/docker-compose.yml`
- Backend image build: `docker/Dockerfile.backend`
- Model bootstrap: `docker/download_models.py`

## Main Commands

```powershell
# Start the full app
docker\start.bat

# Start the stack manually
docker compose -f docker/docker-compose.yml up -d --build

# Start only the backend
docker compose -f docker/docker-compose.yml up -d --build backend

# View backend logs
docker compose -f docker/docker-compose.yml logs -f backend

# Run a backend test inside the container
docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_line_emotion_contract.py
```

## Model Notes

- Default model path: `shared/models/checkpoints`
- The backend can auto-download the official model bundle on first start
- `HF_TOKEN` can be passed through Docker when Hugging Face authentication is required
- `INDTEXTS_AUTO_DOWNLOAD_MODELS=false` disables automatic download

## Runtime Gotchas

- Frontend default host port: `3000`
- Backend default host port: `8001`
- `INDTEXTS_DEVICE=auto` is the default device-selection mode
- The Docker image is NVIDIA/CUDA-based today
- CPU fallback works, but startup and generation are much slower
- Random sampling lowers voice-cloning fidelity

## Maintenance Priorities

- Keep docs aligned with Docker-first usage
- Prefer fixing frontend/backend contract drift before adding new features
- When changing generation behavior, add or update focused backend tests
