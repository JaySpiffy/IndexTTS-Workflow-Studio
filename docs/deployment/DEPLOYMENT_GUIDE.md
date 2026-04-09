# IndexTTS2 Deployment Guide

This project is Docker-first.

The supported way to run and deploy it is through the Docker stack in [../../docker/README.md](../../docker/README.md), not a host-managed Python workflow.

## Requirements

- Docker Desktop with WSL2 backend on Windows, or Docker Engine on Linux
- NVIDIA drivers and NVIDIA Container Toolkit if you want GPU acceleration
- 16GB RAM minimum, 32GB recommended
- Enough free disk space for the model bundle and generated outputs

## First Start

From the repo root:

```powershell
docker\start.bat
```

Or manually:

```powershell
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d --build
```

## Ports

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8001`
- API docs: `http://localhost:8001/docs`

The helper script can automatically choose nearby free ports if the defaults are busy.

## Model Management

The backend uses:

```text
shared/models/checkpoints
```

If required model files are missing, the backend can auto-download the official IndexTTS2 bundle on first start.

Useful environment variables:

- `INDTEXTS_AUTO_DOWNLOAD_MODELS=true|false`
- `INDTEXTS_MODEL_REPO=IndexTeam/IndexTTS-2`
- `HF_TOKEN` for authenticated Hugging Face downloads when needed

## Common Runtime Options

- `INDTEXTS_USE_GPU=true|false`
- `INDTEXTS_DEVICE=auto|cpu|cuda|cuda:0|xpu|mps`
- `INDTEXTS_USE_FP16=auto|true|false`
- `INDTEXTS_FRONTEND_HOST_PORT`
- `INDTEXTS_BACKEND_HOST_PORT`

## Common Commands

```powershell
# Start everything
docker compose -f docker/docker-compose.yml up -d --build

# Start just the backend
docker compose -f docker/docker-compose.yml up -d --build backend

# View backend logs
docker compose -f docker/docker-compose.yml logs -f backend

# Run a focused backend contract test
docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_line_emotion_contract.py

# Stop the stack
docker compose -f docker/docker-compose.yml down
```

## Backend Only Runtime

If you only want the API container during maintenance or debugging:

```powershell
docker compose -f docker/docker-compose.yml up -d --build backend
```

Then open:

- Swagger UI: [http://localhost:8001/docs](http://localhost:8001/docs)
- ReDoc: [http://localhost:8001/redoc](http://localhost:8001/redoc)
- Health: [http://localhost:8001/health](http://localhost:8001/health)

Main backend route areas:

- `GET /health`
- `POST /api/conversation/generate-single`
- `POST /api/conversation/generate`
- `GET /api/conversation/status/{conversation_id}`
- `GET /api/conversation/results/{conversation_id}`
- `POST /api/conversation/projects/save`
- `GET /api/speakers/`
- `POST /api/audio-process/*`
- `POST /api/timeline/*`

Useful backend environment variables:

- `INDTEXTS_DEVICE`
- `INDTEXTS_USE_FP16`
- `INDTEXTS_USE_GPU`
- `INDTEXTS_AUTO_DOWNLOAD_MODELS`
- `INDTEXTS_MODEL_REPO`
- `INDTEXTS_DEBUG`
- `HF_TOKEN`

## Health Checks

- Backend: `http://localhost:8001/health`
- Frontend proxy health: `http://localhost:3000/api/health`

## Production Notes

- Put a reverse proxy in front of the Dockerized frontend/backend if you need HTTPS or a domain name.
- Persist `shared/models`, `shared/audio`, and `shared/data`.
- If you deploy on a GPU host, make sure Docker can see the NVIDIA runtime before starting the stack.

## Troubleshooting

- If GPU is not available, the backend will still run in CPU mode, but startup and generation will be slower.
- If the backend takes a long time on first start, it may be downloading models or loading them on CPU.
- If model download should be disabled, set `INDTEXTS_AUTO_DOWNLOAD_MODELS=false` and pre-seed `shared/models/checkpoints` yourself.
