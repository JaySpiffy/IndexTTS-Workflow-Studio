# IndexTTS2 Docker Deployment

Quick start guide for running IndexTTS2 with Docker.

The bundled startup flow is GPU-first:

- if Docker can see your NVIDIA runtime, the backend starts on GPU
- if not, the app falls back to CPU so it still comes up

## Prerequisites

- Docker Desktop with WSL2 backend (Windows) or Docker Engine (Linux)
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU with CUDA 12.8 support
- At least 16GB RAM, 32GB recommended
- 50GB+ disk space for models

### Install NVIDIA Container Toolkit (Windows)

1. Install latest NVIDIA drivers
2. In Docker Desktop: Settings > General > enable "Use WSL 2 based engine"
3. Settings > Resources > WSL Integration > enable for your distro

### Install NVIDIA Container Toolkit (Linux)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Quick Start

### Option 1: Use pre-existing models (recommended)

If you already have models downloaded:

```powershell
# From the repo's docker folder
docker compose up -d
```

### Option 2: First-time setup with model download

```powershell
# Build and run (the backend downloads the official IndexTTS2 model bundle into shared/models/checkpoints on first start)
docker compose up -d --build

# Watch logs
docker compose logs -f backend
```

If you want to disable the automatic download and provide the models yourself, set:

```powershell
$env:INDTEXTS_AUTO_DOWNLOAD_MODELS="false"
```

### Access the Application

- **Frontend UI:** http://localhost:3000
- **Backend API:** http://localhost:8001
- **API Docs:** http://localhost:8001/docs

The bundled `start.bat` script auto-picks the next free frontend port in the `3000-3010` range and the next free backend port in the `8001-8010` range, so it can usually start cleanly even if other local services already occupy the defaults.

## Architecture

```
┌─────────────────────────────────────────┐
│  Frontend (Nginx) :80                   │
│  - Serves static HTML/CSS/JS            │
│  - Proxies /api/* to backend            │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Backend (FastAPI) :8000                │
│  - IndexTTS2 engine                     │
│  - REST API endpoints                   │
│  - GPU accelerated inference            │
└─────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Bind-Mounted Project Data              │
│  - shared/models (TTS models + HF cache)│
│  - shared/audio/outputs                 │
│  - shared/audio/speakers                │
│  - shared/audio/source_clips/temp/uploads │
└─────────────────────────────────────────┘
```

## Data Management

The stack uses bind-mounted folders from the repo, so generated files and downloaded models stay on disk next to the project.

```powershell
Get-ChildItem ..\shared\models
```

### Backup outputs

```powershell
Compress-Archive -Path ..\shared\audio\outputs\* -DestinationPath ..\shared\audio\outputs-backup.zip -Force
```

### Clean up (WARNING: deletes all data)

```powershell
docker compose down -v
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| INDTEXTS_USE_GPU | true | Enable GPU acceleration |
| INDTEXTS_DEVICE | auto | Device selection: auto, cpu, cuda, cuda:0, xpu, or mps |
| INDTEXTS_USE_FP16 | auto | Precision mode: auto, true, or false |
| INDTEXTS_USE_DEEPSPEED | true | Enable DeepSpeed inference acceleration inside the backend container |
| INDTEXTS_INSTALL_DEEPSPEED | true | Install the DeepSpeed package into the backend image during `docker compose build` |
| INDTEXTS_DEBUG | false | Enable debug logging |
| INDTEXTS_AUTO_DOWNLOAD_MODELS | true | Automatically download the official IndexTTS2 model bundle when required files are missing |
| INDTEXTS_MODEL_REPO | IndexTeam/IndexTTS-2 | Hugging Face repo used for automatic model download |
| INDTEXTS_PORT | 8000 | Backend port inside the container |
| INDTEXTS_FRONTEND_HOST_PORT | 3000 | Frontend port exposed on your host |
| INDTEXTS_BACKEND_HOST_PORT | 8001 | Backend port exposed on your host |

`INDTEXTS_DEVICE=auto` lets the Python runtime pick the best supported accelerator it can see. Today the app can choose between `cuda`, `xpu`, `mps`, and `cpu` in code, but this Docker stack is still built on an NVIDIA CUDA image, so non-NVIDIA accelerators would need a different container base image and runtime setup.

DeepSpeed is the default in the Docker GPU path. The first DeepSpeed-enabled startup can take longer while extensions warm up or compile, but the compose stack mounts persistent Torch extension and Triton cache volumes so those compiled artifacts survive container recreation. If DeepSpeed initialization fails on a given run, the backend falls back to standard GPU inference instead of taking the API down.

In the normal `docker\start.bat` flow, Docker GPU detection happens before the containers start:

- NVIDIA runtime available: `INDTEXTS_USE_GPU=true` and `INDTEXTS_DEVICE=auto`
- NVIDIA runtime unavailable: `INDTEXTS_USE_GPU=false` and `INDTEXTS_DEVICE=cpu`

### Edit docker-compose.yml

```yaml
environment:
  INDTEXTS_USE_GPU: true
  INDTEXTS_DEVICE: auto
  INDTEXTS_USE_FP16: auto
  INDTEXTS_DEBUG: false
```

### Common Overrides

Run these from the `docker/` folder. Keep the `docker-compose.gpu.yml` override in GPU commands so Docker passes the NVIDIA device into the backend container.

```powershell
# Force CPU mode inside the container
$env:INDTEXTS_USE_GPU="false"; $env:INDTEXTS_DEVICE="cpu"; docker compose -f docker-compose.yml up -d

# Force a specific NVIDIA GPU
$env:INDTEXTS_DEVICE="cuda:1"; docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Force FP16 on supported accelerators
$env:INDTEXTS_USE_FP16="true"; docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Disable DeepSpeed but keep GPU inference enabled
$env:INDTEXTS_USE_DEEPSPEED="false"; docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Build a smaller backend image without DeepSpeed installed
$env:INDTEXTS_INSTALL_DEEPSPEED="false"; docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# Override the host ports if you do want the classic mappings
$env:INDTEXTS_FRONTEND_HOST_PORT="80"; $env:INDTEXTS_BACKEND_HOST_PORT="8000"; docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Disable automatic model download
$env:INDTEXTS_AUTO_DOWNLOAD_MODELS="false"; docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## Troubleshooting

### GPU not detected

```powershell
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Models not loading

Check the mounted model directory:

```powershell
Get-ChildItem ..\shared\models\checkpoints
```

### Out of memory

Reduce batch size or enable FP16:

```yaml
environment:
  INDTEXTS_USE_FP16: true
```

### View logs

```powershell
docker compose logs -f backend
docker compose logs -f frontend
```

## Development

### Rebuild after code changes

```powershell
docker compose up -d --build
```

### Run backend only (for debugging)

```powershell
docker compose up backend
```

### Shell into container

```powershell
docker compose exec backend /bin/bash
```

## Port Reference

| Service | Port | Purpose |
|---------|------|---------|
| Frontend | 3000 | Default host port for the web UI |
| Backend | 8001 | Default host port for the API |

