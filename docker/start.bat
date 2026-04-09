@echo off
REM IndexTTS2 Docker Startup Script

echo ============================================
echo   IndexTTS2 Docker Stack
echo ============================================
echo.

if "%INDTEXTS_FRONTEND_HOST_PORT%"=="" (
    for /f %%i in ('powershell -NoProfile -Command "$ports = 3000..3010; foreach ($port in $ports) { if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) { $port; break } }"') do set "INDTEXTS_FRONTEND_HOST_PORT=%%i"
)

if "%INDTEXTS_BACKEND_HOST_PORT%"=="" (
    for /f %%i in ('powershell -NoProfile -Command "$ports = 8001..8010; foreach ($port in $ports) { if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) { $port; break } }"') do set "INDTEXTS_BACKEND_HOST_PORT=%%i"
)

if "%INDTEXTS_FRONTEND_HOST_PORT%"=="" (
    echo [ERROR] Could not find a free frontend host port in the 3000-3010 range.
    pause
    exit /b 1
)

if "%INDTEXTS_BACKEND_HOST_PORT%"=="" (
    echo [ERROR] Could not find a free backend host port in the 8001-8010 range.
    pause
    exit /b 1
)

REM Check Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

REM Check for NVIDIA GPU
echo [INFO] Checking NVIDIA GPU...
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] NVIDIA GPU not detected or NVIDIA Container Toolkit not installed.
    echo           The application will run in CPU mode (slower).
    echo.
    set "INDTEXTS_USE_GPU=false"
    set "INDTEXTS_DEVICE=cpu"
    set "COMPOSE_FILES=-f docker-compose.yml"
) else (
    echo [INFO] NVIDIA GPU detected. Starting with GPU support.
    echo.
    set "INDTEXTS_USE_GPU=true"
    if "%INDTEXTS_DEVICE%"=="" set "INDTEXTS_DEVICE=auto"
    if "%INDTEXTS_USE_DEEPSPEED%"=="" set "INDTEXTS_USE_DEEPSPEED=true"
    set "COMPOSE_FILES=-f docker-compose.yml -f docker-compose.gpu.yml"
)

REM Check if models exist
echo [INFO] Checking for models...
if not exist "..\\shared\\models\\checkpoints\\gpt.pth" (
    echo [WARNING] Model files not found in shared/models/checkpoints/
    echo           Models will be downloaded on first run (~5GB).
    echo.
)

echo [INFO] Starting IndexTTS2...
echo [INFO] Frontend host port: %INDTEXTS_FRONTEND_HOST_PORT%
echo [INFO] Backend host port:  %INDTEXTS_BACKEND_HOST_PORT%
echo.

cd /d "%~dp0"
docker compose %COMPOSE_FILES% up -d --build

echo.
echo ============================================
echo   IndexTTS2 is starting...
echo ============================================
echo.
echo   Frontend:  http://localhost:%INDTEXTS_FRONTEND_HOST_PORT%
echo   Backend:   http://localhost:%INDTEXTS_BACKEND_HOST_PORT%
echo   API Docs:  http://localhost:%INDTEXTS_BACKEND_HOST_PORT%/docs
echo   Device:    %INDTEXTS_DEVICE%
echo   DeepSpeed: %INDTEXTS_USE_DEEPSPEED%
echo.
echo   View logs: docker compose logs -f
echo   Stop:      docker compose down
echo ============================================
echo.

REM Wait and show logs
timeout /t 5 /nobreak >nul
docker compose %COMPOSE_FILES% logs backend --tail 20

pause
