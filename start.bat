@echo off
setlocal EnableExtensions

echo ============================================
echo   Draft to Take Beta
echo ============================================
echo.

cd /d "%~dp0"

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo [INFO] Created .env from .env.example.
)

for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" set "%%A=%%B"
)

if "%DRAFT_TO_TAKE_HOME%"=="" set "DRAFT_TO_TAKE_HOME=%USERPROFILE%\DraftToTake"
if "%DRAFT_TO_TAKE_SHARED_DIR%"=="" set "DRAFT_TO_TAKE_SHARED_DIR=%DRAFT_TO_TAKE_HOME%\shared"

for %%d in (
    "%DRAFT_TO_TAKE_SHARED_DIR%\models"
    "%DRAFT_TO_TAKE_SHARED_DIR%\models\checkpoints"
    "%DRAFT_TO_TAKE_SHARED_DIR%\models\llm"
    "%DRAFT_TO_TAKE_SHARED_DIR%\audio"
    "%DRAFT_TO_TAKE_SHARED_DIR%\audio\speakers"
    "%DRAFT_TO_TAKE_SHARED_DIR%\audio\source_clips"
    "%DRAFT_TO_TAKE_SHARED_DIR%\audio\outputs"
    "%DRAFT_TO_TAKE_SHARED_DIR%\audio\sfx"
    "%DRAFT_TO_TAKE_SHARED_DIR%\audio\ambience"
    "%DRAFT_TO_TAKE_SHARED_DIR%\audio\music"
    "%DRAFT_TO_TAKE_SHARED_DIR%\data"
) do (
    if not exist "%%~d" mkdir "%%~d"
)

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

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [INFO] Checking NVIDIA GPU access from Docker...
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] NVIDIA GPU not detected through Docker.
    echo           Dialogue may run in CPU mode and will be much slower.
    echo           SFX/music generation is disabled unless you opt in manually.
    set "INDTEXTS_USE_GPU=false"
    set "INDTEXTS_DEVICE=cpu"
    set "COMPOSE_FILES=-f docker-compose.yml"
) else (
    echo [INFO] NVIDIA GPU detected.
    set "INDTEXTS_USE_GPU=true"
    if "%INDTEXTS_DEVICE%"=="" set "INDTEXTS_DEVICE=auto"
    if "%INDTEXTS_USE_DEEPSPEED%"=="" set "INDTEXTS_USE_DEEPSPEED=true"
    set "COMPOSE_FILES=-f docker-compose.yml -f docker-compose.gpu.yml"
)

if "%INDTEXTS_SCRIPT_LLM_ENABLED%"=="" set "INDTEXTS_SCRIPT_LLM_ENABLED=true"
if "%INDTEXTS_OMNIVOICE_ENABLED%"=="" set "INDTEXTS_OMNIVOICE_ENABLED=true"
if "%INDTEXTS_SFX_ENABLED%"=="" set "INDTEXTS_SFX_ENABLED=false"

set "COMPOSE_PROFILES_ARGS="
if /I not "%INDTEXTS_SCRIPT_LLM_ENABLED%"=="false" (
    set "COMPOSE_PROFILES_ARGS=%COMPOSE_PROFILES_ARGS% --profile llm"
    echo [INFO] Managed local Qwen sidecar: enabled
) else (
    echo [INFO] Managed local Qwen sidecar: disabled
)

if /I not "%INDTEXTS_OMNIVOICE_ENABLED%"=="false" (
    set "COMPOSE_PROFILES_ARGS=%COMPOSE_PROFILES_ARGS% --profile omnivoice"
    echo [INFO] OmniVoice sidecar: enabled
) else (
    echo [INFO] OmniVoice sidecar: disabled
)

if /I "%INDTEXTS_SFX_ENABLED%"=="true" (
    set "COMPOSE_PROFILES_ARGS=%COMPOSE_PROFILES_ARGS% --profile sfx"
    echo [WARNING] SFX/music sidecar enabled. These model-backed tools are experimental and license-dependent.
) else (
    echo [INFO] SFX/music sidecar: disabled by default for beta
)

echo.
echo [INFO] Shared files live here:
echo        %DRAFT_TO_TAKE_SHARED_DIR%
echo.
echo [INFO] Pulling and starting Draft to Take beta images...
docker compose %COMPOSE_FILES% %COMPOSE_PROFILES_ARGS% pull
if errorlevel 1 (
    echo [ERROR] Docker image pull failed.
    echo         If these are GHCR images, confirm the packages are public or run docker login ghcr.io.
    pause
    exit /b 1
)

docker compose %COMPOSE_FILES% %COMPOSE_PROFILES_ARGS% up -d
if errorlevel 1 (
    echo [ERROR] Docker Compose failed to start the beta stack.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Draft to Take beta is starting
echo ============================================
echo.
echo   Frontend:  http://localhost:%INDTEXTS_FRONTEND_HOST_PORT%
echo   Backend:   http://localhost:%INDTEXTS_BACKEND_HOST_PORT%
echo   API Docs:  http://localhost:%INDTEXTS_BACKEND_HOST_PORT%/docs
echo   Shared:    %DRAFT_TO_TAKE_SHARED_DIR%
echo.
echo   First start may download large models.
echo   Keep this window open if you want to watch logs.
echo.
echo   View logs: collect-diagnostics.bat or docker compose logs -f
echo   Stop:      stop.bat
echo ============================================
echo.

timeout /t 5 /nobreak >nul
docker compose %COMPOSE_FILES% %COMPOSE_PROFILES_ARGS% logs backend --tail 30

pause
