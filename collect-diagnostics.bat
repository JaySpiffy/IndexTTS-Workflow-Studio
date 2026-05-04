@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if "%DRAFT_TO_TAKE_HOME%"=="" set "DRAFT_TO_TAKE_HOME=%USERPROFILE%\DraftToTake"
if "%DRAFT_TO_TAKE_SHARED_DIR%"=="" set "DRAFT_TO_TAKE_SHARED_DIR=%DRAFT_TO_TAKE_HOME%\shared"

set "OUT_DIR=%DRAFT_TO_TAKE_HOME%\diagnostics"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "STAMP=%%i"
set "OUT_FILE=%OUT_DIR%\draft-to-take-beta-%STAMP%.txt"

echo [INFO] Writing diagnostics to:
echo        %OUT_FILE%
echo.

(
    echo Draft to Take Beta Diagnostics
    echo Generated: %DATE% %TIME%
    echo.
    echo == System ==
    ver
    echo.
    echo == Docker Version ==
    docker version
    echo.
    echo == Docker Compose Version ==
    docker compose version
    echo.
    echo == GPU Check ==
    docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
    echo.
    echo == Compose PS ==
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile llm --profile omnivoice --profile sfx ps
    echo.
    echo == Backend Logs ==
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile llm --profile omnivoice --profile sfx logs --tail 250 backend
    echo.
    echo == Frontend Logs ==
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile llm --profile omnivoice --profile sfx logs --tail 100 frontend
    echo.
    echo == Script LLM Logs ==
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile llm logs --tail 150 script-llm
    echo.
    echo == OmniVoice Logs ==
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile omnivoice logs --tail 150 omnivoice
    echo.
    echo == SFX Logs ==
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile sfx logs --tail 150 sfx
) > "%OUT_FILE%" 2>&1

echo [INFO] Diagnostics collected.
echo [INFO] Please review the file before posting it publicly. Do not share private scripts, voices, tokens, or personal paths if you are not comfortable with them.
pause
