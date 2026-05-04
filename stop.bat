@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

if "%INDTEXTS_USE_GPU%"=="false" (
    set "COMPOSE_FILES=-f docker-compose.yml"
) else (
    set "COMPOSE_FILES=-f docker-compose.yml -f docker-compose.gpu.yml"
)

echo [INFO] Stopping Draft to Take beta containers...
docker compose %COMPOSE_FILES% --profile llm --profile omnivoice --profile sfx down
echo [INFO] Done. Shared files and downloaded models were not deleted.
pause
