@echo off
REM IndexTTS2 Docker Stop Script

echo ============================================
echo   Stopping IndexTTS2 Docker Stack
echo ============================================
echo.

cd /d "%~dp0"
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down

echo.
echo [INFO] IndexTTS2 stopped.
echo.
echo To remove volumes (delete all data), run:
echo   docker compose -f docker-compose.yml -f docker-compose.gpu.yml down -v
echo.

pause
