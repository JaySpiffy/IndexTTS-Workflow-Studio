@echo off
REM IndexTTS2 Docker Logs Viewer

cd /d "%~dp0"
docker compose logs -f
