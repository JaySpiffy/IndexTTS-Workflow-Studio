# API Summary

This is the current high-level API summary for the Docker-first IndexTTS2 app.

For the exact live schema, use:

- Swagger UI: [http://localhost:8001/docs](http://localhost:8001/docs)
- ReDoc: [http://localhost:8001/redoc](http://localhost:8001/redoc)
- Health: [http://localhost:8001/health](http://localhost:8001/health)

## Base URL

```text
http://localhost:8001
```

Most app routes live under `/api/...`.

## Main Route Groups

### Health

- `GET /health`
- `GET /api/health`

### Speakers

- `GET /api/speakers/`
- `POST /api/speakers/upload`
- `GET /api/speakers/{speaker_name}`
- `GET /api/speakers/{speaker_name}/audio`
- `DELETE /api/speakers/{speaker_name}`
- `POST /api/speakers/{speaker_name}/validate`

### Speaker Prep

- `GET /api/speakers-tools/list-source-clips`
- `GET /api/speakers-tools/source-clip-diagnostics/{filename}`
- `POST /api/speakers-tools/upload-source-clip`
- `POST /api/speakers-tools/process-source-clip`
- `POST /api/speakers-tools/create-speaker-from-source`

### Conversation Workflow

- `POST /api/conversation/parse-script`
- `POST /api/conversation/generate`
- `GET /api/conversation/status/{conversation_id}`
- `GET /api/conversation/list`
- `GET /api/conversation/results/{conversation_id}`
- `POST /api/conversation/results/{conversation_id}/line/{line_number}/select-version`
- `POST /api/conversation/results/{conversation_id}/line/{line_number}/regenerate`
- `POST /api/conversation/results/{conversation_id}/concatenate`
- `GET /api/conversation/results/{conversation_id}/download`

### Saved Projects

- `GET /api/conversation/projects`
- `GET /api/conversation/projects/{save_name}`
- `POST /api/conversation/projects/save`
- `DELETE /api/conversation/projects/{save_name}`

### Audio Processing

- `POST /api/audio-process/similarity-analysis`
- `POST /api/audio-process/compare-versions`
- `GET /api/audio-process/model-status`

### Timeline

- `GET /api/timeline/list`
- `POST /api/timeline/create`
- `GET /api/timeline/{project_id}`
- `POST /api/timeline/{project_id}/tracks`
- `POST /api/timeline/{project_id}/segments`
- `POST /api/timeline/{project_id}/export`

### Emotion

- `POST /api/emotion-detection/detect`
- `POST /api/emotion-timeline/*`

## Runtime Notes

- This API is meant to run through Docker Compose, not a host-managed Python setup.
- The backend model path defaults to `shared/models/checkpoints`.
- The health payload reports the actual runtime device and whether DeepSpeed is active.
- Use the frontend at [http://localhost:3000](http://localhost:3000) for the full workflow unless you are integrating the API directly.

## Focused Maintenance Commands

```powershell
# Backend only
docker compose -f docker/docker-compose.yml up -d --build backend

# View logs
docker compose -f docker/docker-compose.yml logs -f backend

# Run a focused backend contract test
docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_line_emotion_contract.py
```

## Note On Older Docs

Older long-form API notes have been moved to the archive because they drifted from the current route structure and runtime defaults.
