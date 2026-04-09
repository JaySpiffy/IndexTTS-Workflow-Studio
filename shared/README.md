# Shared Runtime Layout

This folder holds runtime data used by the Docker app.

## Main folders

- `audio/speakers` - active speaker files used by the app
- `audio/speakers_backups` - backups of original speaker files before prep/replacement
- `audio/source_clips` - raw clips and prep inputs
- `audio/outputs` - exported or generated audio outputs worth keeping
- `audio/temp_conversation_segments` - working per-line generation segments
- `audio/temp` - temporary processing scratch space
- `audio/uploads` - imported files uploaded through the UI
- `data/project_saves` - saved conversation projects
- `data/timeline_projects` - saved timeline editor projects
- `models/checkpoints` - active IndexTTS model files
- `models/pretrained` - other supporting pretrained assets

## Archive folders

Where practical, old smoke-test files and one-off prep outputs should be moved into `_archive` subfolders instead of staying mixed with active runtime files.
