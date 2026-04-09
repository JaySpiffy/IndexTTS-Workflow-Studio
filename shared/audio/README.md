# Audio Folder Guide

## Active folders

- `speakers` - finished speaker files the app exposes in the voice picker
- `source_clips` - raw or prep-stage source clips
- `outputs` - conversation and timeline exports you may want to keep
- `temp_conversation_segments` - generated line segments used during conversation workflows
- `temp` - temporary scratch files
- `uploads` - files uploaded through the UI
- `speakers_backups` - backups of local voice files before prep or replacement

## Release rule

Voice files and source clips are local-only user assets.

Do not ship:

- `shared/audio/speakers`
- `shared/audio/source_clips`
- `shared/audio/speakers_backups`

Users should provide their own legally safe voice/source audio when running the app.

## Cleanup rule

Keep active voices in `speakers`.
Keep raw prep material in `source_clips`.
Move throwaway smoke-test outputs or prep experiments into `_archive` folders instead of leaving them mixed with user-facing files.
