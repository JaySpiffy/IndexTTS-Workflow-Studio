# Release Hygiene Checklist

Use this checklist before promoting v2 as the default public version.

## Docs And Entry Points
- Confirm [../../README.md](../../README.md) matches the actual repo purpose and does not present this project as the official upstream IndexTTS repository.
- Confirm Docker docs use the current exposed ports: frontend `3000`, backend `8001`.
- Confirm model download docs point to `shared/models/checkpoints`.
- Confirm no deleted or legacy entrypoints remain referenced in markdown or helper scripts.

## Docker And Config
- Confirm [../../pyproject.toml](../../pyproject.toml) metadata describes the app honestly.
- Confirm `INDTEXTS_DEVICE`, `INDTEXTS_USE_FP16`, `INDTEXTS_AUTO_DOWNLOAD_MODELS`, and `INDTEXTS_MODEL_REPO` are documented consistently.
- Confirm the backend skips model download cleanly when checkpoints are already present.
- Confirm missing checkpoints trigger automatic model download when `INDTEXTS_AUTO_DOWNLOAD_MODELS=true`.

## Functional Verification
- Run the manual smoke packs in [../testing/README.md](../testing/README.md) for startup plus each major tab.
- Run focused backend contract tests:
  - `docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_line_emotion_contract.py`
  - `docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_seed_strategy_contract.py`
  - `docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_infer_seed_helper_contract.py`
  - `docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_project_save_load.py`
  - `docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_review_regeneration_contract.py`
  - `docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_similarity_retry_sampling.py`
  - `docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_source_clip_prep_contract.py`
  - `docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_pacing_contract.py`
  - `docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_audio_leveling_contract.py`
  - `docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_export_finishing_contract.py`
  - `docker compose -f docker/docker-compose.yml exec backend python tests/backend/test_selection_gating_contract.py`
- Verify Docker health endpoints:
  - frontend proxy health at `http://localhost:3000/api/health`
  - backend health at `http://localhost:8001/health`
- Run a browser smoke test for:
  - load app
  - show available voices
  - confirm header shows actual runtime status, for example `GPU: cuda:0 + DeepSpeed`
  - parse script
  - generate conversation
  - verify progress bar advances during generation
  - play audio
  - save project
  - load project
  - regenerate a line
  - verify pacing preset changes the outgoing generation settings
  - verify speaker prep tab diagnoses and prepares a source clip
  - verify selection gating blocks export until every line has one chosen version
  - verify the results page shows per-version seeds plus the seed report export block
  - verify conversation concatenate succeeds after final version selection
  - verify timeline preview/export still works

## UX And Error Handling
- Check empty states for voices, projects, and conversations.
- Check that API errors render useful text instead of generic object dumps.
- Check that tab state, cache-busting, and first-load behavior work in a fresh browser session.
- Check that audio playback works from the review screen.
- Check that frontend rebuilds do not leave stale bundle behavior in a fresh browser session.

## Release Readiness
- Review [V1_TO_V2_PARITY_CHECKLIST.md](V1_TO_V2_PARITY_CHECKLIST.md) and confirm remaining gaps are either closed or documented.
- Confirm [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) still matches the real app.
- Run `powershell -ExecutionPolicy Bypass -File tools/audit_private_assets.ps1` and confirm no unexpected audio files exist outside the local-only folders.
- Write down known limitations before publishing.
- Tag or preserve the legacy v1 state before switching the public default branch.

## Last Hardening Notes

Items already verified during the latest hardening round:

- strict final-selection gating for export
- per-version seed display and seed report export
- pacing presets in the main workflow
- speaker prep tab with diagnostics and prepared output
- richer export pipeline for WAV / MP3 / OGG
- live UI showing `GPU + DeepSpeed` runtime state
