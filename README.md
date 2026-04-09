<div align="center">
  <img src="assets/index_icon.png" width="180" alt="IndexTTS2 Workflow Studio icon"/>

# IndexTTS2 Workflow Studio

Docker-first local workflow studio for speaker prep, multi-speaker generation, review and regeneration, timeline editing, and polished export on top of the official IndexTTS2 models.

**Bring your own voices.** This repo does not ship bundled voice clones or private speaker files.
</div>

## What This Repo Is

This repository is **not** the official IndexTTS model repository.

It is a practical local app built **on top of** the official IndexTTS ecosystem, with:

- `Speaker Prep` for cleaning and evaluating source clips
- `Conversation Workflow` for fast multi-speaker script generation
- `Conversation Results` for review, version selection, and regeneration
- `Timeline Editor` for scene timing, overlaps, and final arrangement
- a Docker-first local runtime with GPU-first behavior and CPU fallback

If you want the upstream project, papers, and hosted demos, use:

- Official code: [index-tts/index-tts](https://github.com/index-tts/index-tts)
- Official model: [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2)
- Official demo: [IndexTTS-2 Demo](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo)
- Paper: [arXiv 2506.21619](https://arxiv.org/abs/2506.21619)

## Why People Use It

This app is built for longer-form, practical TTS work rather than one-off single-line demos.

Example use cases:

- build a 3-speaker podcast or panel conversation
- create character dialogue for games, machinima, or visual novels
- generate narration with multiple takes, then compare and regenerate weak lines
- clean source clips before cloning so the voices sound more stable
- arrange interruptions, overlaps, and scene timing in a timeline before export

## Main Workspaces

### 1. Speaker Prep

- upload or select raw source clips
- trim, convert to mono, normalize, and clean noisy audio
- run quick clone-readiness diagnostics
- save the improved result into the live speaker library

### 2. Conversation Workflow

- paste a multi-speaker script
- see available voices clearly
- apply pacing presets
- parse, generate, and save project state

### 3. Conversation Results

- compare versions line by line
- play clips, compare clips, and review scores
- edit text during review
- regenerate weak lines
- export only after every line has a chosen final take

### 4. Timeline Editor

- build a scene directly in the timeline
- add speaker tracks and segments
- move segments in time
- shape overlaps and interruptions
- preview and export the final arranged scene

## Screenshots

| Speaker Prep | Conversation Workflow |
| --- | --- |
| ![Speaker Prep](docs/assets/manual/speaker-prep-tab.png) | ![Conversation Workflow](docs/assets/manual/conversation-workflow-tab.png) |

| Conversation Results | Timeline Editor |
| --- | --- |
| ![Conversation Results](docs/assets/manual/conversation-results-tab.png) | ![Timeline Editor](docs/assets/manual/timeline-editor-tab.png) |

## Quick Start

### Recommended: Docker

This is the supported runtime path for this repo.

Default behavior:

- use your NVIDIA GPU when Docker can access it
- fall back to CPU only when GPU runtime is unavailable

1. Put your model files in:

```text
shared/models/checkpoints
```

2. Start the app:

```powershell
docker\start.bat
```

Or manually:

```powershell
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d --build
```

3. Open:

- Frontend UI: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://localhost:8001](http://localhost:8001)
- API docs: [http://localhost:8001/docs](http://localhost:8001/docs)

4. Add your own clips:

- finished cloning prompts -> `shared/audio/speakers/`
- raw clips for prep -> `shared/audio/source_clips/`

5. Work through the app in this order:

- `Speaker Prep`
- `Conversation Workflow`
- `Conversation Results`
- `Timeline Editor`

To stop the stack:

```powershell
docker\stop.bat
```

If `shared/models/checkpoints` is empty, the backend can automatically download the official IndexTTS2 model bundle on first start.

## Hardware And Runtime Guidance

Practical guidance for a good local experience:

- NVIDIA GPU is strongly recommended
- CPU fallback works, but startup and generation are much slower
- at least `16 GB RAM`, `32 GB` recommended
- allow `50 GB+` disk space for models, caches, and outputs
- the first DeepSpeed-enabled startup can take longer while extensions warm up or compile

Today the Docker image is NVIDIA/CUDA-based. AMD and Apple GPU paths are not supported by this Docker image.

More runtime details: [docker/README.md](docker/README.md)

## Bring Your Own Voices

This app is intentionally **BYO voice**.

It does not include celebrity voices, personal voice libraries, or redistributable speaker packs.

Expected folders:

- `shared/audio/speakers/` - live speaker prompt files used by the app
- `shared/audio/source_clips/` - raw clips for cleanup and preparation
- `shared/audio/speakers_backups/` - backups of original speaker files before replacement

## Quality Tips

If a voice sounds too fast, robotic, or less faithful than expected:

- use the `Clone Fidelity` preset in the UI
- keep random sampling off
- use natural punctuation and sentence casing in the script
- prefer a clean `8 to 20 second` clip with one speaker and low background noise
- use `Speaker Prep` before blaming the model

There is also:

- a benchmark helper in [backend/scripts/quality_benchmark.py](backend/scripts/quality_benchmark.py)
- a listening review format in [docs/research/LISTENING_FEEDBACK_SYNTAX.md](docs/research/LISTENING_FEEDBACK_SYNTAX.md)
- a scripting guide in [docs/research/INDEXTTS2_SCRIPTING_PLAYBOOK.md](docs/research/INDEXTTS2_SCRIPTING_PLAYBOOK.md)

## Demo Audio

Add your showcase audio links here when you are ready. A good public demo set would include:

- a short multi-speaker roundtable sample
- a cleaner single-speaker narration sample
- a review/regeneration before-vs-after sample
- a timeline overlap or interruption sample

Example structure:

- `Podcast Roundtable Demo` - replace with your link
- `Narration Demo` - replace with your link
- `Regeneration Comparison Demo` - replace with your link
- `Timeline Interruption Demo` - replace with your link

## User Manual And Walkthrough Videos

If you want a guided tour of the app before using it, start here:

- Full user manual with screenshots: [docs/manual/USER_MANUAL.md](docs/manual/USER_MANUAL.md)
- Speaker Prep video: [docs/assets/manual/videos/speaker-prep-tab.webm](docs/assets/manual/videos/speaker-prep-tab.webm)
- Conversation Workflow video: [docs/assets/manual/videos/conversation-workflow-tab.webm](docs/assets/manual/videos/conversation-workflow-tab.webm)
- Conversation Results video: [docs/assets/manual/videos/conversation-results-tab.webm](docs/assets/manual/videos/conversation-results-tab.webm)
- Timeline Editor video: [docs/assets/manual/videos/timeline-editor-tab.webm](docs/assets/manual/videos/timeline-editor-tab.webm)

## Example Script

```text
SpeakerOne: I think we should test three versions before we keep the final line.
SpeakerTwo: Good. If one sounds rushed, regenerate it and compare again.
SpeakerThree: After that, move the best takes into the timeline and export the scene.
```

## Important Directories

- `shared/audio/speakers/` - live speaker reference audio used for cloning
- `shared/audio/source_clips/` - raw clips for preparation or batch processing
- `shared/audio/speakers_backups/` - backups of original speaker files
- `shared/models/checkpoints/` - IndexTTS model files
- `shared/audio/outputs/` - exported outputs
- `shared/audio/temp_conversation_segments/` - per-line conversation audio
- `shared/audio/uploads/` - temporary imported files
- `shared/data/project_saves/` - saved conversation projects
- `shared/data/timeline_projects/` - saved timeline projects
- `frontend/` - browser UI
- `backend/` - FastAPI app plus wrapped IndexTTS runtime
- `docs/` - manuals, research notes, release docs, and supporting references
- `tools/` - maintenance helpers, manual capture scripts, and debug utilities
- `examples/` - reusable sample inputs and saved examples

## Container CLI Usage

If you want a CLI-style run without teaching users a host Python setup, use the backend container:

```powershell
docker compose -f docker/docker-compose.yml exec backend python backend/indextts/cli.py "Your text here" -v /app/shared/audio/speakers/YourVoice.wav -o output.wav --model_dir /app/shared/models/checkpoints -c /app/shared/models/checkpoints/config.yaml
```

## Docs

- Docs index: [docs/README.md](docs/README.md)
- User manual: [docs/manual/USER_MANUAL.md](docs/manual/USER_MANUAL.md)
- Docker guide: [docker/README.md](docker/README.md)
- Deployment guide: [docs/deployment/DEPLOYMENT_GUIDE.md](docs/deployment/DEPLOYMENT_GUIDE.md)
- API summary: [docs/api/API_README.md](docs/api/API_README.md)
- Known limitations: [docs/project/KNOWN_LIMITATIONS.md](docs/project/KNOWN_LIMITATIONS.md)
- Release readiness: [docs/project/RELEASE_READINESS_STATUS.md](docs/project/RELEASE_READINESS_STATUS.md)
- Audio folder guide: [shared/audio/README.md](shared/audio/README.md)

## Credit

The underlying model technology, papers, and official pretrained checkpoints belong to the IndexTTS team. This repository packages those models into a more workflow-focused local application.

## Acknowledgements

- [index-tts/index-tts](https://github.com/index-tts/index-tts)
- [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
- [XTTSv2](https://github.com/coqui-ai/TTS)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [maskgct](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
