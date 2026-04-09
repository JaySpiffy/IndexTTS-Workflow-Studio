<div align="center">
<img src="assets/index_icon.png" width="220" alt="IndexTTS2 app icon"/>
</div>

# IndexTTS2 App Toolkit

This repository is **not** the official IndexTTS model repository.

It is a practical web app and workflow tool built **on top of** the official IndexTTS models, with a Dockerized frontend/backend setup, conversation generation, project save/load, review and regeneration tools, and speaker-preparation utilities.

## What This Repo Is

- A FastAPI + HTML frontend app for working with IndexTTS2 models
- A conversation workflow tool for multi-speaker generation
- A review UI for version comparison, regeneration, and export
- A Docker-first local setup with GPU-aware runtime options
- A workspace for speaker prep, source clip processing, and project saves

## What This Repo Is Not

- Not the upstream research/code release from the IndexTTS team
- Not the canonical place for model papers, demos, or base inference docs
- Not a drop-in replacement for every upstream Gradio/webui workflow

## Upstream IndexTTS Links

This tool uses the official IndexTTS ecosystem. For the original project, papers, and hosted demos, use:

- Official code: [index-tts/index-tts](https://github.com/index-tts/index-tts)
- IndexTTS2 model: [Hugging Face](https://huggingface.co/IndexTeam/IndexTTS-2)
- IndexTTS2 demo: [Hugging Face Space](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo)
- IndexTTS2 paper: [arXiv 2506.21619](https://arxiv.org/abs/2506.21619)

## What This App Adds

- Docker stack with named frontend/backend services
- Browser-based conversation workflow
- Save/load project state
- Review-time text editing and manual regeneration
- Similarity scoring and selection workflow
- Speaker/source-clip prep endpoints
- Better runtime device selection for local deployment

## Quick Start

### Option 1: Docker (Recommended)

This is the easiest and most repeatable way to run the app.

The default behavior is:

- use your NVIDIA GPU when Docker can access it
- fall back to CPU only when GPU runtime is unavailable

1. Put your model files in:

```text
shared/models/checkpoints
```

2. Start the stack:

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

If `shared/models/checkpoints` is empty, the backend will automatically download the official IndexTTS2 model bundle into that folder on first start.

To stop it:

```powershell
docker\stop.bat
```

More Docker details are in [docker/README.md](docker/README.md).

## User Guide And Walkthroughs

If you want a guided tour of the app before using it, start here:

- Full user manual with screenshots: [docs/manual/USER_MANUAL.md](docs/manual/USER_MANUAL.md)
- Speaker Prep video: [docs/assets/manual/videos/speaker-prep-tab.webm](docs/assets/manual/videos/speaker-prep-tab.webm)
- Conversation Workflow video: [docs/assets/manual/videos/conversation-workflow-tab.webm](docs/assets/manual/videos/conversation-workflow-tab.webm)
- Conversation Results video: [docs/assets/manual/videos/conversation-results-tab.webm](docs/assets/manual/videos/conversation-results-tab.webm)
- Timeline Editor video: [docs/assets/manual/videos/timeline-editor-tab.webm](docs/assets/manual/videos/timeline-editor-tab.webm)

The manual and videos cover the four main tabs in the app:

1. `Speaker Prep`
2. `Conversation Workflow`
3. `Conversation Results`
4. `Timeline Editor`

## Model Setup

The Docker stack can download the official IndexTTS2 model automatically on first backend start.

If you prefer to pre-seed it yourself, place the model files in:

```text
shared/models/checkpoints
```

If you keep models somewhere else, set:

```powershell
$env:INDTEXTS_MODEL_PATH="C:\path\to\your\checkpoints"
```

## Main Workflow

This app does **not** ship with bundled voice clones. Users are expected to bring their own legal reference audio.

1. Put finished reference voices in `shared/audio/speakers/`
2. Put raw clips you want to prep in `shared/audio/source_clips/`
3. Open the app
4. Paste a multi-speaker script like:

```text
SpeakerOne: Hello there.
SpeakerTwo: Hi, how are you?
```

4. Parse the script
5. Generate versions
6. Review line versions
7. Regenerate weak lines if needed
8. Concatenate or download selected audio

## Important Directories

- `shared/audio/speakers/`: live speaker reference audio files the app uses for cloning, supplied locally by the user
- `shared/audio/speakers_backups/`: backups of original speaker files before prep/replacement
- `shared/audio/source_clips/`: raw clips for preparation or batch processing, supplied locally by the user
- `shared/models/checkpoints/`: official IndexTTS model files
- `shared/audio/outputs/`: generated outputs
- `shared/audio/temp_conversation_segments/`: per-line conversation audio
- `shared/audio/uploads/`: temporary imported files only
- `shared/data/project_saves/`: saved projects
- `shared/data/timeline_projects/`: timeline editor projects
- `frontend/`: browser UI
- `backend/`: FastAPI app plus wrapped IndexTTS runtime
- `docs/`: supporting docs, research notes, and planning files
- `tools/`: one-off debug helpers and manual checks
- `examples/`: reusable sample inputs and saved examples

## CLI Usage

If you want a CLI-style run without installing Python tooling on the host, use the backend container:

```powershell
docker compose -f docker/docker-compose.yml exec backend python backend/indextts/cli.py "Your text here" -v /app/shared/audio/speakers/YourVoice.wav -o output.wav --model_dir /app/shared/models/checkpoints -c /app/shared/models/checkpoints/config.yaml
```

## Runtime Notes

- Docker is the supported runtime path for this repo
- The default startup path is GPU-first with CPU fallback
- The backend can auto-download the official model bundle on first start
- `INDTEXTS_DEVICE=auto` is the default runtime mode
- `INDTEXTS_USE_FP16=auto` is supported
- Docker uses DeepSpeed by default on the GPU path
- The first DeepSpeed-enabled startup can take longer while extensions warm up or compile
- If DeepSpeed fails to initialize, the backend falls back to normal GPU inference automatically
- Random sampling can reduce voice-cloning fidelity
- The current Docker image is NVIDIA/CUDA-based
- Speaker and source audio are intentionally local-only and are not meant to be redistributed with the app

## Quality Tuning

If a voice sounds too fast, robotic, or less faithful than expected:

- use the `Clone Fidelity` preset in the UI
- keep random sampling off
- use normal punctuation and sentence casing in the script
- prefer a clean 10-20 second reference clip with one speaker and low background noise

There is also a benchmark helper in [backend/scripts/quality_benchmark.py](backend/scripts/quality_benchmark.py) and a human listening review format in [docs/research/LISTENING_FEEDBACK_SYNTAX.md](docs/research/LISTENING_FEEDBACK_SYNTAX.md) so we can compare what the metrics say with what you actually hear.

## App Features

- Multi-speaker conversation generation
- Available voices panel
- Auto emotion detection and editable line emotions
- Review-time text editing
- Manual regeneration by threshold
- Project save/load
- Audio playback and comparison
- Concatenation and export
- Speaker tool APIs for extraction, trimming, and batch prep

## Docs

- Docs index: [docs/README.md](docs/README.md)
- User manual with screenshots: [docs/manual/USER_MANUAL.md](docs/manual/USER_MANUAL.md)
- Manual videos folder: [docs/assets/manual/videos/](docs/assets/manual/videos/)
- App deployment notes: [docs/deployment/DEPLOYMENT_GUIDE.md](docs/deployment/DEPLOYMENT_GUIDE.md)
- Docker usage: [docker/README.md](docker/README.md)
- API summary: [docs/api/API_README.md](docs/api/API_README.md)
- V1 to V2 parity tracker: [docs/project/V1_TO_V2_PARITY_CHECKLIST.md](docs/project/V1_TO_V2_PARITY_CHECKLIST.md)
- Release hygiene checklist: [docs/project/RELEASE_HYGIENE_CHECKLIST.md](docs/project/RELEASE_HYGIENE_CHECKLIST.md)
- Known limitations: [docs/project/KNOWN_LIMITATIONS.md](docs/project/KNOWN_LIMITATIONS.md)
- Release readiness status: [docs/project/RELEASE_READINESS_STATUS.md](docs/project/RELEASE_READINESS_STATUS.md)
- Audio folder guide: [shared/audio/README.md](shared/audio/README.md)
- Voice fidelity research note: [docs/research/INDEXTTS2_VOICE_FIDELITY_RESEARCH_SYNTHESIS_NEXT_ACTIONS.md](docs/research/INDEXTTS2_VOICE_FIDELITY_RESEARCH_SYNTHESIS_NEXT_ACTIONS.md)

## Credit

The underlying model technology, papers, and official pretrained checkpoints belong to the IndexTTS team. This repository packages those models into a more deployment-focused local application workflow.

## Acknowledgements

- [index-tts/index-tts](https://github.com/index-tts/index-tts)
- [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
- [XTTSv2](https://github.com/coqui-ai/TTS)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [maskgct](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
