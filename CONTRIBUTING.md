# Contributing

Thanks for contributing to IndexTTS2 Workflow Studio.

This project is a Docker-first local app built on top of the official IndexTTS2 models. The best contributions here are focused, practical improvements that keep the frontend, backend, Docker runtime, and docs aligned.

## Before You Open A PR

- Check existing issues first.
- Use the issue templates for bugs, setup help, and feature requests.
- Keep changes scoped. Smaller pull requests are much easier to review and verify.

## Development Approach

Prefer:

- the Docker-first runtime
- backend tests run inside the backend container
- docs updates when behavior or setup changes
- focused fixes over broad rewrites

Avoid:

- reintroducing host-managed Python setup as the default path
- bundling voice libraries, speaker files, or model weights in the repo
- mixing unrelated cleanup and feature work in one PR

## Local Workflow

Start the stack:

```powershell
docker\start.bat
```

Or manually:

```powershell
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d --build
```

Useful commands:

```powershell
# Backend health
curl http://localhost:8001/health

# Backend logs
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml logs -f backend

# Focused backend test
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml exec backend python3.10 tests/backend/test_line_emotion_contract.py
```

## Testing Expectations

When changing generation, review, export, speaker prep, or timeline behavior:

- update or add focused backend tests
- run the relevant backend tests in Docker
- run at least the matching manual smoke pack from [docs/testing](docs/testing/README.md)

Useful test docs:

- [Smoke tests](docs/testing/README.md)
- [Release checklist](docs/testing/release/full-release-checklist.md)
- [Test run log template](docs/testing/test-run-log-template.md)

## Pull Request Guidelines

Please include:

- what changed
- why it changed
- how you tested it
- screenshots or short notes for UI changes

If your change touches UI behavior, mention whether a hard refresh is needed because of a frontend bundle/version bump.

## Repo Boundaries

Do not commit:

- local voice files
- raw speaker/source clip libraries
- local model weights or Hugging Face cache data
- generated outputs, temp audio, or personal backups

The repo is intentionally BYO-voices and Docker-first.

## Good Contribution Areas

- frontend/backend contract fixes
- Docker/runtime reliability
- docs and onboarding improvements
- timeline editor polish
- speaker prep improvements
- focused tests and verification workflows

