# Draft to Take Beta Notes

Draft to Take beta is early-access software for testing the local Docker workflow.

## What You Are Testing

- Local Docker startup on real Windows machines.
- First-run model downloads.
- Voice setup and Script Canvas workflow.
- Timeline generation, review, retry, and export.
- Optional local Qwen support for emotion detection and AI Thread workflows.
- Optional OmniVoice sidecar for reusable voice design.
- Optional SFX/music tools, if you deliberately enable them.

## Important Limits

- This beta is not a finished commercial release.
- It may break, take a long time to download models, or fail on some hardware.
- Do not use it for client-critical production work without checking results by ear.
- Do not post private scripts, voice samples, API keys, or personal audio in public issues.
- Generated voices and outputs remain your responsibility.
- Third-party models have their own licenses. Draft to Take does not grant extra rights to those models or their outputs.

## SFX And Music Notice

SFX/music generation is optional, experimental, and disabled by default in the beta launcher.

Some model-backed SFX/music engines may use non-commercial or research-only weights. Treat generated SFX/music as license-dependent unless you have checked the active model terms for your intended use.

## Feedback

Please use the public release repository Issues tab for:

- startup failures
- model download problems
- GPU/VRAM issues
- confusing UI steps
- broken exports
- bad error messages
- feature ideas that would make the workflow easier

When reporting a bug, include your Windows version, GPU, VRAM, Docker Desktop version, and whether you ran `collect-diagnostics.bat`.
