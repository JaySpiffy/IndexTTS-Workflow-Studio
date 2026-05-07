# Draft to Take Beta Manuals

These manuals are for the public Docker beta launcher in this repository.

This repo does not contain the private Draft to Take source code or model weights. It starts the public beta containers, stores your data locally, and downloads supported models into your own machine.

## Start Here

- [User Manual](USER_MANUAL.md) - the main app walkthrough, from first launch to exported mix.
- [Script Canvas Authoring Guide](SCRIPT_CANVAS_AUTHORING_GUIDE.md) - the exact script, emotion, SFX, ambience, and music marker format the canvas understands.
- [IndexTTS2 Prompting Guide](INDEXTTS2_PROMPTING_GUIDE.md) - how to write clean lines for better TTS output.
- [SFX, Ambience, And Music Smoke Test](SFX_AMBIENCE_MUSIC_SMOKE_TEST.md) - a tester checklist for optional sound-design generation.

## Try It Quickly

- [Blackmere Road sample scene](../samples/try-this-first/blackmere-road-45s.md) - import this Markdown file into Script Canvas to test dialogue, emotion comments, ambience, music, and SFX markers.
- [Audio examples](../media/audio-examples/README.md) - short generated MP3 examples visitors can hear before running the app.

## Useful Repo Files

- [README](../README.md) - install, start, beta status, model list, and troubleshooting.
- [BETA_TERMS](../BETA_TERMS.md) - beta terms and safety notes.
- [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) - upstream model and dependency notices.

## Where Your Work Is Stored

The launcher keeps projects, models, voices, and generated audio outside this release folder:

```text
%USERPROFILE%\DraftToTake\shared
```

That means you can update or replace this beta launcher without losing your local projects or downloaded models.
