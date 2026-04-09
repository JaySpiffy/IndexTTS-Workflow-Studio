# V1 To V2 Parity Checklist

This checklist tracks whether v2 is truly ready to replace the older public v1 workflow.

Status key:

- `[x]` implemented and verified
- `[~]` partial, improved, or good enough for now but worth polishing
- `[ ]` still missing before replacing v1

## Core Workflow

- `[x]` Multi-speaker conversation generation
- `[x]` Review screen with per-line version comparison
- `[x]` Manual per-line version selection
- `[x]` Review-time text editing before regeneration
- `[x]` Threshold-based regeneration for weak lines
- `[x]` Project save/load for scripts, settings, results, and selections
- `[x]` Final export gating so every line must have an intentional chosen version
- `[x]` Progress reporting that moves during real generation work

## Voice And Quality Controls

- `[x]` Available voices panel in the main workflow
- `[x]` Speaker prep surfaced in the main UI
- `[x]` Trim, mono conversion, normalization, and noise cleanup for source clips
- `[x]` Clone-readiness diagnostics and recommended prep recipe
- `[x]` Batch speaker cleanup with backup preservation
- `[x]` Seed controls and seed persistence in the workflow
- `[x]` Seed metadata visible in results
- `[~]` Seed export/reporting is present, but import/replay as a first-class workflow is still lighter than ideal

## Emotion And Dialogue Control

- `[x]` Auto emotion detection from text
- `[x]` Per-line emotion wiring carried through generation
- `[x]` Editable emotion state in the review workflow
- `[x]` Script packs and reusable markdown test scripts
- `[x]` Dialogue pacing presets: `Natural`, `Calm`, `Argument`, `Panic`
- `[x]` Per-speaker delivery shaping
- `[x]` Scene-level gap and pause shaping

## Export And Finishing

- `[x]` Conversation export with intentional selected versions
- `[x]` Timeline export
- `[x]` Output format support for WAV, MP3, and OGG
- `[x]` MP3 bitrate control
- `[x]` Volume matching / line leveling on export
- `[x]` Peak protection on final mix
- `[x]` Silence trimming controls
- `[x]` Overlap-aware export path for timeline and planned interruptions
- `[~]` Rich mastering-style effects from v1 are still thinner here

## Timeline Editor

- `[x]` Real timeline editor surfaced in the main app
- `[x]` Drag segments horizontally to retime them
- `[x]` Pop-out editor window
- `[x]` Waveform preview for the selected segment
- `[x]` Segment splitting
- `[x]` Track-level mute / solo / level controls
- `[x]` Overlap ducking on export
- `[~]` Full DAW-style inline waveforms and trim handles are still not as deep as a dedicated editor

## Runtime And Deployment

- `[x]` Docker-first runtime
- `[x]` GPU-first startup with CPU fallback
- `[x]` Device reporting in health/UI
- `[x]` Automatic model download into `shared/models/checkpoints`
- `[x]` DeepSpeed working in the Docker GPU path
- `[x]` Shared runtime storage layout cleaned up and documented

## Release Notes

### No Longer Major Blockers

These items used to be real gaps and are now in place:

- project save/load
- selection gating
- speaker prep in the main UI
- seed workflow
- richer export controls
- timeline editor as a real first-class feature
- pacing controls

### Still Worth Polishing Before Replacing V1

- `[~]` Advanced FX-style finishing chain is still lighter than the old app
  - pitch-shift / tone-shaping
  - compression / EQ / reverb style controls
  - dedicated advanced preview flow for those effects
- `[~]` Speaker library UX should stay tidy and release-clean
- `[~]` Timeline UX can still be polished further for heavy editing sessions

### Current Recommendation

v2 is no longer in broad feature catch-up. It is in final release-hardening and polish.

The remaining work is mostly:

1. final release smoke and doc cleanup
2. any last speaker-library tidying
3. optional advanced finishing polish if you want deeper v1-style audio effects before launch
