# Audiobook Night Train Demo Pack

Reusable audiobook scene written with generic placeholder voice labels for public demos.

Replace the placeholder labels in this pack, such as `SpeakerThree`, `SpeakerSix`, and `SpeakerEight`, with the exact voice labels from your own `Available Voices` list before you generate it.

This pack is tuned for a quick dramatic excerpt:

- one narrator voice anchoring the scene
- measured pauses for atmosphere
- a whisper moment that benefits from timeline control

## Voices Used

- `SpeakerThree`
- `SpeakerSix`
- `SpeakerEight`

## Suggested Settings

- Preset: `Clone Fidelity`
- Random sampling: `Off`
- Auto-detect emotions: `Off` if you want to follow the manual plan below
- Similarity threshold: `0.70`
- Auto-regen attempts: `2`

## Pasteable Script

```text
SpeakerThree: The night train left the coast at nineteen minutes past midnight, carrying rain on every window and silence in every seat.

SpeakerThree: Across from me, a woman in a dark blue coat kept one gloved hand on a tin box no larger than a prayer book.

SpeakerEight: You are awake early, she said, though the hour had already crossed into something stranger than early.

SpeakerThree: I told her sleep had stepped off somewhere before the last station and had not bothered to return.

SpeakerEight: People only travel with that face when they are running toward something, she said, or away from it.

SpeakerSix: Tickets, please.

SpeakerThree: The conductor lifted his lantern, and for a second the compartment shone gold enough to make the woman look almost unreal.

SpeakerEight: Do not tell him my name, she whispered.

SpeakerThree: The train leaned hard into a curve, the box rattled once against the window, and every passenger in the carriage looked up at the same time.

SpeakerSix: Final stop before dawn.

SpeakerThree: That was when I understood the box was not meant to survive the morning.
```

## Line IDs

- `L01` SpeakerThree
- `L02` SpeakerThree
- `L03` SpeakerEight
- `L04` SpeakerThree
- `L05` SpeakerEight
- `L06` SpeakerSix
- `L07` SpeakerThree
- `L08` SpeakerEight
- `L09` SpeakerThree
- `L10` SpeakerSix
- `L11` SpeakerThree

## Emotion And Timing Plan

Use this as a companion planning block for manual setup or future automation.

```yaml
scene:
  title: "Audiobook Night Train Demo"
  overlap_policy: explicit_only
  default_gap_ms: 260
  default_duck_db: -5
  max_auto_overlap_ms: 160

lines:
  - id: L01
    speaker: SpeakerThree
    emotion_text: "calm narration, nocturnal atmosphere"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false

  - id: L02
    speaker: SpeakerThree
    emotion_text: "measured, observant"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false

  - id: L03
    speaker: SpeakerEight
    emotion_text: "soft, curious"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L04
    speaker: SpeakerThree
    emotion_text: "wry, restrained"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 250
    allow_overlap: false

  - id: L05
    speaker: SpeakerEight
    emotion_text: "knowing, almost kind"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 280
    allow_overlap: false

  - id: L06
    speaker: SpeakerSix
    emotion_text: "formal interruption"
    emotion_weight: 0.95
    start_mode: overlap_previous
    overlap_prev_ms: 70
    duck_prev_db: -4
    fade_in_ms: 15
    allow_overlap: true

  - id: L07
    speaker: SpeakerThree
    emotion_text: "quietly unsettled"
    emotion_weight: 0.92
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L08
    speaker: SpeakerEight
    emotion_text: "urgent whisper"
    emotion_weight: 1.1
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false

  - id: L09
    speaker: SpeakerThree
    emotion_text: "cinematic tension"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false

  - id: L10
    speaker: SpeakerSix
    emotion_text: "warning disguised as routine"
    emotion_weight: 0.98
    start_mode: after_previous
    gap_after_ms: 300
    allow_overlap: false

  - id: L11
    speaker: SpeakerThree
    emotion_text: "quiet revelation"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 420
    allow_overlap: false
    notes: "Give the final line room to breathe before the clip ends."
```

## Quick Reuse Notes

- This is a good demo for narration plus dialogue in the same scene.
- Keep pauses slightly longer than the podcast pack so the atmosphere survives export.
- If the whisper line feels too forward, lower that segment a little in the timeline and preview again.
