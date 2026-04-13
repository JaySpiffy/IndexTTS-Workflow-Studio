# Game Dialogue Checkpoint Breach Pack

Reusable in-game banter scene written with generic placeholder voice labels for public demos.

Replace the placeholder labels in this pack, such as `SpeakerFour`, `SpeakerSix`, `SpeakerSeven`, and `SpeakerEight`, with the exact voice labels from your own `Available Voices` list before you generate it.

This pack is tuned for a fast mission beat:

- tight command-and-response pacing
- one interruption moment
- enough motion to show why the timeline matters

## Voices Used

- `SpeakerFour`
- `SpeakerSix`
- `SpeakerSeven`
- `SpeakerEight`

## Suggested Settings

- Preset: `Clone Fidelity`
- Random sampling: `Off`
- Auto-detect emotions: `Off` if you want to follow the manual plan below
- Similarity threshold: `0.66`
- Auto-regen attempts: `2`

## Pasteable Script

```text
SpeakerFour: Control, we are at the checkpoint and the gate is welded shut.

SpeakerSeven: Give me four seconds with the panel.
SpeakerSeven: Maybe six if the turret notices me first.

SpeakerSix: You have three.
SpeakerSix: Patrol on the upper bridge just changed route.

SpeakerEight: I am reading two heat signatures behind the blast door.
SpeakerEight: They already know you are there.

SpeakerFour: Copy that.
SpeakerFour: Quiet entry is cancelled.
SpeakerFour: Stylish entry remains available.

SpeakerSeven: Panel is spoofed.
SpeakerSeven: Gate is opening.
SpeakerSeven: Nobody say anything heroic yet.

SpeakerSix: Too late.
SpeakerSix: Drone inbound from the east stairwell.

SpeakerEight: Tagging weak points now.
SpeakerEight: Left rotor first.

SpeakerFour: I see it.
SpeakerFour: Taking the shot.

SpeakerSeven: Shot landed.
SpeakerSeven: Door is halfway open and the alarm is fully offended.

SpeakerSix: Push through, grab the core, and meet me at extraction in ninety seconds.

SpeakerFour: On it.
SpeakerFour: If anyone asks, this was always the subtle plan.
```

## Line IDs

- `L01` SpeakerFour
- `L02` SpeakerSeven
- `L03` SpeakerSeven
- `L04` SpeakerSix
- `L05` SpeakerSix
- `L06` SpeakerEight
- `L07` SpeakerEight
- `L08` SpeakerFour
- `L09` SpeakerFour
- `L10` SpeakerFour
- `L11` SpeakerSeven
- `L12` SpeakerSeven
- `L13` SpeakerSeven
- `L14` SpeakerSix
- `L15` SpeakerSix
- `L16` SpeakerEight
- `L17` SpeakerEight
- `L18` SpeakerFour
- `L19` SpeakerFour
- `L20` SpeakerSeven
- `L21` SpeakerSeven
- `L22` SpeakerSix
- `L23` SpeakerFour
- `L24` SpeakerFour

## Emotion And Timing Plan

Use this as a companion planning block for manual setup or future automation.

```yaml
scene:
  title: "Game Dialogue Checkpoint Breach"
  overlap_policy: explicit_only
  default_gap_ms: 120
  default_duck_db: -4
  max_auto_overlap_ms: 220

lines:
  - id: L01
    speaker: SpeakerFour
    emotion_text: "radio report, under pressure"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 120
    allow_overlap: false

  - id: L02
    speaker: SpeakerSeven
    emotion_text: "focused hacker energy"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 80
    allow_overlap: false

  - id: L03
    speaker: SpeakerSeven
    emotion_text: "nervous joke"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 140
    allow_overlap: false

  - id: L04
    speaker: SpeakerSix
    emotion_text: "tight command"
    emotion_weight: 1.0
    start_mode: overlap_previous
    overlap_prev_ms: 100
    duck_prev_db: -4
    fade_in_ms: 15
    allow_overlap: true

  - id: L05
    speaker: SpeakerSix
    emotion_text: "urgent tactical update"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 120
    allow_overlap: false

  - id: L06
    speaker: SpeakerEight
    emotion_text: "cool, precise"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 80
    allow_overlap: false

  - id: L07
    speaker: SpeakerEight
    emotion_text: "warning"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 140
    allow_overlap: false

  - id: L08
    speaker: SpeakerFour
    emotion_text: "steady response"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 60
    allow_overlap: false

  - id: L09
    speaker: SpeakerFour
    emotion_text: "accepting the chaos"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 60
    allow_overlap: false

  - id: L10
    speaker: SpeakerFour
    emotion_text: "grinning under pressure"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 130
    allow_overlap: false

  - id: L11
    speaker: SpeakerSeven
    emotion_text: "locked in"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 60
    allow_overlap: false

  - id: L12
    speaker: SpeakerSeven
    emotion_text: "quick update"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 60
    allow_overlap: false

  - id: L13
    speaker: SpeakerSeven
    emotion_text: "trying to stay light"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 130
    allow_overlap: false

  - id: L14
    speaker: SpeakerSix
    emotion_text: "hard interruption"
    emotion_weight: 1.0
    start_mode: overlap_previous
    overlap_prev_ms: 110
    duck_prev_db: -4
    fade_in_ms: 15
    allow_overlap: true

  - id: L15
    speaker: SpeakerSix
    emotion_text: "incoming threat callout"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 120
    allow_overlap: false

  - id: L16
    speaker: SpeakerEight
    emotion_text: "calm targeting callout"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 60
    allow_overlap: false

  - id: L17
    speaker: SpeakerEight
    emotion_text: "precise cue"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 120
    allow_overlap: false

  - id: L18
    speaker: SpeakerFour
    emotion_text: "visual confirmation"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 60
    allow_overlap: false

  - id: L19
    speaker: SpeakerFour
    emotion_text: "action beat"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 140
    allow_overlap: false

  - id: L20
    speaker: SpeakerSeven
    emotion_text: "relieved surprise"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 60
    allow_overlap: false

  - id: L21
    speaker: SpeakerSeven
    emotion_text: "sarcastic systems report"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 140
    allow_overlap: false

  - id: L22
    speaker: SpeakerSix
    emotion_text: "mission command"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 180
    allow_overlap: false

  - id: L23
    speaker: SpeakerFour
    emotion_text: "moving fast"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 60
    allow_overlap: false

  - id: L24
    speaker: SpeakerFour
    emotion_text: "cocky closer"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 260
    allow_overlap: false
    notes: "Keep the last line punchy and leave a brief tail after it."
```

## Quick Reuse Notes

- This is the fastest of the three demo packs and the easiest one to showcase overlap.
- Use two or more tracks in the timeline if you want the interruptions to feel cleaner.
- If the command voice overwhelms the mix, pull that track down slightly and preview again.
