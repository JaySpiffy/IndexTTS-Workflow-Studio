# Podcast Roundtable Demo Pack

Reusable roundtable scene written with generic placeholder voice labels for public demos.

Replace the placeholder labels in this pack, such as `SpeakerOne`, `SpeakerTwo`, and `SpeakerFour`, with the exact voice labels from your own `Available Voices` list before you generate it.

This pack is tuned for a short social preview:

- quick host-to-guest handoffs
- light overlap on interruptions
- clean enough to audition in one pass

## Voices Used

- `SpeakerOne`
- `SpeakerTwo`
- `SpeakerFour`

## Suggested Settings

- Preset: `Clone Fidelity`
- Random sampling: `Off`
- Auto-detect emotions: `Off` if you want to follow the manual plan below
- Similarity threshold: `0.68`
- Auto-regen attempts: `2`

## Pasteable Script

```text
SpeakerOne: Welcome back to the roundtable.
SpeakerOne: Today we are testing whether local multi-speaker TTS can actually feel directed instead of merely generated.

SpeakerFour: Which is a risky promise.
SpeakerFour: People forgive weird opinions faster than they forgive weird breathing.

SpeakerTwo: Good.
SpeakerTwo: I brought both opinions and measurable breathing.

SpeakerOne: The part I like is the review loop.
SpeakerOne: We generate a few takes, keep the line that works, and only regenerate the one that misses.

SpeakerFour: That changes the whole mood.
SpeakerFour: You stop babysitting one giant render and start shaping a conversation.

SpeakerTwo: The timeline editor is where it finally feels like production.
SpeakerTwo: Tiny pauses, quick overlaps, cleaner handoffs, and less accidental chaos.

SpeakerOne: So this demo is short on purpose.
SpeakerOne: We wanted something you can hear in under a minute and immediately understand the workflow.

SpeakerFour: Also, under a minute is the natural habitat of my attention span.

SpeakerTwo: Great.
SpeakerTwo: We have finally optimized the product around honesty.
```

## Line IDs

- `L01` SpeakerOne
- `L02` SpeakerOne
- `L03` SpeakerFour
- `L04` SpeakerFour
- `L05` SpeakerTwo
- `L06` SpeakerTwo
- `L07` SpeakerOne
- `L08` SpeakerOne
- `L09` SpeakerFour
- `L10` SpeakerFour
- `L11` SpeakerTwo
- `L12` SpeakerTwo
- `L13` SpeakerOne
- `L14` SpeakerOne
- `L15` SpeakerFour
- `L16` SpeakerTwo
- `L17` SpeakerTwo

## Emotion And Timing Plan

Use this as a companion planning block for manual setup or future automation.

```yaml
scene:
  title: "Podcast Roundtable Demo"
  overlap_policy: explicit_only
  default_gap_ms: 170
  default_duck_db: -4
  max_auto_overlap_ms: 180

lines:
  - id: L01
    speaker: SpeakerOne
    emotion_text: "warm host energy"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 110
    allow_overlap: false

  - id: L02
    speaker: SpeakerOne
    emotion_text: "clear, confident setup"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L03
    speaker: SpeakerFour
    emotion_text: "playful skepticism"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 90
    allow_overlap: false

  - id: L04
    speaker: SpeakerFour
    emotion_text: "dry punchline"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L05
    speaker: SpeakerTwo
    emotion_text: "quietly amused"
    emotion_weight: 0.9
    start_mode: overlap_previous
    overlap_prev_ms: 110
    duck_prev_db: -3
    fade_in_ms: 20
    allow_overlap: true

  - id: L06
    speaker: SpeakerTwo
    emotion_text: "deadpan confidence"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L07
    speaker: SpeakerOne
    emotion_text: "thoughtful, upbeat"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 90
    allow_overlap: false

  - id: L08
    speaker: SpeakerOne
    emotion_text: "explaining the workflow"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 210
    allow_overlap: false

  - id: L09
    speaker: SpeakerFour
    emotion_text: "agreeing, energized"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 90
    allow_overlap: false

  - id: L10
    speaker: SpeakerFour
    emotion_text: "animated explanation"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 210
    allow_overlap: false

  - id: L11
    speaker: SpeakerTwo
    emotion_text: "precise, approving"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 90
    allow_overlap: false

  - id: L12
    speaker: SpeakerTwo
    emotion_text: "matter-of-fact"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L13
    speaker: SpeakerOne
    emotion_text: "friendly wrap-up"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 80
    allow_overlap: false

  - id: L14
    speaker: SpeakerOne
    emotion_text: "inviting, clear"
    emotion_weight: 0.95
    start_mode: after_previous
    gap_after_ms: 230
    allow_overlap: false

  - id: L15
    speaker: SpeakerFour
    emotion_text: "self-aware joke"
    emotion_weight: 1.0
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L16
    speaker: SpeakerTwo
    emotion_text: "tiny laugh before the line"
    emotion_weight: 0.9
    start_mode: overlap_previous
    overlap_prev_ms: 120
    duck_prev_db: -3
    fade_in_ms: 20
    allow_overlap: true

  - id: L17
    speaker: SpeakerTwo
    emotion_text: "dry closer"
    emotion_weight: 0.9
    start_mode: after_previous
    gap_after_ms: 280
    allow_overlap: false
    notes: "Leave a beat after the last line for the outro."
```

## Quick Reuse Notes

- Paste the `Pasteable Script` block into the main conversation workflow.
- Keep `Use Random Sampling` off if you want stronger voice fidelity.
- The overlap lines are intentional; keep them subtle in the timeline.
- This pack is short enough to use as a social audio teaser without trimming it again.
