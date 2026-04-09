# Overlap Planning Template

This is the safest way to add interruptions and overlap without turning every scene into accidental crosstalk.

## Core Rule

Default to no overlap.

If a line does not explicitly request overlap, it should start after the previous line ends plus a normal gap.

That means the AI planner should follow:

- `after_previous` by default
- overlap only when the plan file says so
- small overlaps first, usually `120` to `260` ms
- bigger overlaps only for shouting, panic, or interruption scenes

## Script First, Timing Second

Write the script in the normal app format first:

```text
SpeakerName: spoken line here
```

Best script habits before planning overlap:

- one speaker per line
- short lines
- punctuation for pauses, not drama spam
- no overlap unless the scene needs interruption

## Recommended Structure

Use the script file for dialogue, and a separate companion plan file for timing.

Example:

```yaml
scene:
  title: "Dark Garden Parody"
  overlap_policy: explicit_only
  default_gap_ms: 220
  default_duck_db: -4
  max_auto_overlap_ms: 260

lines:
  - id: L01
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false
    notes: "Normal opening line."

  - id: L02
    start_mode: after_previous
    gap_after_ms: 160
    allow_overlap: false
    notes: "Dry response, no interruption."

  - id: L03
    start_mode: overlap_previous
    overlap_prev_ms: 160
    duck_prev_db: -5
    allow_overlap: true
    notes: "Joe cuts in because he is reacting in disbelief."

  - id: L04
    start_mode: after_previous
    gap_after_ms: 240
    allow_overlap: false
    notes: "Let this line land cleanly."
```

## Start Modes

- `after_previous`
  Start after the prior line ends.

- `overlap_previous`
  Start before the prior line fully ends.

- `simultaneous`
  Start at the same time as another line. Use rarely.

- `hard_cut_previous`
  Fade or cut the previous line early. Use very rarely.

## Useful Timing Fields

- `gap_after_ms`
  Silence after the previous line before this one starts.

- `overlap_prev_ms`
  How much this line starts before the prior line ends.

- `duck_prev_db`
  Lower the previous speaker during the overlap.

- `fade_in_ms`
  Small fade-in for the new speaker.

- `fade_out_prev_ms`
  Small fade-out for the interrupted speaker.

- `allow_overlap`
  Explicit true or false safety switch.

## AI Planning Rules

If we want an AI to generate this plan from a script, it should follow these rules:

1. Never invent overlap unless the scene clearly suggests interruption, argument, panic, or crosstalk.
2. Keep overlap off for monologues, punchlines, emotional reveals, and important exposition.
3. Use short overlap for reactions:
   - `120` to `180` ms
4. Use medium overlap for heated arguments:
   - `180` to `260` ms
5. Only use strong overlap above `260` ms when the scene explicitly calls for chaos.
6. If unsure, choose `after_previous`.
7. Add ducking when the interrupting speaker should not bulldoze the current lead voice.

## Best Workflow

The cleanest workflow is:

1. Write the script normally.
2. Keep the lines short enough for clean TTS pacing.
3. Add line IDs.
4. Upload a companion overlap plan.
5. Let the mixer or timeline editor apply timing from that plan only.

That keeps generation simple and makes overlap a controlled post-process instead of a risky hidden behavior.

## Tiny Example

```yaml
scene:
  title: "Argument Test"
  overlap_policy: explicit_only
  default_gap_ms: 220

lines:
  - id: L01
    start_mode: after_previous
    gap_after_ms: 220
    allow_overlap: false

  - id: L02
    start_mode: overlap_previous
    overlap_prev_ms: 160
    duck_prev_db: -4
    allow_overlap: true
    notes: "Interrupting rebuttal."

  - id: L03
    start_mode: after_previous
    gap_after_ms: 280
    allow_overlap: false
    notes: "Pause so the next line lands."
```

## Recommendation For This Repo

The safest repo standard is:

- line IDs from the script
- overlap timing from a companion YAML or Markdown block
- no overlap unless `allow_overlap: true`

That gives us arguments and interruptions without making ordinary scenes messy.
