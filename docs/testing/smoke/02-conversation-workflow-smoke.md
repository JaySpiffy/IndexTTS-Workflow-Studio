# Conversation Workflow Smoke

## Purpose

Verify script loading, parsing, generation settings, progress tracking, pacing presets, and queue behavior from the `Conversation Workflow` tab.

## Preconditions

- stack startup smoke has passed
- at least two local speaker WAV files exist in `shared/audio/speakers`
- recommended script pack: [../../../test_scripts/dark_garden_parody.md](../../../test_scripts/dark_garden_parody.md)

## Steps

1. Open the `Conversation Workflow` tab.
2. Confirm the available voices list loads and shows your local voices.
3. Load a script pack or paste a short two-speaker script manually.
4. Click `Parse Script`.
5. Confirm the preview renders the expected lines and speaker labels.
6. Set a pacing preset such as `Natural` or `Calm`.
7. Adjust one visible generation setting, for example `Versions per Line`.
8. Start generation.
9. Watch the progress panel.
10. Confirm the progress text changes over time instead of staying stuck at `Initializing...`.
11. If a second browser tab or window is available, start another generation while the first one is still busy and confirm the second job shows a queued state.
12. Wait for generation to complete.

## Expected Results

- voices load successfully
- script parsing succeeds with no malformed-line crash
- generation starts and the progress bar moves
- pacing preset changes stay selected
- queued jobs show queue wording instead of silently failing
- completed generation exposes a valid conversation ID and results path

## Common Failure Signs

- parse succeeds but lines disappear
- progress remains at `0%` while backend is actively working
- second generation request crashes instead of queueing
- queued task never leaves the queue after the first task completes
- generation completes but no results are available

## Optional Queue Check

Use two small conversations back to back:
- first request should move to `running`
- second request should show queued wording
- second request should start automatically when the first finishes
