# Timeline Editor Smoke

## Purpose

Verify standalone timeline creation, track creation, segment creation, waveform/details panel behavior, preview/export, and timeline save/load.

## Preconditions

- stack startup smoke has passed
- at least one local speaker is available
- recommended script pack for imports: [../../../test_scripts/flower_crisis_parody_timeline_pack.md](../../../test_scripts/flower_crisis_parody_timeline_pack.md)

## Steps

1. Open the `Timeline Editor` tab.
2. Create a blank timeline project.
3. Add a speaker track directly from the timeline editor.
4. Add a new segment from the track header or empty lane.
5. Enter text, assign the track, set start time and duration, then create the segment.
6. Select the segment and confirm the details panel updates.
7. Generate audio for the selected segment.
8. Confirm the waveform/details area updates after audio exists.
9. Drag the segment horizontally and confirm the timing changes.
10. If available, split the segment once and confirm both resulting pieces appear.
11. Preview the timeline mix.
12. Export or download the timeline mix.
13. Save the timeline project, reload it, and confirm the track and segment state persist.

## Expected Results

- blank timeline creation works without requiring conversation import
- new tracks and segments can be created from inside the editor
- selected segment details reflect the active clip
- generated segment audio is playable
- drag timing updates the segment position
- preview/export succeeds
- save/load restores the timeline project

## Common Failure Signs

- ruler and lanes drift out of alignment
- adding a segment only works from an external form and not the editor itself
- segment timing updates visually but does not persist
- preview/export fails after successful segment generation
- saved timeline reloads with the wrong counts or empty tracks

## Optional Checks

- test mute/solo controls on tracks
- test pop-out timeline editor if you are validating focused editing mode
