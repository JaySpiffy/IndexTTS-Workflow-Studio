# Conversation Results Smoke

## Purpose

Verify version cards, pacing-aware review scores, selection gating, playback, regeneration, and final concatenation from the `Conversation Results` tab.

## Preconditions

- a conversation has already been generated from the workflow tab
- recommended test conversation uses at least two lines and multiple versions per line

## Steps

1. Open the `Conversation Results` tab.
2. Load the newest generated conversation if it is not already open.
3. Confirm each line shows multiple version cards.
4. Check that version cards show:
   - `Review`
   - `Quality`
   - `Similarity`
   - `Pacing` when available
5. Play at least one version.
6. Use `Auto-Select Best`.
7. Confirm each line changes to a valid selected state.
8. Clear selections once and confirm export becomes blocked.
9. Re-select versions manually or with `Auto-Select Best`.
10. Regenerate one line or one version if that flow is available.
11. Confirm the new version appears and the page remains usable.
12. Concatenate or download the selected result.

## Expected Results

- result cards render cleanly
- pacing-aware review data appears on versions
- auto-select uses the best review candidate instead of selecting randomly
- export is blocked when any line has no selected version
- playback works
- regeneration adds or replaces versions without breaking the page
- final concatenate succeeds after valid selections exist

## Common Failure Signs

- results page loads but version cards are missing
- scores show as blank or `NaN`
- auto-select picks obviously incomplete or unscored versions
- export works even when a line has no chosen version
- regeneration completes in the backend but the UI never refreshes

## Suggested Evidence

- screenshot of a line with visible `Review` and `Pacing` badges
- screenshot of the selection-gating warning when no final version is chosen
