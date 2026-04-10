# Speaker Prep Smoke

## Purpose

Verify that source clip intake, diagnostics, prep actions, and prepared-output playback all work from the `Speaker Prep` tab.

## Preconditions

- stack startup smoke has passed
- at least one local source clip exists in `shared/audio/source_clips`

## Steps

1. Open the `Speaker Prep` tab.
2. Confirm the source clip list loads.
3. Select the first available clip if one is not already auto-selected.
4. Run diagnostics for the clip if they are not shown automatically.
5. Review the reported trim/quality recommendations.
6. Click `Apply Recommended Prep` if available.
7. If a suggested trim action is shown, use it.
8. Run the prep action and wait for completion.
9. Play the prepared output.
10. If the UI offers it, load the prepared clip back into the active source or save it as a speaker file.

## Expected Results

- source clips appear in the picker
- diagnostics return a score and actionable notes
- prep completes without a server error
- prepared output gets a new file name
- audio playback opens and works
- any “load output as source” or save action succeeds

## Common Failure Signs

- source clip list is empty even though files exist
- diagnostics spinner never resolves
- prep reports success but no output file is created
- prepared output cannot be played
- UI shows a source clip but backend says file not found

## Suggested Evidence

- screenshot of the diagnostics panel
- output filename shown after prep
- any before/after score shown in the result area
