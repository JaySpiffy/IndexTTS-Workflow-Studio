# IndexTTS2 User Manual

This manual describes the current Docker-first local app as it exists in this repository.

The app is organized around four main tabs:

1. `Speaker Prep`
2. `Conversation Workflow`
3. `Conversation Results`
4. `Timeline Editor`

All screenshots in this guide were captured from the live local app on `http://localhost:3000`.
The public screenshots and videos use generic placeholder voice labels such as `SpeakerOne` and `SpeakerTwo` to avoid implying that the app ships with bundled voice clones.

## Short Video Walkthroughs

- [Speaker Prep Walkthrough](../assets/manual/videos/speaker-prep-tab.webm)
- [Conversation Workflow Walkthrough](../assets/manual/videos/conversation-workflow-tab.webm)
- [Conversation Results Walkthrough](../assets/manual/videos/conversation-results-tab.webm)
- [Timeline Editor Walkthrough](../assets/manual/videos/timeline-editor-tab.webm)

## Quick Start

1. Start the app with `docker\start.bat`.
2. Open `http://localhost:3000`.
3. Check the status badge in the header.
4. Add or prepare your own voices first.
5. Build a script in `Conversation Workflow`.
6. Review and choose versions in `Conversation Results`.
7. Move into `Timeline Editor` when you want hand-placed scene timing, interruptions, or a custom final arrangement.

## Header And Status Badge

- The top-left logo confirms you are in the main IndexTTS2 app.
- The header status badge shows whether the backend is connected and which runtime it is using.
- On a healthy NVIDIA setup, the badge should look like `API Connected - GPU: cuda:0 + DeepSpeed`.
- If Docker falls back to CPU, the badge will say so clearly.
- The theme toggle in the header switches light and dark mode.

## Speaker Prep

![Speaker Prep Screenshot](../assets/manual/speaker-prep-tab.png)

`Speaker Prep` is where you clean and evaluate source clips before you turn them into live speaker prompts.

### Source Clip Intake

- `Audio File` uploads a local clip into the app's source-clip library.
- `Optional Save Name` lets you give the uploaded clip a cleaner local filename.
- `Upload Source Clip` imports the clip into the source library.
- `Refresh Clips` reloads the source-clip list from disk.
- The status text under the buttons tells you what the tab expects from a good prompt clip.
- The best-practice badges summarize the recommended capture style:
  - roughly `8 to 20 seconds`
  - one clear speaker
  - low room noise
  - natural pacing and punctuation

### Source Clips

- The `Source Clips` panel lists the raw clips currently available for prep.
- Each clip card gives you a quick health signal before you even run full diagnostics.
- `Select` makes a clip the active working clip.
- `Play` previews the raw clip.
- `Diagnose` runs the clone-readiness checks on that clip.

### Clip Diagnostics

- The diagnostics section is the first place to look before building a new speaker prompt.
- The score badge gives a quick readiness summary such as `Fair`, `Good`, or `Ready`.
- The summary line explains why the clip scored the way it did.
- `Run Diagnostics` reruns the analysis on the selected clip.
- `Play Source Clip` previews the currently selected source file.
- `Delete Source Clip` removes the selected source file from the prep library.
- `Apply Recommended Prep` copies the recommended cleanup settings into the prep controls.
- `Use Suggested Trim` applies the diagnostic trim window to the trim controls.
- `Reset Prep Controls` clears the prep settings back to the safer defaults.
- The metrics grid explains the main technical traits of the clip, such as:
  - duration
  - channel count
  - sample rate
  - level
  - peak
  - silence percentage
- The recommendations list tells you what to fix before you use the clip for cloning.

### Prepare Clip

- `Trim Start` and `Trim End` let you isolate the strongest speaking section.
- `Output Name` controls the saved filename of the prepared result.
- `Save Prepared Clip To` decides whether the output becomes:
  - a live speaker prompt in `shared/audio/speakers`
  - another source clip in `shared/audio/source_clips`
- `Convert to mono` is the safest default for speaker prompts.
- `Normalize loudness` helps bring weak clips into a healthier range.
- `Noise cleanup` is useful when the clip is clearly noisy, but can make clones less natural if pushed too hard.
- `Vocal isolation` is a rescue tool for messy clips, not something you should turn on automatically.
- `Target Peak` controls the normalization ceiling.
- `Noise Reduction Strength` controls how aggressive cleanup should be.
- `Prepare Selected Clip` runs the prep pipeline without promoting the result directly into the live speaker set.
- `Prepare And Create Speaker` runs the prep and saves the result straight into the live speaker folder.
- `Play Prepared Output` previews the processed result after a prep run.
- `Load Output As Active Source` lets you immediately reuse the prepared output for another polish pass.

### When To Use This Tab

- Use `Speaker Prep` first when a voice sounds weak, robotic, noisy, or unstable.
- Use it again when you replace a speaker with a cleaner source clip.
- Skip it only when you already know the prompt clip is clean, dry, and clone-ready.

## Conversation Workflow

![Conversation Workflow Screenshot](../assets/manual/conversation-workflow-tab.png)

`Conversation Workflow` is the fastest way to create a multi-speaker scene from a script.

### Project Save / Load

- `Save Filename` controls the project save filename.
- `Saved Projects` lists the saved conversation projects currently available.
- `Start New Project` clears the current conversation-building state without deleting saved projects.
- `Save Current Project` stores the current script, settings, results state, and selections.
- `Refresh List` reloads the available saved projects.
- `Load Selected Project` restores a saved project into the workflow and results views.
- `Delete Selected` removes the chosen saved project.

### Available Voices

- This section shows every live speaker prompt currently loaded from `shared/audio/speakers`.
- Each card shows:
  - the speaker name
  - the exact script label to use
  - the source filename
  - the file size
- `Refresh Voices` reloads the speaker list.
- This app is BYO-voices for release; it does not bundle private voice files.

### Conversation Script

- `Conversation Title` is optional but useful for saved projects and manual exports.
- `Script` expects one line per speaker turn in `SpeakerName: text` format.
- `Load script pack` imports a reusable Markdown script pack from `test_scripts` or any compatible pack file.
- `Auto-detect emotions from text` lets the app try to infer a usable emotion profile before generation.
- `Parse Script` validates the speaker labels, breaks the script into lines, and prepares the script preview and pacing controls.
- `Clear` clears the current script editor contents.
- The `Script Preview` shows the parsed conversation lines before generation.

### Generation Settings

- `Quality Preset` chooses the high-level generation mode.
- `Versions per Line` sets how many candidates to generate for each line.
- `Similarity Threshold` sets the target cloning threshold used by review logic and regeneration tools.
- `Auto-regen Attempts` controls how many extra attempts the app should make automatically.
- `Seed Strategy` controls how seeds are assigned across versions.
- `Per-Speaker Delivery` lets you gently slow down or speed up individual speakers when one voice consistently feels rushed or sluggish.

### Pacing Controls

- `Dialogue Pace Preset` provides ready-made scene pacing modes such as `Natural`, `Calm`, `Argument`, and `Panic`.
- `Scene Pace` controls the overall gap and rhythm profile for the scene.
- `Base Gap Between Lines` sets the default space between lines before punctuation-aware shaping is added.
- `Respect punctuation pauses` tells the exporter to use punctuation as a soft rhythm hint.
- The pacing status panel tells you whether the current scene is using custom per-speaker pacing.

### Generation Controls

- `Emotion Weight` controls how strongly emotion shaping should influence the result.
- `Use Random Sampling` can increase variety but usually lowers voice-cloning fidelity.
- `Generate Conversation` launches generation for the parsed script.
- The generation progress panel appears during generation and reports the live backend status.

### Advanced Settings

- `Advanced Settings` contains the lower-level decode and model controls.
- These are best used after you already have a reliable workflow with the safer presets.
- Controls here include:
  - maximum text tokens
  - sampling toggles
  - top-p
  - top-k
  - temperature
  - beam count
  - repetition penalty
  - related low-level generation settings

### When To Use This Tab

- Use `Conversation Workflow` when you want the fastest script-to-audio path.
- Use it to generate the first pass of a scene before moving into detailed review or timeline editing.

## Conversation Results

![Conversation Results Screenshot](../assets/manual/conversation-results-tab.png)

`Conversation Results` is the review, selection, and export area for generated conversation lines.

### Select Conversation

- The conversation list shows the generated tasks available in the current backend state.
- Each item shows:
  - conversation ID
  - status
  - progress
- `Open In Timeline` sends the selected conversation into the timeline editor as an editable scene.

### Line Versions

- Each line block shows the speaker, the current text, and the selected/final status.
- `Auto-Select Best` picks one candidate version per line automatically.
- `Clear Selections` clears the current final-version choices.
- The editable text field lets you change a line before targeted regeneration.
- Each version card shows:
  - version number
  - similarity score
  - quality score
- Each version card includes:
  - `Play`
  - `Compare`
  - `Download`
- The listening review sub-panel on each version lets you capture structured human listening notes.
- Each line also includes regeneration controls:
  - `Regenerate All`
  - `Regenerate Below Threshold`
- The line-level status badge makes it clear whether the line still needs a final selected version.

### Listening Review Export

- `Copy Reviews` copies the structured listening notes to the clipboard.
- `Download Reviews` saves the review block to disk.
- `Clear Reviews` clears the locally stored review notes for the current conversation.
- This section is meant for human listening feedback rather than model output.

### Seed Reproducibility Export

- `Copy Seed Report` copies the seed metadata for the current conversation.
- `Download Seed Report` saves that metadata as JSON.
- This section is useful when you want to reproduce or compare a generation run later.

### Concatenation & Export

- The readiness summary tells you whether every line has one final selected version.
- `Concatenate Audio` creates the full conversation export from the selected line versions.
- `Download Selected` downloads the currently selected line versions individually.
- The overlap planning box accepts an optional planning document for special timing behavior.
- `Scene pacing`, `Base gap`, and punctuation pause settings shape the final exported rhythm.
- `Output format` selects the final export format.
- `MP3 bitrate` only applies when the selected output format is MP3.
- `Match line volumes on export` normalizes the selected lines before the final mix.
- `Target loudness` and `Peak safety limit` control the finishing loudness behavior.
- `Trim the final mix if overlap or gain makes it too loud` protects the export from peak overload.
- `Trim leading silence` and `Trim trailing silence` clean the edges of the finished export.
- `Fade in` and `Fade out` apply a short entrance or exit fade to the final exported mix.
- After a successful concatenate run, the player actions let you preview or download the final conversation.

### When To Use This Tab

- Use `Conversation Results` whenever you care about choosing the best line versions manually.
- Use it before export if you want the most deliberate final conversation.

## Timeline Editor

![Timeline Editor Screenshot](../assets/manual/timeline-editor-tab.png)

`Timeline Editor` is the hand-built scene editor. It is no longer just an import target; it can now build scenes from scratch.

### Start A Timeline

- `New Timeline Name` sets the name for a new blank scene.
- `Saved Timelines` lists the saved timeline projects currently available.
- `Create Blank Timeline` creates an empty scene without needing a conversation first.
- `Import Selected Conversation` turns the selected conversation results into a timeline project.
- `Refresh Timelines` reloads the project list.
- `Load Selected Timeline` opens the chosen timeline project.
- `Delete Selected` removes the chosen saved timeline.

### Standalone Scene Builder

- This is the new scratch-building area for timeline-first work.
- `Add Speaker Track` lets you create a new lane for a speaker directly inside the editor flow.
- `Track Name` controls the lane name shown in the timeline.
- `Track Speaker` chooses which speaker prompt powers that track.
- `Add Track` creates the lane immediately.
- `Write Segment` is the quick segment-creation block.
- `Segment Text` is the spoken line for the next segment.
- `Track` chooses which timeline track will receive the segment.
- `Emotion` sets the quick emotion hint for that segment.
- `Start Time` controls the segment start when auto-placement is off.
- `Visual Duration` controls the initial visible length on the timeline.
- `Place the new segment at the end of the chosen track automatically` lets the editor append the next segment to the clean end of that track.
- `Add Segment To Track` writes a new segment onto an existing track.
- `Add Track + Segment` creates a track and places the first segment in one action.
- `Open Advanced Segment Modal` opens the more detailed segment editor modal.

### Render & Export

- `Output format` selects WAV, MP3, or OGG.
- `MP3 bitrate` controls bitrate when MP3 is selected.
- `Match segment volumes before mixing` levels individual clips before the mix.
- `Duck interrupting overlaps on export` lowers interrupted material during overlaps.
- `Overlap ducking` controls how far ducking pulls the quieter track down.
- `Ducking fade` controls how quickly the ducking ramps in and out.
- `Target loudness` and `Peak safety limit` shape the final mix loudness.
- `Trim the final mix if overlaps push it too loud` protects the overall export from overload.
- `Trim leading silence` and `Trim trailing silence` clean the rendered mix boundaries.
- `Fade in` and `Fade out` apply short fades to the final rendered mix.
- `Generate Selected Audio` generates audio only for the selected segment.
- `Generate Missing Audio` fills in segments that still do not have audio.
- `Export Timeline` renders the current timeline into a finished audio file.
- `Preview Timeline` plays the current mixed arrangement.
- `Download Mix` downloads the finished timeline mix.
- `Pop Out Editor` opens the timeline in a focused standalone window.

### Timeline Canvas

- The top ruler shows the scene timeline in seconds.
- The ruler and lanes now share one aligned grid so clip positions match the visible second marks.
- Each track header includes:
  - track name
  - speaker filename
  - level slider
  - `Mute`
  - `Solo`
  - quick `Segment` add action
- You can add a segment directly from the lane area without going back to the upper forms.
- Double-clicking an empty part of a lane opens segment creation at that exact time.
- Empty tracks show an inline `Add segment here` action.
- Segment blocks can be selected for deeper editing.
- The canvas is designed for manual placement, interruptions, and scene timing decisions.

### Selected Segment

- The selected segment panel shows the current segment text, timing, and audio status.
- `Edit Segment` opens the edit modal.
- `Generate Audio` generates or regenerates audio for the selected segment.
- `Play Segment` previews that segment.
- `Split Segment` breaks one segment into two editable pieces.
- `Pop Out Editor` opens the editor in a focused window.
- `Delete Segment` removes the segment.
- The waveform preview appears after audio exists for the segment.

### What Makes The Timeline Professional

- A shared ruler/grid so time marks and clips line up.
- Track-level actions close to the lane, not hidden away from the canvas.
- Quick segment insertion from the track header or lane itself.
- Canvas-first editing with drag, placement, and timing cues.
- Export controls close to the editor so arrangement and render decisions stay in one place.

## Runtime Folders You Should Know

- Live speakers: `shared/audio/speakers`
- Original speaker backups: `shared/audio/speakers_backups`
- Raw prep clips: `shared/audio/source_clips`
- Final exports: `shared/audio/outputs`
- Per-line temporary audio: `shared/audio/temp_conversation_segments`
- Saved projects: `shared/data/project_saves`
- Saved timelines: `shared/data/timeline_projects`

## Refreshing The Manual Screenshots

This repo includes a reusable screenshot capture script:

```powershell
cd tools\manual
node .\capture_user_manual_screenshots.mjs
```

That command refreshes:

- `docs/assets/manual/conversation-workflow-tab.png`
- `docs/assets/manual/speaker-prep-tab.png`
- `docs/assets/manual/conversation-results-tab.png`
- `docs/assets/manual/timeline-editor-tab.png`

## Last Note

If the UI changes later, refresh the screenshots first and then update the matching section in this manual. That keeps the guide trustworthy and avoids a drift between the app and the documentation.
