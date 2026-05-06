# Draft to Take Beta User Manual

Draft to Take is a local-first audio production studio for turning scripts into generated dialogue, reviewed takes, timeline clips, and exported mixes.

The main beta workflow is:

```text
Voice Studio -> Characters -> Studio / Script Canvas -> Embedded Timeline -> Download Mix
```

You can also skip the AI Thread and paste or import your own script directly.

## Quick Start

1. Start Docker Desktop and wait until it is fully running.
2. Double-click `start.bat` in this repo.
3. Open the frontend URL printed in the terminal, usually `http://localhost:3000`.
4. Open `Voice Studio` to create or prepare reusable voice assets.
5. Open `Characters` to turn those voices into cast profiles.
6. Open `Studio` and start a blank Script Canvas project.
7. Write, paste, import, or ask the optional AI Thread to draft your script.
8. Place the active scene or full episode on the embedded timeline.
9. Detect emotions if you want Qwen to set IndexTTS2 delivery vectors.
10. Generate audio, listen, lock good takes, retry weak takes, then download the final mix.

First launch can be slow because Docker images and model files are large. Keep the launcher terminal open while it pulls images, downloads models, and starts services.

## Home

The Home page is a launch pad:

- `Studio` opens the Script Canvas workflow.
- `Voice Studio` opens reusable voice creation and source clip prep.
- `Characters` opens cast profiles linked to voices.
- `Recent Projects` opens saved Script Canvas projects.
- Delete buttons let you remove old recent projects when the list gets noisy.

Saved projects live under:

```text
%USERPROFILE%\DraftToTake\shared\data\project_saves
```

## Voice Studio

A voice is a reusable audio asset. It is not a character yet. One voice can be linked to many characters later.

### Create Voice

Use `Create Voice` to generate a reusable synthetic voice through the optional OmniVoice sidecar.

The beta uses dropdowns for:

- gender
- age
- pitch
- style
- English accent

The app builds an English-only OmniVoice instruction from those choices. Generated WAV files are saved as prepared voices, then Script Canvas still renders final dialogue with IndexTTS2.

### Voice Library

The Voice Library lists prepared speaker WAV files. Badges show where a voice came from:

- `AI Synth` for generated synthetic voices.
- `Audio Clone` for prepared or uploaded source-audio voices.

Use the library views to keep large collections manageable:

- `Compact` for table-like review.
- `Roomy` for more readable rows.
- `Grid` for compact cards, up to several per row on wide screens.

Selecting a voice opens the side inspector with full metadata and actions. Use `Play` to audition, `Create Character` to cast it, and `Delete` to remove unwanted voices.

### Source Clips

Source Clips are raw audio files waiting for cleanup, diagnosis, and promotion into the voice library.

Good source clips are usually:

- 8 to 20 seconds.
- one clear speaker.
- dry audio with low room noise.
- no background music.
- no overlapping voices.
- natural pacing.

Use Source Clip prep to trim, normalize, diagnose, and prepare a better speaker prompt before generating final dialogue.

## Characters

Characters are the LLM-facing cast profiles that Script Canvas uses.

Each character has:

- exact script label
- linked reusable voice
- role aliases
- speaking style
- rhythm
- role notes
- optional example lines
- optional comedy function, emotional range, relationship notes, and scene role

Keep character profiles focused on writing and performance. Engine defaults and safety limits belong in the app, not in every character.

One prepared voice can be reused by multiple characters.

## Script Canvas

Script Canvas is the main writing-to-audio workspace.

### Creative Flow

The beta supports three writing shapes:

- `Script / Episode` for scenes, sitcoms, podcasts, plays, and screen-style dialogue.
- `Audiobook` for book chapters and page-sized narration sections.
- `Speech / Monologue` for talks, presentations, and single-speaker work.

Audiobook mode can import text or Markdown as page-sized sections so a whole book does not have to be sent through the model at once.

### AI Thread

The AI Thread is optional and can be turned off in Settings. When enabled, it can help plan, draft, revise, and ask quick multiple-choice questions.

The Qwen sidecar is also used for emotion detection, so turning off the AI Thread does not mean emotion detection has to be unavailable.

Good AI Thread requests:

```text
Plan a 10 minute sci-fi comedy with Captain Quibble, Zini Spark, and Bolt Crumple.
```

```text
Rewrite the selected scene so it is tenser, but keep the same speaker labels.
```

```text
Add two more short lines before the gate opens.
```

### Raw Script Format

Script Canvas expects one production line per row:

```text
Speaker Label: spoken words go here
```

Use exact character labels. Keep lines short and speakable.

Bad:

```text
Beat 1: The heroes enter.
Captain Quibble: (shouting angrily) Nobody panic while thunder explodes above us!!!
```

Good:

```text
Captain Quibble: Nobody panic.
Captain Quibble: Houses cannot remember names.
Bolt Crumple: Then why did the letterbox whisper mine?
```

### Markdown Import

Use `Import MD` when you want to prepare a script outside the app.

Markdown import can preserve:

- chapters
- scenes
- dialogue lines
- emotion vectors
- SFX markers
- ambience markers
- music markers

Example:

```text
## Chapter 1: The Door That Remembers

### Scene 1: Arrival In The Rain
Professor Plink: Rain climbed Harrow House in silver threads. [[AMBIENCE: steady cold rain outside, distant road water, no voices]] <!-- emotion: fear=0.28 low_mood=0.14 calm=0.04 -->
Bolt Crumple: Then why did the letterbox whisper my punishment nickname? <!-- emotion: fear=0.3 surprise=0.12 anger=0.04 -->
```

### Canvas Actions

- `Place Active Scene` puts the current scene on the timeline.
- `Full Episode Timeline` places every drafted chapter and scene.
- `Detect Active Scene Emotions` detects vectors only for the active scene.
- `Detect Timeline Emotions` detects vectors across the loaded timeline.
- `Generate Audio` renders missing dialogue first, then batches sound cues.
- `Judge Takes` checks generated takes without replacing them.
- `Retry Bad Takes` keeps locked good takes and regenerates weak unlocked ones.
- `Copy Script` copies the current script.

## Emotion Controls

Draft to Take uses Qwen to suggest IndexTTS2 emotion vectors.

Official vector order:

```text
joy, anger, sadness, fear, disgust, low_mood, surprise, calm
```

Limits:

- each emotion is capped at `0.5`
- total vector sum is capped at `1.5`
- most lines sound better with subtle values

Manual sliders let you adjust a selected line. If a line already has manual emotion vectors, audio generation should not overwrite them unless you explicitly detect emotions again.

## SFX, Ambience, And Music

SFX, ambience, and music are optional beta features. They require the SFX/music sidecar to be enabled in `.env`.

```text
INDTEXTS_SFX_ENABLED=true
```

Then run `start.bat` again.

Script Canvas markers:

```text
[[SFX: iron gate latch lifting by itself, wet hinge tremor | duration=2.2]]
[[AMBIENCE: steady cold rain outside, distant road water, no voices]]
[[MUSIC: low string dread bed, slow pressure pulse, no vocals | duration=24]]
```

These become separate timeline tracks, so they can overlap dialogue. Ambience is best for scene-wide beds such as rain, park noise, harbor air, room tone, traffic, or spaceship hum.

Generated SFX and music still need auditioning. Lock good takes and delete bad ones from the library when you are done.

## Embedded Timeline

The old standalone timeline page is not the main workflow anymore. Timeline editing now happens inside Script Canvas.

Timeline basics:

- dialogue, SFX, ambience, and music can live on separate tracks
- ambience, SFX, and music can overlap spoken lines
- track controls help with volume, mute, solo, lock, and collapse
- selected dialogue clips show text and emotion controls
- selected sound clips show prompt, duration, volume, and regenerate controls
- preview the mix before downloading

Long speaker names are shortened on track labels so the timeline stays readable.

## Audiobook Canvas

Audiobook mode is for long prose.

The intended flow is:

1. Upload or paste a text or Markdown book/chapter.
2. Let the app split it into book chapters and page-sized sections.
3. Process one section at a time.
4. Keep a running memory of characters, tone, pronunciation, and plot state.
5. Review or regenerate individual pages without rebuilding the whole book.

Use `Preserve text` when you want the audiobook to stay faithful to the source. Use `Adapt text` only when you want Qwen to rewrite prose into a more performable script.

## Where Files Go

Your persistent local data is stored here:

```text
%USERPROFILE%\DraftToTake\shared
```

Important folders:

- `shared\models` - downloaded model files.
- `shared\models\checkpoints` - IndexTTS2 checkpoints and model cache.
- `shared\models\llm` - Qwen GGUF files.
- `shared\audio\speakers` - prepared speaker WAV files.
- `shared\audio\source_clips` - raw source clips.
- `shared\audio\sfx` - generated or imported SFX assets.
- `shared\audio\ambience` - generated or imported ambience beds.
- `shared\audio\music` - generated or imported music assets.
- `shared\audio\outputs` - exported mixes.
- `shared\data` - saved projects, timelines, and app data.

## Troubleshooting

### The app looks stuck on first start

Large Docker images and models may still be downloading. Keep the terminal open and watch for progress or errors.

### GPU is not detected

Check Docker Desktop WSL2 integration and NVIDIA GPU support. CPU fallback may work, but it is much slower.

### Audio sounds robotic or breaks up

Try shorter lines, cleaner speaker clips, lower emotion intensity, and the balanced quality preset. Lock good takes before retrying weak ones.

### SFX or music fails

SFX/music is optional and heavier than dialogue. Make sure `INDTEXTS_SFX_ENABLED=true`, Docker can use the GPU, and there is enough free VRAM.

### I need to report a bug

Run:

```text
collect-diagnostics.bat
```

Review the output before sharing it publicly. Do not post private scripts, speaker samples, generated audio, tokens, or personal data unless you are comfortable doing so.
