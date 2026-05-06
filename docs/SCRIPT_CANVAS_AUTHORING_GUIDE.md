# Script Canvas Authoring Guide

Use this guide when writing scripts for Draft to Take Script Canvas by hand, with another AI, or in a Markdown file.

Script Canvas wants production text first:

- exact speaker labels
- short speakable lines
- optional emotion vectors
- optional sound markers placed where the cue should happen

## Dialogue Format

Write one production line per row:

```text
Speaker Label: spoken words go here
```

Rules:

- Use the exact character label from the app.
- Use one speaker per line.
- Keep one spoken thought per line.
- Aim for roughly 6 to 18 words.
- Split long thoughts at natural breath points.
- Do not write stage directions as spoken text.
- Do not write `Beat 1`, `Beat 2`, or placeholders as character dialogue.

Good:

```text
Captain Quibble: Nobody panic. Houses cannot remember names.
Bolt Crumple: Then why did the letterbox whisper my punishment nickname?
```

Bad:

```text
Beat 1: The heroes enter the house.
Captain Quibble: (angrily yelling) Nobody panic while the door opens and thunder crashes.
```

## Chapters And Scenes

Markdown headings can create long episodes:

```text
## Chapter 1: The Door That Remembers

### Scene 1: Arrival In The Rain
Professor Plink: Rain climbed Harrow House in silver threads.
```

Use headings for full episodes. Keep scenes small enough to review, regenerate, and place without rebuilding the whole project.

## SFX, Ambience, And Music Markers

Put sound markers inside the line where they should start.

The app strips markers before dialogue TTS, then places them on separate timeline tracks.

### SFX

SFX is for short events:

```text
Dr. Nerva Mopp: The brass knocker has a pulse. [[SFX: wet brass heartbeat under fingertips, close microphone | duration=2.0]]
```

Use SFX for:

- doors
- knocks
- sparks
- impacts
- machinery hits
- creature sounds
- brief foley events

### Ambience

Ambience is for scene-wide environmental beds:

```text
Professor Plink: Rain climbed Harrow House in silver threads. [[AMBIENCE: steady cold rain outside, distant road water, no voices]]
```

Use ambience for:

- rain
- park birds
- harbor air
- room tone
- traffic beds
- spaceship hum
- outside night air

Ambience usually does not need a duration. The timeline can stretch or loop it across the scene.

### Music

Music is for beds, stings, tension pulses, and transitions:

```text
Professor Plink: The house waited with all its windows shut. [[MUSIC: low uneasy horror pulse, sparse strings, no vocals | duration=18]]
```

### Sound Marker Rules

- Put the marker where the cue should start.
- Keep prompts concrete and audio-focused.
- Use `duration=1.2` or `| 1.2s` when exact timing matters.
- Prefer SFX durations from `0.5` to `3.0` seconds.
- Prefer music durations from `8` to `30` seconds.
- Do not put voice lines inside SFX, ambience, or music prompts unless you truly want vocal sound.
- For ambience, describe a stable place, not a pile of foreground actions.

Good:

```text
[[SFX: heavy old door folding inward, rain abruptly cut off | duration=2.6]]
[[SFX: dry wallpaper scratching itself into letters | duration=2.0]]
[[AMBIENCE: damp harbor street ambience, steady winter wind, distant wooden cart rumble]]
[[MUSIC: reversed music box lullaby, thin and distant, no vocals | duration=12]]
```

Avoid:

```text
[[SFX: boom]]
[[MUSIC: scary]]
[[AMBIENCE: harbor, quill scratch, rope creak, ship bells, crowd voices]]
```

## Emotion Vectors

Script Canvas can import or detect IndexTTS2 emotion vectors.

Official vector order:

```text
joy, anger, sadness, fear, disgust, low_mood, surprise, calm
```

Limits:

- each emotion must be `0.0` to `0.5`
- total vector sum must stay at or below `1.5`
- most natural lines should use subtle values
- strong emotion usually means one main emotion plus one or two small supporting emotions

Inline Markdown comment:

```text
Bolt Crumple: Tell it to stop borrowing my lungs. I use those. <!-- emotion: fear=0.22 anger=0.1 joy=0.06 -->
```

Good:

```text
Captain Quibble: We step in together. If the house counts, we make the math difficult. <!-- emotion: calm=0.28 fear=0.14 anger=0.04 -->
Zini Spark: It called me little matchstick in my grandmother's voice. <!-- emotion: fear=0.3 sadness=0.12 surprise=0.08 -->
```

Too much:

```text
Professor Plink: Welcome to the museum. <!-- emotion: joy=1.4 -->
```

Better:

```text
Professor Plink: Welcome to the museum. <!-- emotion: joy=0.22 surprise=0.08 calm=0.06 -->
```

## Prompt For Another AI

If you ask another AI to write for Draft to Take, give it this:

```text
Write for Draft to Take Script Canvas.
Use exact Speaker: line formatting.
Use only the provided character labels.
Keep lines short and speakable.
Place [[SFX: ... | duration=...]], [[AMBIENCE: ...]], or [[MUSIC: ... | duration=...]] inside the line where the cue should start.
Use AMBIENCE for continuous scene beds and SFX for short events.
Do not make sound cues into separate speakers.
Do not write stage directions into dialogue.
If adding emotion comments, use only joy, anger, sadness, fear, disgust, low_mood, surprise, calm.
Keep each emotion <= 0.5 and total <= 1.5.
```

## Mini Example

```text
## Chapter 1: The Door That Remembers

### Scene 1: Arrival In The Rain
Professor Plink: Rain climbed Harrow House in silver threads. [[AMBIENCE: steady cold rain outside, distant road water, no voices]] [[MUSIC: cold rain horror bed, low bowed metal, no vocals | duration=12]] <!-- emotion: fear=0.28 low_mood=0.14 calm=0.04 -->
Captain Quibble: Nobody panic. Houses cannot remember names. <!-- emotion: calm=0.28 fear=0.1 joy=0.04 -->
Bolt Crumple: Then why did the letterbox whisper my punishment nickname? <!-- emotion: fear=0.3 surprise=0.12 anger=0.04 -->
Dr. Nerva Mopp: The brass knocker has a pulse. [[SFX: wet brass heartbeat under fingertips, close microphone | duration=2.0]] <!-- emotion: disgust=0.16 fear=0.3 calm=0.06 -->
Professor Plink: Captain struck the knocker, and the door answered from inside the wood. [[SFX: deep knock answered by a hollow knock inside wood | duration=2.4]] <!-- emotion: fear=0.34 surprise=0.12 calm=0.04 -->
```

## What Happens In The App

1. Markdown import creates chapters, scenes, dialogue lines, emotion vectors, and sound markers.
2. `Full Episode Timeline` places dialogue, SFX, ambience, and music into one timeline.
3. Dialogue renders with IndexTTS2.
4. SFX, ambience, and music render through the optional sound-design sidecar.
5. Sound assets live on separate tracks and can overlap dialogue.
6. Ambience beds can cover a whole scene even when the generated bed is shorter.
7. The final mix exports dialogue, SFX, ambience, and music together.
