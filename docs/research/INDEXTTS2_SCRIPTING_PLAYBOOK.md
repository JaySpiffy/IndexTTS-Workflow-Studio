# IndexTTS2 Scripting Playbook

This project is a local app built on top of the official IndexTTS2 models.

The placeholder speaker labels in this guide, such as `SpeakerOne` and `SpeakerTwo`, are generic public examples. Replace them with the exact voice labels shown in your own `Available Voices` panel when you use the app.

The most important scripting truth is simple:

- punctuation helps
- line length helps
- wording helps
- but public IndexTTS2 still does not expose precise duration control as a normal user-facing feature

So the best results come from a mix of:

1. clean source clips
2. short, speakable lines
3. deliberate punctuation
4. emotion controls when needed
5. app-level timing tools such as overlap plans and the timeline editor

## What The Official Material Suggests

- Official IndexTTS v1.5 explicitly says pauses can be controlled through punctuation marks.
- Official IndexTTS2 exposes speaker conditioning, emotional reference audio, emotion text, and direct emotion vectors.
- Official IndexTTS2 also says the precise duration-control research feature is not yet enabled in the public release.

Practical conclusion:

- punctuation is still a real prosody hint
- punctuation is not a replacement for true timing control
- when scene timing matters, use the app's overlap and timeline tools as well

## Core Writing Rules

### 1. One speaker, one utterance, one line

Prefer:

```text
SpeakerSix: This is a disaster.
SpeakerFour: Are you serious?
```

Avoid:

```text
SpeakerSix: This is a disaster. SpeakerFour: Are you serious?
```

### 2. Use the exact speaker label the app expects

Use the labels shown in the app's `Available Voices` panel, for example:

```text
SpeakerOne:
SpeakerFour:
SpeakerSix:
```

### 3. Keep lines short and speakable

Best range:

- ideal: 6 to 18 words
- still usually fine: up to about 22 words
- above that: pacing often gets flatter, faster, or more robotic

If a thought feels long, split it.

Prefer:

```text
SpeakerTwo: This is a terrible idea.
SpeakerTwo: It is also, unfortunately, very scalable.
```

Avoid:

```text
SpeakerTwo: This is a terrible idea, but also unfortunately very scalable, and that is the kind of sentence that tends to rush in TTS.
```

### 4. Write for speech, not for prose

Prefer contractions, natural phrasing, and complete spoken thoughts.

Prefer:

```text
SpeakerFour: I don't know, man. That sounds insane.
```

Less natural:

```text
SpeakerFour: I do not know, my friend. That sounds highly irrational.
```

### 5. Put the emotion in the wording first

Punctuation helps shape delivery, but the words do most of the work.

Weak:

```text
SpeakerOne: Wow!!!
```

Stronger:

```text
SpeakerOne: Oh no. No, this is bad. This is way worse than I thought.
```

## Punctuation Guide

Use punctuation as a performance hint, not a decoration layer.

### `,` comma

Use for a light pause or a slight turn in thought.

```text
SpeakerSix: Look, frankly, this whole thing is ridiculous.
```

### `.` period

Use for a neutral stop. This is the safest punctuation mark.

```text
SpeakerFour: That does not sound good.
```

### `?` question mark

Use for questioning contour or disbelief.

```text
SpeakerFour: Wait, that was your plan?
```

### `!` exclamation mark

Use sparingly for stronger attack or sharper emphasis.

```text
SpeakerSix: Get him out of here!
```

Too many `!` marks can sound forced or machiney.

### `...` ellipsis

Use sparingly for hesitation, dread, trailing off, or stunned silence.

```text
SpeakerOne: Oh no... this is getting worse.
```

Overusing ellipses can make everything drag.

### `:` colon and `;` semicolon

These are safe inside the spoken text because the parser splits on the first colon only, for example:

```text
SpeakerSix: Listen: this is the problem.
```

That said, use them sparingly. They read more like writing than speech.

### All caps

Avoid heavy all-caps writing. One emphasized word can work, but whole lines often sound stiff.

Better:

```text
SpeakerTwo: That is a bad idea. A very bad idea.
```

Worse:

```text
SpeakerTwo: THAT IS A VERY BAD IDEA!!!
```

## Pacing Patterns That Usually Work

### Calm explanation

- short declarative lines
- mostly periods and occasional commas
- very light emotion text if used

```text
SpeakerThree: Sit down.
SpeakerThree: We are going to solve this properly.
```

### Argument or interruption

- shorter lines
- stronger punctuation in a few key places
- separate the interruption into its own line
- use overlap planning or the timeline editor for true crosstalk

```text
SpeakerFour: That is not what I said.
SpeakerSix: Yes, it is.
SpeakerFour: No, it is not.
```

### Nervous spiral

- short starts
- hesitations
- line breaks where a person would breathe or panic

```text
SpeakerOne: No. No, hold on.
SpeakerOne: I do not like this at all.
SpeakerOne: This feels cursed.
```

### Villain monologue

- break the monologue into chunks
- let each line land a thought
- use commas and periods more than dramatic punctuation spam

```text
SpeakerTwo: People only panic when the branding is bad.
SpeakerTwo: If you rename the disaster, they call it strategy.
```

## Better Ways To Control Scene Timing

If you need real timing, do not try to force everything through punctuation.

Use these tools instead:

1. `emotion_text`
   - helps mood and delivery
2. overlap plan
   - use `allow_overlap` and `overlap_prev_ms` when interruptions should begin early
3. timeline editor
   - use for real start offsets, overlap, ducking, splitting, and scene shaping

## Suggested Script Workflow

### For high clone fidelity

- keep `Use Random Sampling` off
- prefer cleaner, shorter lines
- avoid punctuation spam
- choose words that fit the speaker naturally

### For emotionally guided lines

- write emotionally clear dialogue
- then add `emotion_text` if needed
- keep punctuation natural rather than theatrical

### For scenes with arguments or panic

- script short interjections
- do not cram both speakers into one line
- use overlap planning or the timeline editor after generation

## Good vs Bad Examples

### Bad

```text
SpeakerSix: THIS IS INSANE!!! WHY IS EVERYONE DOING THIS!!! I CANNOT BELIEVE THIS IS HAPPENING!!!
```

Problems:

- too long
- too much shouting punctuation
- likely to sound rushed and artificial

### Better

```text
SpeakerSix: This is insane.
SpeakerSix: Why is everyone doing this?
SpeakerSix: I cannot believe this is happening.
```

### Bad

```text
SpeakerOne: Oh my god this is like really bad and kind of creepy and I hate it and I think I am going to lose my mind here
```

### Better

```text
SpeakerOne: Oh my God. This is bad.
SpeakerOne: This is actually creepy.
SpeakerOne: I think I am losing my mind here.
```

## Recommended House Standard For This Repo

Use this as the default writing standard:

- exact app speaker labels
- one utterance per line
- target 6 to 18 words per line
- break long thoughts into multiple lines
- use punctuation intentionally
- avoid punctuation spam
- use overlap planning for real interruptions
- use timeline editing for serious scene shaping

## Source Notes

- Official IndexTTS v1.5: punctuation-controlled pauses are explicitly mentioned in the project README.
- Official IndexTTS2: speaker reference, emotion reference, emotion text, and emotion vectors are public, but precise duration control is not yet exposed as a normal release feature.
- Local parser note: this app parses script lines with `^([^:]+):\\s*(.+)$`, so it splits only on the first colon.
