# IndexTTS2 Prompting Guide

This guide explains how to write text that Draft to Take can send cleanly to IndexTTS2.

## The Big Idea

Treat these as separate controls:

- speaker label chooses the local prepared voice
- spoken text gives the words
- emotion vectors steer delivery
- timeline placement controls timing, gaps, overlap, and export

Do not try to force everything into the spoken text. The app has better controls for delivery and timing.

## Best Script Shape

Use:

```text
Speaker Label: Clear spoken line.
Other Speaker: Another clear spoken line.
```

Rules:

- Use exact local speaker or character labels.
- Use one speaker per line.
- Use one utterance per line.
- Aim for 6 to 18 words.
- Avoid going past 22 words unless there is a good reason.
- Split long thoughts where a real person would breathe, react, or be interrupted.
- Use natural punctuation.
- Keep raw spoken text free of bracketed stage directions.

## What To Avoid

Avoid:

```text
Speaker One: (angrily slamming table) I told you this would happen!!!
Speaker Two: [whispering nervously] I do not know what to do...
```

Prefer:

```text
Speaker One: I told you this would happen.
Speaker One: Nobody listened.
Speaker Two: Keep your voice down.
Speaker Two: I do not know what to do.
```

Why:

- parenthetical directions may be spoken literally
- punctuation spam can sound forced
- short lines give the model cleaner phrase boundaries
- emotion should be controlled by wording and app-side emotion controls

## Emotion Controls

Draft to Take uses the official IndexTTS2-style vector set:

```text
joy, anger, sadness, fear, disgust, low_mood, surprise, calm
```

Limits:

- each emotion is capped at `0.5`
- total vector sum is capped at `1.5`
- subtle values usually sound better than max values

Good emotion comments for import:

```text
Captain Quibble: Nobody panic. <!-- emotion: calm=0.28 fear=0.1 joy=0.04 -->
Zini Spark: The rain laughed when I said my name. <!-- emotion: fear=0.3 surprise=0.12 sadness=0.08 -->
```

Avoid:

```text
Professor Plink: Welcome. <!-- emotion: joy=1.4 -->
```

## Emotion Text

Some paths can use short emotion text prompts. Keep them simple and delivery-focused:

- `quiet concern`
- `dry disbelief`
- `barely contained panic`
- `warm narrator`
- `tired sarcasm`

Avoid plot summaries or timing instructions:

- `He is angry because the app crashed in scene two`
- `Say it at exactly 3.2 seconds`

## Punctuation

Punctuation helps, but it is not a timing system.

- comma: light pause or turn in thought
- period: clean stop
- question mark: questioning contour
- exclamation mark: occasional emphasis
- ellipsis: hesitation or trailing off, used sparingly

For exact timing, use the timeline.

## Text Cleanup

Draft to Take may normalize text before TTS to reduce unstable pronunciation.

Examples:

- `We're ready` can become `We are ready`.
- `It'll work` can become `It will work`.
- `*very serious*` can become `very serious`.

This helps keep the canvas readable while giving IndexTTS2 cleaner input.

## Source Clip Quality

The prepared speaker prompt still matters a lot.

Best source clips:

- 8 to 20 seconds
- one clear speaker
- dry audio
- low room noise
- no background music
- no heavy reverb
- no overlapping voices
- natural pacing

If a voice sounds weak, robotic, or unstable, fix the speaker prompt before over-tuning the script.

## Long Scripts

For sitcoms, podcasts, audiobooks, or long scenes:

- plan chapters and scenes first
- draft one scene or page at a time
- carry continuity context forward
- keep lines short even when the episode is long
- build timeline clips per scene or full episode
- lock good takes before regenerating weak ones

The writing model can preserve continuity, but the TTS text should stay line-level and speakable.

## Practical Quality Notes

- Use `Balanced` as the first quality preset.
- Listen before deciding a take is good.
- Lock strong takes before retrying weak ones.
- Avoid overdriven emotions.
- Avoid very long lines.
- Use clean source voices.
- Use SFX, ambience, and music tracks for sound design instead of putting sound descriptions into dialogue.

## Verified Sources

- Official IndexTTS repo: <https://github.com/index-tts/index-tts>
- Official IndexTTS2 model card: <https://huggingface.co/IndexTeam/IndexTTS-2>
- Official demo code: <https://github.com/index-tts/index-tts/blob/main/webui.py>
- Paper: <https://arxiv.org/abs/2506.21619>
