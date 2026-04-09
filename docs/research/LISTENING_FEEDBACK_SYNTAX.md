# Listening Feedback Syntax

This format gives us a compact, repeatable way to turn what you hear into something I can reason about.

## How To Rate A Clip

Use one block per clip version:

```text
CLIP=b4a5cd17/L1/V2
VERDICT=bad
SIMILARITY=2
NATURALNESS=2
PACE=1
ROBOTIC=5
CLARITY=3
EMOTION=2
ISSUES=too_fast,robotic,weak_similarity
ACTION=more_faithful,slower,cleaner_ref
NOTES=Voice is rushed and does not sound enough like Trump.
```

Leave a blank line, then add the next clip:

```text
CLIP=b4a5cd17/L1/V3
VERDICT=ok
SIMILARITY=3
NATURALNESS=3
PACE=3
ROBOTIC=3
CLARITY=4
EMOTION=3
ISSUES=slightly_flat
ACTION=keep_testing
NOTES=Closer to the target voice but still a bit synthetic.
```

## Score Meaning

All numeric scores are `1` to `5`.

- `SIMILARITY`: `1` = does not sound like the target speaker, `5` = very close match
- `NATURALNESS`: `1` = very synthetic, `5` = very human sounding
- `PACE`: `1` = much too fast, `3` = about right, `5` = much too slow
- `ROBOTIC`: `1` = not robotic, `5` = very robotic
- `CLARITY`: `1` = hard to understand, `5` = very clear
- `EMOTION`: `1` = wrong or flat emotion, `5` = convincing emotion

## Suggested Issue Tags

Use any words you want in `NOTES`, but these tags are useful because they are easy to group:

- `too_fast`
- `too_slow`
- `robotic`
- `metallic`
- `monotone`
- `weak_similarity`
- `muffled`
- `slurred`
- `harsh_s`
- `clipped_end`
- `noisy_ref`

## Suggested Action Tags

- `more_faithful`
- `slower`
- `faster`
- `cleaner_ref`
- `split_line`
- `less_emotion`
- `more_emotion`
- `more_clarity`
- `keep_testing`

## Parse It Into JSON

Save your listening notes to a file like `feedback.txt`, then run:

```powershell
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml exec backend python /app/backend/scripts/parse_listening_feedback.py /app/feedback.txt
```

That produces a structured JSON summary with:

- average scores
- top issue tags
- top requested actions
- quick tuning recommendations

## Why This Helps

It gives us a shared language for tuning.

Instead of:

```text
This one is kind of worse and weird.
```

we can say:

```text
CLIP=b4a5cd17/L1/V2
VERDICT=bad
SIMILARITY=2
NATURALNESS=2
PACE=1
ROBOTIC=5
CLARITY=3
EMOTION=2
ISSUES=too_fast,robotic,weak_similarity
ACTION=more_faithful,slower,cleaner_ref
```

That is much easier for me to compare across presets, lines, and speaker clips.
