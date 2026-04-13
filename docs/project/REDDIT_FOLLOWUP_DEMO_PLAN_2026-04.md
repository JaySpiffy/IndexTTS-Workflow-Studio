# Reddit Follow-Up Demo Plan

This follow-up is meant to answer the two clearest asks from the first round of posts:

- hear actual output
- see the timeline workflow in motion

## Assets

- Audio sample: `docs/assets/social/audio/podcast_roundtable_demo_pack.mp3`
- Audio sample: `docs/assets/social/audio/audiobook_night_train_demo_pack.mp3`
- Audio sample: `docs/assets/social/audio/game_dialogue_checkpoint_breach_pack.mp3`
- Timeline workflow video: `docs/assets/social/timeline-workflow-demo.webm`
- Timeline workflow still: `docs/assets/social/timeline-workflow-demo.png`

## Source Packs

- `test_scripts/podcast_roundtable_demo_pack.md`
- `test_scripts/audiobook_night_train_demo_pack.md`
- `test_scripts/game_dialogue_checkpoint_breach_pack.md`

## Render Order

1. Podcast roundtable
2. Audiobook night train
3. Game dialogue checkpoint breach

Use `tools/manual/render_public_demo_packs.ps1` to regenerate the public-safe audio samples against the local speaker library.

Use `tools/manual/capture_reddit_followup_assets.mjs` to refresh the timeline workflow video and still.

## Recommended Follow-Up

Primary target: `r/AudioAI`

Reason:
The strongest explicit request there was to hear real output, and this follow-up directly answers that with short samples plus a timeline workflow clip.

## Suggested Post Title

`Follow-up: IndexTTS Workflow Studio now has 3 reusable demo packs + timeline workflow clip`

## Suggested Post Body

Built a follow-up for the people who asked to hear real results instead of just screenshots.

I put together 3 short public-safe demo packs for different use cases:

- podcast roundtable
- audiobook excerpt
- game dialogue

I also recorded a short timeline workflow clip so you can see the review -> select -> timeline -> export flow in one pass.

Demo assets in this follow-up:

- podcast sample
- audiobook sample
- game dialogue sample
- timeline workflow video

What I am trying to make this tool good at:

- local multi-speaker generation
- review and selective regeneration per line
- timing and overlap control in the timeline editor
- reusable packs/templates instead of one-off scripts

If you tried the earlier build and had notes about pacing, sample quality, or export workflow, I would love to hear what still feels rough.

## Posting Checklist

- Confirm the three audio files play cleanly end to end.
- Confirm the timeline video opens and the exported frame is readable at Reddit embed size.
- If you want the audio samples inside the Reddit post itself, convert them into a single preview video or push the assets somewhere public first. Reddit is still a poor home for standalone audio files.
- Upload the still image only if the subreddit prefers image/video preview over external links.
- Link the repo plus mention that the demo packs are reusable and public-safe.
- Keep the first comment ready with the three pack names and what each one is meant to show.
