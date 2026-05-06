# SFX, Ambience, And Music Smoke Test

Use this checklist when testing the optional sound-design sidecar.

SFX, ambience, and music are experimental beta features. They are disabled by default because they use heavier model-backed generation and may need more VRAM.

## Enable Sound Design

1. Open `.env`.
2. Set:

```text
INDTEXTS_SFX_ENABLED=true
```

3. Run:

```text
start.bat
```

4. Wait for the app to start and confirm the SFX service is available in the UI.

If Docker cannot see your NVIDIA GPU, SFX/music generation may fail or run too slowly to be useful.

## Smoke Script

Paste this into Script Canvas or save it as Markdown and use `Import MD`:

```text
## Chapter 1: Smoke Test

### Scene 1: Outside The Door
Narrator: Rain worked at the old street like it had a private grudge. [[AMBIENCE: steady cold rain on pavement, distant traffic, no voices]] [[MUSIC: low uneasy pulse, sparse strings, no vocals | duration=16]] <!-- emotion: fear=0.18 low_mood=0.14 calm=0.08 -->
Captain Quibble: Nobody touch the door until it agrees to behave. <!-- emotion: calm=0.28 fear=0.08 anger=0.04 -->
Bolt Crumple: It just winked at me through the keyhole. [[SFX: tiny brass keyhole click, close and dry | duration=1.2]] <!-- emotion: fear=0.24 surprise=0.14 joy=0.04 -->
Narrator: The latch lifted by itself, slow as a finger learning a trick. [[SFX: old iron latch lifting, wet hinge tremor | duration=2.2]] <!-- emotion: fear=0.3 surprise=0.12 low_mood=0.06 -->
Captain Quibble: Excellent. Polite machinery is always the worst kind. <!-- emotion: calm=0.22 fear=0.12 anger=0.06 -->
```

## Test Steps

1. Open `Studio`.
2. Start a blank Script Canvas project.
3. Paste or import the smoke script.
4. Confirm the production view shows dialogue lines without the marker text being spoken.
5. Click `Full Episode Timeline` or `Place Active Scene`.
6. Confirm the timeline has separate dialogue, SFX, ambience, and music tracks.
7. Click `Detect Active Scene Emotions`.
8. Confirm dialogue lines receive emotion vectors.
9. Click `Generate Audio`.
10. Wait for dialogue generation first, then SFX, ambience, and music generation.
11. Preview the timeline mix.
12. Download the mix.

## Expected Result

- Dialogue clips have generated audio.
- SFX clips are short and placed near the marker location.
- Ambience sits on its own track and can cover the scene.
- Music sits on its own track and can overlap dialogue.
- Preview and downloaded mix include dialogue plus sound-design assets.

## Useful Checks

- Lock good dialogue takes before retrying weak ones.
- Audition generated SFX and music; not every take will be good.
- Delete bad SFX/music/ambience assets from their libraries when you are done.
- If a generated ambience bed has silent gaps, regenerate it with a steadier prompt such as `continuous rain bed, no silence, no voices`.
- If a cue fails, check whether files appeared in the library anyway. Some model calls can finish writing files after the HTTP request times out.

## Common Problems

### HTTP 422 From SFX Service

This usually means the sidecar rejected the request. Try a shorter prompt, shorter duration, or fewer batch takes.

### Backend Temporarily Unavailable

The backend may be restarting or waiting on a sidecar. Wait a moment, then retry. If it keeps happening, run `collect-diagnostics.bat`.

### Not Enough VRAM

Dialogue, Qwen, OmniVoice, SFX, and music models all compete for VRAM. Draft to Take tries to unload idle managed models, but manually loaded or active models may still occupy memory.

Stop other GPU workloads, run `stop.bat`, then start again.

### Garbled SFX

Use more concrete foley prompts and generate multiple takes. Examples:

```text
short dry wooden knock, close microphone, no music, no speech
small metal latch click, single sound, close and dry, no voices
steady cold rain bed, no speech, no music, seamless loop
```
