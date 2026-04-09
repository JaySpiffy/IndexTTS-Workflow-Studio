# IndexTTS2 Frontend

This folder contains the shipped web UI for the Docker-first IndexTTS2 app.

The frontend is served by the Docker stack and talks to the backend through the `/api/...` proxy path. For normal use, start the app with `..\docker\start.bat` and open [http://localhost:3000](http://localhost:3000).

## Current Layout

```text
frontend/
  index.html                  # Main app shell
  assets/
    css/                      # Shipped styling bundles
    images/                   # Static images and favicon
  src/
    app.js                    # Frontend bootstrap and cache-bust version
    custom-media-player.js    # Shared audio player and waveform UI
    modules/                  # Main tab/workflow modules
```

## Main UI Areas

- `Speaker Prep`
- `Conversation Workflow`
- `Conversation Results`
- `Timeline Editor`

## Development Notes

- Prefer verifying frontend changes through the Docker stack, not by teaching a separate host-managed workflow.
- After changing frontend assets, use a hard refresh if your browser still shows cached behavior.
- The build/version string lives in `src/app.js` and the cache-busted asset links in `index.html`.
- The main user-facing walkthrough is in [../docs/manual/USER_MANUAL.md](../docs/manual/USER_MANUAL.md).

## Archived Debug Assets

Older one-off debug pages, experimental HTML test harnesses, and legacy frontend files have been moved out of this folder into [../tools/frontend_debug](../tools/frontend_debug). They are kept for historical troubleshooting only and are not part of the release-facing app surface.
