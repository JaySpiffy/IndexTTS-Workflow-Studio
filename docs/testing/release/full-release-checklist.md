# Full Release Checklist

## Purpose

Run this before a public release, repo promotion, or major announcement.

## Step 1: Runtime And Environment

- Run [../smoke/00-stack-startup-smoke.md](../smoke/00-stack-startup-smoke.md)
- confirm frontend and backend health endpoints are healthy
- confirm the header shows the expected runtime mode
- confirm your local voices are present only because you added them, not because the repo ships them

## Step 2: Tab-by-Tab Smoke

- run [../smoke/01-speaker-prep-smoke.md](../smoke/01-speaker-prep-smoke.md)
- run [../smoke/02-conversation-workflow-smoke.md](../smoke/02-conversation-workflow-smoke.md)
- run [../smoke/03-conversation-results-smoke.md](../smoke/03-conversation-results-smoke.md)
- run [../smoke/04-timeline-editor-smoke.md](../smoke/04-timeline-editor-smoke.md)

## Step 3: Contract Tests

Run the focused backend tests referenced in [../../project/RELEASE_HYGIENE_CHECKLIST.md](../../project/RELEASE_HYGIENE_CHECKLIST.md).

At minimum, verify the current high-risk paths:
- generation progress
- queue behavior
- seed handling
- line emotion flow
- speaker prep
- pacing
- export finishing
- selection gating

## Step 4: Docs And Repo Surface

- confirm [../../../README.md](../../../README.md) matches the shipped app
- confirm [../../manual/USER_MANUAL.md](../../manual/USER_MANUAL.md) matches the current UI
- confirm manual screenshots and videos are current enough for release
- confirm release notes and known limitations are up to date

## Step 5: Asset And Privacy Hygiene

- run `powershell -ExecutionPolicy Bypass -File tools/audit_private_assets.ps1`
- confirm no local-only speaker assets are being shipped in the repo
- confirm `shared/audio/speakers`, `shared/audio/source_clips`, and `shared/audio/speakers_backups` remain local-only
- archive or remove throwaway smoke outputs before final screenshots or demos

## Step 6: Final Human Listening Pass

Use at least three benchmark scenes:
- calm or balanced dialogue
- interruption or argument scene
- timeline-edited mix with overlap or pacing changes

Listen for:
- voice similarity
- pacing realism
- loudness consistency
- overlap intelligibility
- unnatural robotic moments

## Step 7: Record The Run

- copy [../test-run-log-template.md](../test-run-log-template.md)
- save the completed log under [../runs/](../runs/)
- record what passed, what failed, and any blockers before calling the release ready

## Pass Condition

Call the release ready when:
- all smoke packs pass
- targeted backend tests pass
- docs are current
- no private assets are leaking
- the final human listening pass sounds good enough to stand behind
