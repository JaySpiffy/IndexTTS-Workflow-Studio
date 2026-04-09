# Release Readiness Status

Last updated: 2026-04-08

## Current State

v2 is through the main release-hardening pass and is now at final sign-off quality.

The major workflow gaps that used to block replacement are now covered:

- Docker-first GPU runtime with DeepSpeed
- project save/load
- seed workflow and seed visibility
- speaker prep in the main UI
- pacing controls
- selection gating before export
- richer export pipeline
- timeline editor with overlap-aware export
- public manual screenshots and videos sanitized for release
- fresh browser load, generation, and export path verified on the live Docker stack

## What Still Needs Attention

1. Final human listening sign-off on a few benchmark scripts before publishing
2. Optional advanced finishing / FX polish if you want deeper v1-style mastering controls

## Recommendation

Do not add broad new feature areas right now.

Focus on:

1. final human listening pass
2. any last tiny UX cleanup found during that pass
3. release when the listening results are good
