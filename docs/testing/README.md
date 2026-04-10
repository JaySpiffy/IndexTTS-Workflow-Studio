# Testing Guide

This folder holds repeatable manual test packs for the live app.

Use these documents when:
- checking a new feature before calling it done
- running a smoke test after a Docker rebuild
- doing a release-readiness pass
- reproducing a bug with shared steps and expected results

## Structure

- `smoke/`
  - short, reusable checks for each major surface of the app
- `release/`
  - broader end-to-end verification packs used before publishing

## Recommended Order

1. Run [smoke/00-stack-startup-smoke.md](smoke/00-stack-startup-smoke.md)
2. Run the tab-specific smoke docs you changed
3. Run [release/full-release-checklist.md](release/full-release-checklist.md) before a release

## Recording A Run

If you want a written record of a smoke pass or release pass:

1. Copy [test-run-log-template.md](test-run-log-template.md)
2. Save the copy into [runs/](runs/)
3. Fill in what passed, what failed, and what still needs follow-up

## Test Data

These smoke packs assume you have your own local voices in `shared/audio/speakers`.

Recommended script packs:
- [../../test_scripts/dark_garden_parody.md](../../test_scripts/dark_garden_parody.md)
- [../../test_scripts/absurd_garden_parody.md](../../test_scripts/absurd_garden_parody.md)
- [../../test_scripts/flower_crisis_parody_timeline_pack.md](../../test_scripts/flower_crisis_parody_timeline_pack.md)

## Evidence To Capture

When a test fails, record:
- the exact step number
- a screenshot of the page state
- any toast or modal error text
- browser console errors if relevant
- backend logs if the issue looks server-side

## Release Notes

These test packs are manual by design. They complement, not replace, the focused backend contract tests in `tests/backend/`.
