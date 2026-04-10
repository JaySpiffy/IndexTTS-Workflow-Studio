# Test Run Log Template

Copy this file into `docs/testing/runs/` when you want to record a smoke pass, release pass, or regression check.

Suggested filename:
- `YYYY-MM-DD-smoke.md`
- `YYYY-MM-DD-release.md`
- `YYYY-MM-DD-regression-<short-topic>.md`

---

# Test Run Log

## Summary

- Date:
- Tester:
- Purpose:
- App build or bundle version:
- Runtime:
- Result:

## Environment

- Frontend URL:
- Backend URL:
- Device mode:
- DeepSpeed:
- Local voices present:
- Script packs used:

## Packs Run

- [ ] `00-stack-startup-smoke`
- [ ] `01-speaker-prep-smoke`
- [ ] `02-conversation-workflow-smoke`
- [ ] `03-conversation-results-smoke`
- [ ] `04-timeline-editor-smoke`
- [ ] `full-release-checklist`

## Results By Area

| Area | Status | Notes |
| --- | --- | --- |
| Stack startup | | |
| Speaker Prep | | |
| Conversation Workflow | | |
| Conversation Results | | |
| Timeline Editor | | |
| Export / playback | | |
| Docs / manual | | |

## Backend Contract Tests

List the focused tests you ran and whether they passed.

| Test | Result | Notes |
| --- | --- | --- |
| `tests/backend/...` | | |

## Bugs Or Regressions Found

1. 

## Follow-Up Actions

1. 

## Sign-Off

- Release-ready:
- Remaining blockers:
- Notes for next pass:
