# Test Run Log

## Summary

- Date: 2026-04-10
- Tester: Codex
- Purpose: Verify the bounded generation queue and pacing-aware review slice after rebuild, and confirm the app comes up cleanly on the normal GPU Docker path
- App build or bundle version: `2026-04-10-queue-pacing-1`
- Runtime: Docker, frontend `http://localhost:3000`, backend `http://localhost:8001`
- Result: Pass

## Environment

- Frontend URL: `http://localhost:3000`
- Backend URL: `http://localhost:8001`
- Device mode: `cuda:0`
- DeepSpeed: `true`
- Local voices present: yes, 8 local speaker files loaded
- Script packs used: none in this focused regression pass

## Packs Run

- [x] `00-stack-startup-smoke`
- [ ] `01-speaker-prep-smoke`
- [ ] `02-conversation-workflow-smoke`
- [ ] `03-conversation-results-smoke`
- [ ] `04-timeline-editor-smoke`
- [ ] `full-release-checklist`

## Results By Area

| Area | Status | Notes |
| --- | --- | --- |
| Stack startup | Pass | Confirmed frontend and backend both healthy after cold start. Backend needed normal model-load time before the proxy stopped returning temporary `502` startup responses. |
| Speaker Prep | Not run | Not part of this focused queue/pacing regression pass. |
| Conversation Workflow | Partial pass | Verified frontend served the new bundle and backend queue/progress contracts passed. Did not run a full manual generation flow in-browser during this pass. |
| Conversation Results | Partial pass | Verified pacing-aware review code shipped in the live bundle and backend review-score contracts passed. Did not manually walk a full results selection/export flow in-browser during this pass. |
| Timeline Editor | Not run | Not part of this focused regression pass. |
| Export / playback | Not run | Not part of this focused regression pass. |
| Docs / manual | Pass | New testing docs and run-log workflow added and linked. |

## Backend Contract Tests

| Test | Result | Notes |
| --- | --- | --- |
| `tests/backend/test_generation_queue_contract.py` | Pass | Reran in the final GPU-backed backend container after restart. Confirms queued second job and bounded pending limit behavior. |
| `tests/backend/test_pacing_review_contract.py` | Pass | Reran in the final GPU-backed backend container after restart. Confirms pacing metadata and review-score behavior. |
| `tests/backend/test_selection_gating_contract.py` | Pass | Passed during the rebuilt backend verification pass before the final GPU restart. |
| `tests/backend/test_generation_progress_contract.py` | Pass | Passed during the rebuilt backend verification pass before the final GPU restart. |
| `tests/backend/test_pacing_contract.py` | Pass | Passed during the rebuilt backend verification pass before the final GPU restart. |

## Browser / Runtime Notes

1. Initial browser loads during backend model startup showed temporary `502 Bad Gateway` responses through the frontend proxy. This resolved once backend health reached `healthy`.
2. Final live checks confirmed:
   - `http://localhost:8001/health` returned `runtime_device: cuda:0`
   - `http://localhost:3000/api/health` returned healthy JSON
   - the header badge showed `API Connected - GPU: cuda:0 + DeepSpeed`
   - network requests for `/api/health`, `/api/speakers/`, `/api/speakers-tools/list-source-clips`, `/api/conversation/list`, and `/api/conversation/projects` returned `200`

## Bugs Or Regressions Found

1. No code regression found in the queue/pacing slice.
2. One verification issue was caught and corrected during the pass: the app had been restarted once on the plain compose file instead of the GPU overlay, which caused a temporary CPU runtime. Restarting with the GPU compose path restored `cuda:0 + DeepSpeed`.

## Follow-Up Actions

1. Run the full tab-by-tab smoke pack when you want a broader release-confidence pass, especially Conversation Results and Timeline Editor.
2. Use this run log as the example format for future regression or release passes.

## Sign-Off

- Release-ready for this slice: yes
- Remaining blockers: none for the queue/pacing changes themselves
- Notes for next pass: if the goal is full release confidence rather than a focused regression, run all smoke packs plus the full release checklist and save the result as a dedicated release log
