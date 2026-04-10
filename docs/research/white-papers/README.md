# IndexTTS2 White Papers

This folder is the research workspace for deeper strategic and technical investigation around this project.

It follows the same pattern as the `Threadline` white papers flow:

- prompt packs for external deep-research runs
- a place to keep original/raw research outputs
- synthesis notes that turn research into concrete product and engineering actions

## Suggested Workflow

1. Run one research prompt at a time from `INDEXTTS2_DEEP_RESEARCH_PROMPTS.md`.
2. Save the raw result in `original/` or as a first-pass note in this folder.
3. Pull the useful findings into `INDEXTTS2_RESEARCH_SYNTHESIS_NEXT_ACTIONS.md`.
4. Turn the synthesis into a small, explicit implementation plan before touching code.

## Current Focus Areas

- model-adjacent additions that improve quality without bloating the app
- source-clip standards and speaker-prep quality heuristics
- pacing, prosody, and scene timing control
- serving/runtime throughput strategies such as queueing, batching, or slot-like scheduling
- export/mastering and post-processing strategy

## Current Coverage

Already covered with local raw research docs:

- `Voice Cloning Source Clip Quality Research.docx`
  - covers source-clip standards and speaker-prep heuristics
- `IndexTTS2 Workflow Model Integration Research.docx`
  - covers adjacent-model evaluation and integration strategy
- `IndexTTS2 Workflow Pacing Strategies.docx`
  - covers pacing, prosody, and scene-timing strategy
- `IndexTTS2 Throughput Optimization Research.docx`
  - covers throughput, queueing, batching, and slot-style scheduling

Working summaries now available:

- `VOICE_CLONING_SOURCE_CLIP_QUALITY_RESEARCH_SUMMARY.md`
- `INDEXTTS2_MODEL_INTEGRATION_RESEARCH_SUMMARY.md`
- `PACING_PROSODY_SCENE_TIMING_RESEARCH_SUMMARY.md`
- `INDEXTTS2_THROUGHPUT_OPTIMIZATION_RESEARCH_SUMMARY.md`
- `INDEXTTS2_RESEARCH_DRIVEN_ROADMAP.md`

All four planned research lanes now have working summaries.

## Files

- `INDEXTTS2_DEEP_RESEARCH_PROMPTS.md`
  - reusable prompts for external deep research
- `INDEXTTS2_RESEARCH_SYNTHESIS_NEXT_ACTIONS.md`
  - current synthesis and recommended next steps
- `original/`
  - raw source material, long-form responses, or archived research drafts
