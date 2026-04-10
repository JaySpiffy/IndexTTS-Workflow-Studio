# IndexTTS2 Research Synthesis And Next Actions

Date: 2026-04-10

This note defines the research pattern this project should use for deeper, white-paper-style investigation before larger model or architecture changes.

It is intentionally modeled after the stronger research workflow used in the `Threadline` project:

- prompt packs for targeted deep research
- raw outputs kept separately
- one synthesis note that turns findings into concrete product decisions

## Executive Summary

The right next research phase for this app is not:

- "find the coolest model on the internet"
- "add every model named in upstream acknowledgements"
- "turn the app into a giant multi-engine TTS launcher"

The right next research phase is:

1. identify which adjacent model categories actually improve user-perceived quality
2. formalize source-clip quality standards into diagnostics and prep decisions
3. improve pacing and naturalness through application-layer control
4. explore throughput improvements that fit the current architecture

All four main research lanes are now seeded with long-form raw outputs:

- `original/Voice Cloning Source Clip Quality Research.docx`
- `original/IndexTTS2 Workflow Model Integration Research.docx`
- `original/IndexTTS2 Workflow Pacing Strategies.docx`
- `original/IndexTTS2 Throughput Optimization Research.docx`

Working markdown summaries now also exist:

- `VOICE_CLONING_SOURCE_CLIP_QUALITY_RESEARCH_SUMMARY.md`
- `INDEXTTS2_MODEL_INTEGRATION_RESEARCH_SUMMARY.md`
- `PACING_PROSODY_SCENE_TIMING_RESEARCH_SUMMARY.md`
- `INDEXTTS2_THROUGHPUT_OPTIMIZATION_RESEARCH_SUMMARY.md`
- `INDEXTTS2_RESEARCH_DRIVEN_ROADMAP.md`

That means all four intended research lanes now have usable working summaries.

That means the immediate research priority shifts from "start all four lanes" to:

1. synthesize the first two into decisions
2. synthesize pacing into concrete product controls
3. synthesize throughput into a practical systems roadmap

The first engineering wave driven by those two completed research lanes is now in place:

- stronger speaker-prep diagnostics and quality gates
- `DeepFilterNet` as an optional speech-cleanup backend
- verified `WeTextProcessing` normalization contracts
- optional `CAMPPlus` speaker-similarity backend for reranking and analysis

## Current Working Beliefs

These are current working beliefs, not final conclusions:

- The biggest quality gains are still likely to come from better source-clip handling and better selection logic, not swapping out the core model.
- Supporting models around IndexTTS2 are probably higher leverage than immediately adding full alternate TTS engines.
- Pacing and scene timing should continue to be treated as an app-layer problem unless upstream exposes more direct control.
- Throughput ideas such as queueing, short-window batching, or slot-like scheduling are worth researching, but should be justified carefully against VRAM cost and implementation complexity.

## Research Priorities

### Priority 1: Source Clip Standards

Goal:
- define what a "good speaker clip" really is for this app

Why first:
- bad speaker clips poison every later stage
- better diagnostics help every user immediately

Expected deliverables:
- source-clip rubric
- diagnostics thresholds
- stronger prep recommendations
- source doc now available:
  - `original/Voice Cloning Source Clip Quality Research.docx`

### Priority 2: Model-Adjacent Additions

Goal:
- decide which supporting models are actually worth integrating

Why second:
- there are many possible additions, but only a few are likely to matter enough

Expected deliverables:
- ranked adjacent-model roadmap
- fit/complexity/licensing assessment
- source doc now available:
  - `original/IndexTTS2 Workflow Model Integration Research.docx`

### Priority 3: Pacing And Prosody

Goal:
- improve perceived naturalness without depending on unreleased upstream controls

Why third:
- users notice pacing problems quickly
- the app already has a strong place to own this problem

Expected deliverables:
- recommended pacing controls by layer
- scoring ideas for rushed/robotic output
- possible next app controls
- summary now available:
  - `PACING_PROSODY_SCENE_TIMING_RESEARCH_SUMMARY.md`

### Priority 4: Throughput

Goal:
- improve responsiveness and batch throughput without wrecking stability

Why fourth:
- speed matters, but should not outrank quality and product coherence

Expected deliverables:
- throughput architecture options
- recommended MVP
- VRAM/latency tradeoff note
- summary now available:
  - `INDEXTTS2_THROUGHPUT_OPTIMIZATION_RESEARCH_SUMMARY.md`

## Suggested Folder Usage

- save long-form research outputs in `docs/research/white-papers/original/`
- keep the best reusable prompts in `INDEXTTS2_DEEP_RESEARCH_PROMPTS.md`
- update this file when research becomes concrete enough to change roadmap decisions

## Next Practical Step

Before making bigger model or architecture changes, use the four working summaries to update the product roadmap with:

- verified facts
- recommended action
- explicit "build now / defer / avoid" decisions
