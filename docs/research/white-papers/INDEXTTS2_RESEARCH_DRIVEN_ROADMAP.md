# IndexTTS2 Research-Driven Roadmap

Date: 2026-04-10

This note turns the completed white-paper lanes into one practical roadmap for the app.

Current source inputs:

- [VOICE_CLONING_SOURCE_CLIP_QUALITY_RESEARCH_SUMMARY.md](VOICE_CLONING_SOURCE_CLIP_QUALITY_RESEARCH_SUMMARY.md)
- [INDEXTTS2_MODEL_INTEGRATION_RESEARCH_SUMMARY.md](INDEXTTS2_MODEL_INTEGRATION_RESEARCH_SUMMARY.md)
- [PACING_PROSODY_SCENE_TIMING_RESEARCH_SUMMARY.md](PACING_PROSODY_SCENE_TIMING_RESEARCH_SUMMARY.md)
- [INDEXTTS2_THROUGHPUT_OPTIMIZATION_RESEARCH_SUMMARY.md](INDEXTTS2_THROUGHPUT_OPTIMIZATION_RESEARCH_SUMMARY.md)

Open research lanes still to complete:

- none

Current implementation status against this roadmap:

- stronger speaker-prep diagnostics and clip health scoring: completed
- `DeepFilterNet` integration for source-clip cleanup: completed
- `WeTextProcessing` normalization verification and contract coverage: completed
- `CAMPPlus` similarity reranking prototype: completed as an optional backend

## Executive Summary

The current research points to a clear strategic direction:

Do not widen the app into a giant multi-engine TTS product.

Instead, make the current IndexTTS2 workflow stronger by improving:

1. what goes into the model
2. how generated candidates are filtered and selected
3. how text is normalized before inference
4. how throughput is handled around the core model
5. how the final exported scene is polished

This is a workflow-strengthening roadmap, not a "replace the core engine" roadmap.

## What The Completed Research Already Tells Us

### 1. Source quality is still the first quality gate

The source-clip research reinforces that bad reference audio still poisons the whole pipeline.

That means the app should continue moving toward:

- curated speaker library entries
- clearer speaker-health scoring
- stronger reject-vs-repair logic
- better explanation of why a source clip is weak

### 2. Supporting models matter more than alternate TTS engines

The model-integration research strongly supports:

- lightweight supporting models around IndexTTS2
- not bolting on full alternate engines casually

That is a good fit for the product as it exists now.

### 3. The best near-term additions are practical, not glamorous

The strongest candidates so far are:

- `DeepFilterNet` for source-clip cleanup
- `Wespeaker / CAM++` for reranking and anti-drift selection
- `WeTextProcessing` for text normalization
- later, a mastering layer for final export polish

These are attractive because they improve user-visible quality without turning the app into an incoherent model launcher.

## Build / Defer / Avoid

## Build Or Evaluate Seriously

### A. Stronger speaker health scoring

Why:

- already aligned with existing speaker prep
- low product risk
- helps every user immediately

Likely work:

- unified speaker-health score
- clearer reject reasons
- more explicit "repairable vs non-repairable" labeling

### B. `DeepFilterNet` evaluation

Why:

- strong fit for CPU-side source-clip cleanup
- low VRAM pressure
- directly improves reference quality

Goal:

- test whether it improves prep output enough to justify shipping

### C. `Wespeaker / CAM++` reranking prototype

Why:

- directly targets voice drift and unstable candidates
- fits current multi-version generation flow

Goal:

- over-generate a few candidates
- compare embeddings against the reference
- use that signal to rank or discard weaker outputs

### D. `WeTextProcessing` evaluation

Why:

- lightweight
- practical
- helps sanitize problematic text before inference

Goal:

- improve numbers, dates, currency, and symbol-heavy input handling without GPU-heavy dependencies

## Defer Until Product Priorities Are Chosen And Proven

### A. Throughput architecture changes

We now have a working throughput synthesis, but the actual build sequence should still be phased carefully:

- bounded queue first
- micro-batching second
- advanced slot work later

### B. Final mastering becoming a default export layer

It looks promising, but should stay in the evaluation bucket until the pacing and throughput picture is clearer.

## Avoid For Now

- `XTTSv2` as an alternate backend
- full alternate TTS engines as casual add-ons
- heavyweight preprocessing stacks that compete hard for VRAM
- brittle academic alignment tools as default user-facing workflows
- architecture changes that make the Docker runtime dramatically harder to reason about

## Recommended Implementation Sequence

## Phase 1: Harden The Existing Speaker Quality Gate

Do next:

1. strengthen the speaker-health score
2. make reject reasons more explicit
3. make neutral/stable identity clips the recommended default
4. verify existing prep heuristics against real user clips

Why first:

- lowest risk
- highest immediate user-visible value
- makes every later model evaluation more meaningful

## Phase 2: Evaluate Lightweight Support Models

Do next:

1. evaluate `DeepFilterNet`
2. evaluate `WeTextProcessing`

Why second:

- both are relatively low-churn additions
- both fit the current app well
- both improve the pipeline before or around core generation

## Phase 3: Improve Candidate Selection

Do next:

1. prototype `Wespeaker / CAM++` reranking
2. compare manual best-pick vs embedding-assisted best-pick
3. decide whether reranking belongs in the default flow

Why third:

- this directly improves trust in generated results
- it strengthens the review workflow without replacing it

## Phase 4: Turn The Research Into Product Decisions

Still needed:

1. decide which pacing improvements should become default UX
2. decide whether throughput work starts with queueing or cache expansion
3. decide which optional quality backends remain opt-in versus default

Why fourth:

- these choices will shape the next larger architectural decisions
- they should be informed by the current app, not guessed

## Phase 5: Evaluate Final Export Polish

Do later:

1. reassess mastering and spectral matching
2. decide if it should be optional or default

Why later:

- export polish matters most after input quality and generation quality are more stable

## Biggest Open Questions

These are the main unresolved questions that the product roadmap should answer next:

1. How much pacing should be solved in scripting, how much in generation, and how much in export?
2. What throughput strategy gives the best gain without destabilizing the runtime?
3. Which heuristics should become hard gates versus soft warnings in speaker prep?
4. Which adjacent model additions are good enough to ship by default, not just demo experimentally?

## Current Bottom Line

If we only used the first two completed research lanes, the best next engineering direction was:

1. strengthen speaker-prep diagnostics and clip health scoring
2. evaluate `DeepFilterNet`
3. evaluate `WeTextProcessing`
4. prototype `Wespeaker / CAM++` reranking
5. defer larger runtime changes until better pacing and throughput decisions are in place

Items 1 through 4 are now implemented or prototyped in the app.

The pacing/prosody lane is now summarized, and it points toward product-layer improvements such as:

- pacing-aware review scoring
- speaker baseline pacing profiles
- line-level pacing intent labels
- stronger scene timing presets

The throughput lane is now summarized, and it points toward systems work such as:

- bounded internal queueing
- short-window micro-batching
- stronger prompt and speaker caching
- deferring advanced slot-style execution until later

That means the most coherent path forward now is:

1. decide which pacing improvements should ship next as product defaults
2. decide whether throughput starts with queueing or cache expansion
3. decide which optional scoring and cleanup backends should remain default versus opt-in
