# IndexTTS2 Deep Research Prompts

This file contains a focused set of deep-research prompts for the next phase of this project.

Use them the same way as the Threadline white-paper prompts:

- one prompt at a time
- separate verified facts, inference, and recommendations clearly
- prefer official docs, upstream repos, papers, issues, and model cards first
- avoid generic "best TTS models" content unless it is tied back to this app's actual architecture
- keep recommendations grounded in the current product, which is already a Docker-first local IndexTTS2 workflow app

Current project context:

- this app is built around the official IndexTTS2 models
- it already has:
  - speaker prep
  - multi-speaker conversation workflow
  - review/regeneration
  - timeline editing
  - pacing controls
  - export finishing
  - Docker-first GPU runtime
- it is strongest as a workflow studio around one core model stack
- it is weaker at:
  - proving which adjacent models are actually worth integrating
  - formalizing source-clip quality standards
  - improving natural pacing and prosody without model retraining
  - increasing throughput without turning the runtime into a mess

Important current strategic question:

- "what research-backed additions would make this app materially better without collapsing its coherence?"

Important non-goals:

- do not assume the answer is "support every TTS model"
- do not optimize for benchmark novelty over user-meaningful quality
- do not recommend major architecture churn unless the benefit is clear

---

## Prompt 1: Model-Adjacent Additions That Actually Improve The App

```text
You are conducting deep product and technical research for a Docker-first local application built around the official IndexTTS2 models.

Treat this prompt as fully standalone. Assume you have zero prior context beyond what is written below.

Working rules:
- Separate verified facts, inference, and recommendation clearly.
- Prefer official model docs, official repos, papers, issues, and maintainer guidance first.
- Do not drift into generic "best TTS model" lists.
- Optimize for practical additions to a real product, not academic novelty.

Public repo to inspect:
- https://github.com/JaySpiffy/IndexTTS-Workflow-Studio

Useful public entry points:
- README: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/blob/main/README.md
- user manual: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/blob/main/docs/manual/USER_MANUAL.md
- research docs: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/tree/main/docs/research

Project context:
- This app is a Docker-first local workflow studio built on top of the official IndexTTS2 model ecosystem.
- It is not trying to be the official upstream model repo and it is not trying to be a giant "all TTS models" launcher.
- Its core product flow is:
  - Speaker Prep
  - Conversation Workflow
  - Conversation Results
  - Timeline Editor
- It already includes:
  - speaker prep for trimming, mono conversion, normalization, cleanup, and diagnostics
  - multi-speaker conversation generation
  - line-by-line review and regeneration
  - project save/load
  - pacing controls and presets
  - timeline editing with overlaps and export
  - Docker-first GPU runtime with DeepSpeed support
- It is strongest when it stays coherent around one main workflow.
- Current weaknesses and open strategic questions include:
  - proving which adjacent models are worth integrating
  - improving voice fidelity and naturalness further
  - making source-clip quality standards more rigorous
  - increasing throughput without creating an unmaintainable runtime
- Non-goals:
  - do not assume the answer is "support every TTS model"
  - do not optimize for benchmark novelty over user-perceived workflow quality
  - do not recommend major architecture churn unless the benefit is clear

Research task:
Determine which additional model types would most improve the app if integrated around IndexTTS2.

Research questions:
1. Which model categories are the highest-value complements to an IndexTTS2-based workflow app?
2. Which of these are likely to improve user-perceived quality the most?
3. Which of these are likely to improve speed, robustness, or automation the most?
4. Which model additions would create too much architectural or licensing complexity for too little gain?
5. What is the best "adjacent model roadmap" in order of impact vs complexity?

Model categories to evaluate:
- source-clip speech enhancement / denoise / dereverb
- speaker similarity embedding and reranking
- transcript alignment / pronunciation / text normalization helpers
- post-processing/mastering enhancement
- alternate full TTS engines such as XTTSv2 or Tortoise as optional backends
- serving/runtime acceleration ideas such as batching or slot-based scheduling

Output requirements:
- executive summary
- verified facts vs inference
- evaluation matrix with columns:
  - model category
  - likely user-visible benefit
  - implementation complexity
  - runtime cost
  - licensing risk
  - fit with this app
  - recommendation
- top 3 additions to actively pursue
- top 3 additions to avoid for now
- final roadmap
```

---

## Prompt 2: Source Clip Quality Standards And Speaker Prep Research

```text
You are conducting deep technical and workflow research for an IndexTTS2-based local voice workflow application.

Treat this prompt as fully standalone. Assume you have zero prior context beyond what is written below.

Working rules:
- Separate verified facts, inference, and recommendations clearly.
- Prefer official docs, papers, maintainer issues, and primary sources first.
- Focus on standards that could be turned into application diagnostics and UI guidance.

Public repo to inspect:
- https://github.com/JaySpiffy/IndexTTS-Workflow-Studio

Useful public entry points:
- README: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/blob/main/README.md
- user manual: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/blob/main/docs/manual/USER_MANUAL.md
- speaker-prep-adjacent research: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/tree/main/docs/research

Project context:
- This app is a Docker-first local workflow studio built around the official IndexTTS2 models.
- One of its major user-facing tabs is `Speaker Prep`.
- The app already has a speaker-prep workflow with:
  - trim
  - mono conversion
  - normalization
  - noise cleanup
  - quick diagnostics
- The goal is to make speaker prep smarter, more trustworthy, and more automatic.
- The app needs clearer standards for what makes a source clip good or bad for cloning.
- The result of this research should be practical enough to turn into:
  - diagnostics scores
  - warnings and recommendations
  - auto-prep defaults
  - acceptance/rejection thresholds for the speaker library
- Non-goals:
  - do not assume a perfect studio recording is always required
  - do not give vague generic "use clean audio" advice without operationalizing it
  - do not optimize for academic purity over real app heuristics

Research task:
Find the most reliable research-backed standards for source-clip quality in voice cloning workflows, especially around IndexTTS2-like systems.

Research questions:
1. What characteristics most strongly correlate with good voice-cloning results?
2. What clip duration ranges are generally best?
3. How important are noise, reverb, compression artifacts, background music, and multiple speakers?
4. What should an app score or flag in a source clip before allowing it into the speaker library?
5. Which diagnostics can realistically be automated?

Output requirements:
- executive summary
- verified facts vs inference
- "good source clip" checklist
- "bad source clip" failure checklist
- suggested app diagnostics matrix
- suggested prep defaults
- recommended acceptance thresholds where possible
```

---

## Prompt 3: Pacing, Prosody, And Scene Timing Without Retraining

```text
You are researching how a workflow app built around IndexTTS2 can improve perceived pacing and naturalness without training a new foundation model.

Treat this prompt as fully standalone. Assume you have zero prior context beyond what is written below.

Working rules:
- Separate verified facts, inference, and recommendations clearly.
- Prefer official docs, papers, issues, and primary project sources.
- Focus on application-layer controls, not hand-wavy "the model should just be better."

Public repo to inspect:
- https://github.com/JaySpiffy/IndexTTS-Workflow-Studio

Useful public entry points:
- README: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/blob/main/README.md
- user manual: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/blob/main/docs/manual/USER_MANUAL.md
- scripting playbook: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/blob/main/docs/research/INDEXTTS2_SCRIPTING_PLAYBOOK.md
- voice fidelity note: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/blob/main/docs/research/INDEXTTS2_VOICE_FIDELITY_RESEARCH_SYNTHESIS_NEXT_ACTIONS.md

Project context:
- This app is a Docker-first local workflow studio built around IndexTTS2.
- It already supports:
  - punctuation-aware scripting guidance
  - speaker pacing controls
  - scene pacing presets
  - timeline editing
  - overlap control
  - ducking
  - export-time timing controls
- Users still notice when pacing feels unnatural, too rushed, too robotic, or not emotionally plausible.
- The public IndexTTS2 release does not expose all academic timing controls directly, so the app likely needs to own more of this at the application layer.
- The result of this research should be practical enough to influence:
  - script guidance
  - generation presets
  - post-generation shaping
  - timeline behavior
  - automatic scoring and selection
- Non-goals:
  - do not default to "train a new model"
  - do not recommend controls that cannot realistically fit a local desktop Docker workflow
  - do not ignore the existing app-layer controls already present in the product

Research task:
Determine the best application-layer strategies for improving pacing, prosody, and scene timing around an IndexTTS2 workflow app.

Research questions:
1. Which controls are most likely to improve perceived naturalness?
2. What should be handled at script time vs generation time vs export time?
3. What kinds of pacing presets are most useful?
4. When should an app alter timing after generation rather than trying to force the model?
5. What evaluation signals can help detect rushed or robotic delivery automatically?

Output requirements:
- executive summary
- verified facts vs inference
- pacing control matrix by layer:
  - scripting
  - generation
  - post-generation
  - timeline/export
- recommended product controls
- recommended automatic scoring signals
- final implementation priorities
```

---

## Prompt 4: Throughput Research For A Local IndexTTS2 Workflow Studio

```text
You are conducting systems research for a local Docker-first IndexTTS2 workflow application.

Treat this prompt as fully standalone. Assume you have zero prior context beyond what is written below.

Working rules:
- Separate verified facts, inference, and recommendations clearly.
- Prefer upstream docs, implementation notes, papers, and inference-system references first.
- Focus on speedups that would fit a real production app, not only lab benchmarks.

Public repo to inspect:
- https://github.com/JaySpiffy/IndexTTS-Workflow-Studio

Useful public entry points:
- README: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/blob/main/README.md
- manual: https://github.com/JaySpiffy/IndexTTS-Workflow-Studio/blob/main/docs/manual/USER_MANUAL.md

Project context:
- This app is a Docker-first local workflow studio built around IndexTTS2.
- It currently uses one main model stack per backend runtime.
- It already supports:
  - GPU-first runtime
  - CPU fallback
  - DeepSpeed
  - CUDA kernels where available
  - pacing controls and timeline export
- It is not currently designed as a giant distributed serving platform.
- The team is considering whether the app could benefit from:
  - queueing
  - short-window batching
  - LM Studio-style parallel-slot ideas
  - one-backend-per-GPU strategies
- The goal is higher throughput and better responsiveness, not reckless VRAM bloat or a fragile architecture.
- The result of this research should be practical enough to turn into an MVP systems roadmap.
- Non-goals:
  - do not assume "multi-GPU" means easy shared-VRAM model parallelism
  - do not propose heavyweight infra that undermines the local Docker-first product
  - do not prioritize benchmark throughput if it worsens user-visible quality or stability

Research task:
Identify the best realistic throughput strategies for this app over the next phase.

Research questions:
1. What throughput patterns from LLM serving do and do not translate well to TTS inference?
2. Would queueing, short-window batching, or slot scheduling likely help this model family?
3. Which stages of the TTS pipeline are the best candidates for batching?
4. What are the VRAM and latency tradeoffs?
5. What is the best MVP strategy for improving throughput without destabilizing quality or maintainability?

Output requirements:
- executive summary
- verified facts vs inference
- architecture options table
- top 3 viable throughput strategies
- top 3 strategies to avoid for now
- phased implementation roadmap
```
