# Pacing, Prosody, And Scene Timing Research Summary

Date: 2026-04-10

This note summarizes the pacing and prosody research lane for the IndexTTS2 workflow studio.

It is intended to answer one practical product question:

How should this app improve perceived pacing and naturalness without waiting for a new foundation model or unreleased upstream controls?

## Source Inputs

Imported raw research output:

- `original/IndexTTS2 Workflow Pacing Strategies.docx`

Primary sources:

- Official IndexTTS repository README:
  - `https://github.com/index-tts/index-tts`
- Official IndexTTS public README raw:
  - `https://raw.githubusercontent.com/index-tts/index-tts/main/README.md`
- W3C Speech Synthesis Markup Language 1.1 draft:
  - `https://www.w3.org/TR/2007/WD-speech-synthesis11-20070611/`
- IndexTTS2 paper landing page:
  - `https://www.isca-archive.org/ssw_2025/suni25_ssw.html`

Local app context:

- `../INDEXTTS2_SCRIPTING_PLAYBOOK.md`
- `../INDEXTTS2_VOICE_FIDELITY_RESEARCH_SYNTHESIS_NEXT_ACTIONS.md`
- `../../../backend/api/core/pacing.py`
- `../../../backend/api/core/audio_mixing.py`
- `../../../frontend/src/modules/conversationWorkflow.js`

## Executive Summary

The most coherent way to improve pacing in this app is to treat pacing as a layered workflow problem, not a single model knob problem.

The strongest pacing gains for the public IndexTTS2 release are likely to come from:

1. better script structure
2. better generation defaults and intent labeling
3. subtle post-generation tempo shaping
4. stronger timeline and export-time scene control
5. automatic pacing-aware scoring during review

Current evidence does not support waiting for a public "true duration control" feature before improving naturalness. The app already owns the right seams:

- scripting guidance
- generation presets
- per-speaker delivery shaping
- scene pacing presets
- timeline placement and overlap
- export-time pause shaping

That means the next pacing work should stay focused on workflow intelligence rather than large model churn.

## Verified Facts

### 1. The current app already owns several meaningful pacing layers

Verified from local code:

- `backend/api/core/pacing.py` already supports safe delivery-rate shaping and scene pacing presets.
- `backend/api/core/audio_mixing.py` already applies punctuation-aware pause logic and scene gaps at export time.
- `frontend/src/modules/conversationWorkflow.js` already exposes pacing presets and scene pacing controls to the user workflow.

This matters because the product is not starting from zero. Pacing is already partly an app-level feature.

### 2. The public IndexTTS release does not expose a simple, production-ready duration-control workflow for this app today

Verified from the public repo and prior product research:

- the public release emphasizes emotion control, reference handling, sampling behavior, and deployment/runtime setup
- public-facing documentation does not present a clean end-user timing-control surface that this app can simply forward directly

This means natural pacing should continue to be treated as an application-layer responsibility unless upstream exposes more direct public controls later.

### 3. SSML-style pacing controls are established application-layer patterns

Verified from the W3C SSML specification:

- pacing can be influenced through explicit break insertion
- speaking rate can be altered through prosody controls

This does not mean the app needs to become a full SSML editor. It does mean that app-layer timing control is a legitimate design choice, not a hack.

### 4. Punctuation and segmentation remain meaningful control surfaces

Verified from the app's own scripting playbook and existing implementation:

- punctuation already influences pause behavior
- line segmentation already affects how dialogue is generated and exported

This means script guidance is still one of the most cost-effective pacing levers available.

## Inference

These conclusions are informed by the sources above, but are still product inferences rather than direct upstream claims.

### 1. Scene timing should mostly be owned after generation, not forced inside the model

Best fit:

- generation should determine the voice and phrase shape
- export and timeline layers should determine scene rhythm, overlap, recovery space, and interruption timing

Why:

- post-generation timing is easier to reason about
- it keeps the Docker runtime simpler
- it avoids overfitting the workflow to unstable model internals

### 2. Post-generation tempo shaping is useful only when kept subtle

Large time-stretching tends to sound synthetic. Small shaping is much safer.

Recommended default mindset:

- subtle correction for "slightly rushed" or "slightly dragging"
- not aggressive retiming as a primary creative tool

### 3. Pacing quality should become part of candidate scoring, not only human listening

Right now the product is strong at generating, reviewing, and manually selecting.

The next gain is helping users spot:

- rushed lines
- unnaturally flat lines
- scene gaps that feel too tight or too loose
- line-to-line pacing drift for the same speaker

## Pacing Control Matrix By Layer

## Scripting

Best responsibilities:

- keep lines short and speakable
- keep one clear thought per line
- use punctuation as a soft cue
- split interruptions into separate lines
- use explicit overlap/timeline planning for real crosstalk

Best product guidance:

- script helper guidance for calm, natural, argument, and panic scenes
- warnings for run-on lines
- warnings for punctuation abuse such as repeated exclamation marks
- split suggestions for overly dense lines

## Generation

Best responsibilities:

- choose the right pacing preset
- choose the right emotion intensity
- choose fidelity-vs-expression tradeoffs carefully
- keep segment sizes reasonable

Best product controls:

- scene pacing preset
- per-speaker pacing baseline
- per-line delivery intent labels such as `neutral`, `calm`, `urgent`, `hesitant`, `interrupting`
- careful default guidance around `emo_alpha`, randomness, and segment length

What generation should not do alone:

- force exact scene timing
- solve all interruption behavior
- act as the only fix for scene rhythm

## Post-Generation

Best responsibilities:

- subtle rate correction
- silence trimming
- consistency shaping for a known speaker

Best product controls:

- safe speaker delivery-rate shaping
- "slow down slightly" and "tighten slightly" actions
- per-speaker baseline presets

Guardrails:

- keep time-stretching within a conservative range by default
- favor regenerate-and-select over extreme retiming

## Timeline / Export

Best responsibilities:

- final scene rhythm
- explicit gap control
- interruption timing
- overlap placement
- ducking and intelligibility

Best product controls:

- scene gap presets
- punctuation-aware pauses
- overlap entry timing
- ducking strength
- visual spacing tools in the timeline

This layer is the correct place to make a scene feel conversational instead of merely readable.

## Recommended Product Controls

The strongest next pacing controls for the product are:

1. speaker pacing baselines
   - learn or store each speaker's preferred default delivery rate
2. line-level pacing intent
   - lightweight labels such as `calm`, `natural`, `tense`, `interrupt`, `hesitant`
3. automatic line split suggestions
   - flag lines that are too dense to sound natural
4. pacing-aware review hints
   - warn when a candidate is probably rushed or unnaturally flat
5. stronger scene presets
   - presets that coordinate script guidance, generation defaults, and export timing together
6. subtle local timeline retiming
   - only for small adjustments, not as a substitute for good generation

## Recommended Automatic Scoring Signals

These are the most promising signals for detecting rushed or robotic delivery automatically.

### Strong first-wave signals

- words per second
- characters per second
- pause ratio relative to line length
- pause mismatch relative to punctuation
- line duration drift for the same speaker under the same preset

### Strong second-wave signals

- energy variability across the line
- pitch contour variability
- abrupt ending detection
- syllable-rate overload
- articulation loss under high compression
- overlap intelligibility heuristics in timeline scenes

### Product interpretation layer

These raw signals should roll up into human-facing hints such as:

- `sounds rushed`
- `too flat`
- `pause pattern feels unnatural`
- `consider splitting this line`
- `scene gap is likely too tight`

## Final Implementation Priorities

The most coherent pacing roadmap now is:

1. add pacing-aware scoring to review and selection
2. add speaker baseline pacing profiles
3. add line-level pacing intent labels
4. add automatic split suggestions for dense lines
5. improve timeline scene tools for markers, snapping, and pause visualization

## Build / Defer / Avoid

## Build Next

- pacing-aware candidate scoring
- speaker pacing baselines
- better scene presets
- automatic split suggestions

## Defer

- deeper runtime-level timing intervention until throughput research is complete
- advanced learned post-processing chains that try to "fix" bad pacing after the fact

## Avoid For Now

- treating aggressive time-stretching as the main pacing solution
- waiting on unreleased upstream duration controls before improving the app
- solving scene rhythm purely with generation-time model knobs

## Bottom Line

The public IndexTTS2 workflow does not need a new core model to sound more natural.

The next pacing gains should come from:

- smarter scripts
- smarter presets
- better review scoring
- better scene timing tools

This is good news for the product because all of those fit the current Docker-first workflow studio cleanly.
