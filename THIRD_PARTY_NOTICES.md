# Third-Party Notices

Draft to Take beta is an application layer around third-party local AI models and runtime dependencies. This file is not legal advice. Re-check official upstream sources before relying on model outputs for public or commercial use.

## IndexTTS2

Draft to Take uses the official IndexTTS2 model/runtime family for dialogue generation.

Official sources:

- Code: https://github.com/index-tts/index-tts
- Model: https://huggingface.co/IndexTeam/IndexTTS-2
- Model license: https://huggingface.co/IndexTeam/IndexTTS-2/blob/main/LICENSE.txt

Draft to Take does not bundle IndexTTS2 model weights in this release repo. The app may download model files into your local shared model folder on first run.

## Qwen / Script LLM

The beta can run an optional managed local Qwen GGUF sidecar for emotion detection and AI Thread workflows.

Default model source:

- https://huggingface.co/ufoym/Qwen3-8B-Q4_K_M-GGUF

## OmniVoice

The beta can run an optional OmniVoice sidecar for reusable voice design.

Official source:

- https://huggingface.co/k2-fsa/OmniVoice
- https://github.com/k2-fsa/OmniVoice

Users remain responsible for lawful, authorized use of generated or cloned voices.

## SFX And Music Models

The optional SFX/music sidecar is disabled by default in the beta launcher.

Possible model sources:

- Woosh sound effects: https://github.com/SonyResearch/Woosh
- Woosh checkpoint mirror used by the sidecar default: https://huggingface.co/AEmotionStudio/woosh-models
- MusicGen music beds: https://huggingface.co/facebook/musicgen-small

Woosh and MusicGen model-backed generation should be treated as free/experimental and license-dependent unless the active model/backend terms have been checked and are clearly suitable for your intended use.

Draft to Take Pro should not be treated as granting extra rights to third-party models or their outputs.

## User Data

Do not upload private scripts, speaker samples, source clips, generated private scenes, tokens, or model weights to public issue reports.
