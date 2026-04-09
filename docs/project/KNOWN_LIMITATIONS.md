# Known Limitations

These are the main limitations still worth documenting before v2 replaces v1 publicly.

## Runtime

- The Docker GPU image is NVIDIA/CUDA-based today.
- GPU-first startup works well, but AMD and Apple GPU paths are not supported by this Docker image.
- The first DeepSpeed-enabled startup can take longer while extensions warm up or compile.

## Audio Quality

- Reference clip quality still heavily affects clone fidelity.
- Pacing controls are much better now, but public IndexTTS2 still does not expose fully precise duration control as a normal released feature.
- Advanced FX-style finishing is lighter than a dedicated audio editor or a fully loaded v1 mastering chain.

## Timeline

- The timeline editor is real and usable, but it is still not a full DAW.
- Selected-segment waveform preview exists, but full inline waveform lanes and trim handles are still lighter than pro audio tools.

## Workflow

- Seed export/reporting exists, but seed replay/import is still less polished than the rest of the workflow.
- During active development, frontend bundle updates can still require a hard refresh in an already-open browser tab.

## Release Position

The app is in final release-hardening rather than broad parity catch-up. The remaining work is mostly polish, documentation, and any optional advanced audio-finishing features you still want before replacing v1.
