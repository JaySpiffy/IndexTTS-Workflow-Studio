"""
Quality benchmark helper for comparing generation presets against the live API.

Usage inside the backend container:

    python backend/scripts/quality_benchmark.py --speaker Pr.D.Trump.wav --text "Hello there."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEAKERS_ROOT = PROJECT_ROOT / "shared" / "audio" / "speakers"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import backend.api.core.audio_processing as audio_processing  # noqa: E402


PRESETS: Dict[str, Dict[str, Any]] = {
    "clone_fidelity": {
        "emotion_weight": 0.7,
        "use_random_sampling": False,
        "max_text_tokens_per_segment": 180,
        "do_sample": False,
        "top_p": 0.75,
        "top_k": 20,
        "temperature": 0.7,
        "length_penalty": 0.0,
        "num_beams": 5,
        "repetition_penalty": 7.0,
        "max_mel_tokens": 2000,
    },
    "balanced": {
        "emotion_weight": 1.0,
        "use_random_sampling": False,
        "max_text_tokens_per_segment": 120,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 0.8,
        "length_penalty": 0.0,
        "num_beams": 3,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 1500,
    },
    "expressive": {
        "emotion_weight": 1.0,
        "use_random_sampling": False,
        "max_text_tokens_per_segment": 120,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 40,
        "temperature": 0.9,
        "length_penalty": 0.0,
        "num_beams": 3,
        "repetition_penalty": 9.0,
        "max_mel_tokens": 1700,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark IndexTTS generation presets via the live API")
    parser.add_argument("--speaker", required=True, help="Speaker filename from /app/speakers, for example Pr.D.Trump.wav")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/api/conversation", help="Base API URL inside the backend container")
    parser.add_argument("--preset", action="append", choices=sorted(PRESETS.keys()), help="Specific preset(s) to test. Defaults to all presets.")
    parser.add_argument("--output", help="Optional path to write JSON results")
    return parser.parse_args()


def ensure_api_ready(api_base: str) -> Dict[str, Any]:
    health_url = api_base.replace("/api/conversation", "/health")
    response = requests.get(health_url, timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("model_loaded"):
        raise RuntimeError(f"Model is not ready yet: {payload}")
    return payload


def generate_variant(api_base: str, speaker_filename: str, text: str, preset_name: str) -> Dict[str, Any]:
    payload = {
        "speaker_filename": speaker_filename,
        "text": text,
        "emotion_control_method": "from_speaker",
        "emotion_vectors": [],
        "emotion_text": None,
        **PRESETS[preset_name],
    }
    response = requests.post(f"{api_base}/generate-single", json=payload, timeout=900)
    response.raise_for_status()
    return response.json()


def main() -> int:
    args = parse_args()

    health = ensure_api_ready(args.api_base)
    if not audio_processing.initialize_speaker_model():
        raise RuntimeError("Speaker similarity model could not be initialized")

    speaker_path = SPEAKERS_ROOT / args.speaker
    if not speaker_path.exists():
        raise FileNotFoundError(f"Speaker file not found: {speaker_path}")

    preset_names = args.preset or list(PRESETS.keys())
    results = {
        "speaker": args.speaker,
        "text": args.text,
        "health": health,
        "presets": [],
    }

    for preset_name in preset_names:
        api_result = generate_variant(args.api_base, args.speaker, args.text, preset_name)
        audio_path = PROJECT_ROOT / api_result["audio_path"]
        quality = audio_processing.analyze_speaker_similarity_with_quality(
            audio_processing.speaker_similarity_model,
            str(speaker_path),
            str(audio_path),
        )
        results["presets"].append(
            {
                "preset": preset_name,
                "settings": PRESETS[preset_name],
                "audio_path": str(audio_path),
                "similarity": quality["similarity"],
                "robotic_score": quality["robotic_score"],
                "quality_score": quality["quality_score"],
            }
        )

    results["presets"].sort(key=lambda item: item["quality_score"], reverse=True)

    output_text = json.dumps(results, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"Wrote benchmark results to {output_path}")
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
