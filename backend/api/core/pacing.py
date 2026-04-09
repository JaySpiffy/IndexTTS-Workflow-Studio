"""
Helpers for speaker delivery-rate shaping and scene pacing defaults.
"""

from __future__ import annotations

import math
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_SCENE_PACING_PROFILE = "balanced"
DEFAULT_SCENE_GAP_MS = 140
MIN_DELIVERY_RATE = 0.85
MAX_DELIVERY_RATE = 1.15

SCENE_PACING_PRESETS: Dict[str, Dict[str, int]] = {
    "relaxed": {
        "default_gap_ms": 220,
        "statement_pause_ms": 110,
        "question_pause_ms": 150,
        "exclamation_pause_ms": 95,
        "ellipsis_pause_ms": 220,
    },
    "balanced": {
        "default_gap_ms": 140,
        "statement_pause_ms": 75,
        "question_pause_ms": 110,
        "exclamation_pause_ms": 70,
        "ellipsis_pause_ms": 170,
    },
    "snappy": {
        "default_gap_ms": 75,
        "statement_pause_ms": 35,
        "question_pause_ms": 55,
        "exclamation_pause_ms": 30,
        "ellipsis_pause_ms": 90,
    },
    "tense": {
        "default_gap_ms": 45,
        "statement_pause_ms": 20,
        "question_pause_ms": 40,
        "exclamation_pause_ms": 20,
        "ellipsis_pause_ms": 70,
    },
}


def clamp_delivery_rate(value: Optional[Any]) -> float:
    """Clamp arbitrary delivery-rate input to the supported safe range."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 1.0
    return max(MIN_DELIVERY_RATE, min(MAX_DELIVERY_RATE, numeric))


def should_apply_delivery_rate(delivery_rate: Optional[Any]) -> bool:
    """Return whether the requested rate differs meaningfully from neutral."""
    return not math.isclose(clamp_delivery_rate(delivery_rate), 1.0, abs_tol=0.005)


def normalize_scene_pacing_profile(profile: Optional[Any]) -> str:
    """Return a supported scene pacing preset name."""
    candidate = str(profile or DEFAULT_SCENE_PACING_PROFILE).strip().lower()
    return candidate if candidate in SCENE_PACING_PRESETS else DEFAULT_SCENE_PACING_PROFILE


def resolve_scene_pacing(profile: Optional[Any]) -> Dict[str, int]:
    """Resolve a scene pacing preset into its pause defaults."""
    normalized = normalize_scene_pacing_profile(profile)
    return {"profile": normalized, **SCENE_PACING_PRESETS[normalized]}


def build_speaker_pacing_map(speaker_pacing: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, float]:
    """Build a filename/stem lookup for per-speaker delivery-rate overrides."""
    pacing_map: Dict[str, float] = {}
    for entry in speaker_pacing or []:
        if not isinstance(entry, dict):
            continue
        speaker_filename = str(entry.get("speaker_filename") or entry.get("speaker") or "").strip()
        if not speaker_filename:
            continue
        delivery_rate = clamp_delivery_rate(entry.get("delivery_rate", 1.0))
        pacing_map[speaker_filename.lower()] = delivery_rate
        pacing_map[Path(speaker_filename).stem.lower()] = delivery_rate
    return pacing_map


def resolve_speaker_delivery_rate(
    speaker_filename: Optional[str],
    speaker_pacing_map: Optional[Dict[str, float]],
) -> float:
    """Resolve the effective delivery rate for a specific speaker file."""
    if not speaker_filename or not speaker_pacing_map:
        return 1.0

    normalized_filename = str(speaker_filename).strip().lower()
    stem = Path(normalized_filename).stem.lower()
    return clamp_delivery_rate(
        speaker_pacing_map.get(normalized_filename, speaker_pacing_map.get(stem, 1.0))
    )


def _build_atempo_filter(delivery_rate: float) -> str:
    """Build an ffmpeg atempo filter chain for the requested rate."""
    remaining = clamp_delivery_rate(delivery_rate)
    factors = []

    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0

    factors.append(remaining)
    return ",".join(f"atempo={factor:.5f}" for factor in factors)


def apply_delivery_rate_to_file(audio_path: str, delivery_rate: float) -> Dict[str, Any]:
    """Apply a subtle tempo change in-place while preserving pitch."""
    clamped_rate = clamp_delivery_rate(delivery_rate)
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found for pacing: {audio_path}")

    if not should_apply_delivery_rate(clamped_rate):
        return {
            "applied": False,
            "delivery_rate": 1.0,
            "audio_path": str(source),
            "reason": "neutral_rate",
        }

    temp_output = source.with_name(f"{source.stem}.paced-{uuid.uuid4().hex[:8]}{source.suffix}")
    filter_chain = _build_atempo_filter(clamped_rate)
    env = os.environ.copy()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-filter:a",
                filter_chain,
                str(temp_output),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        temp_output.replace(source)
    except subprocess.CalledProcessError as error:
        if temp_output.exists():
            temp_output.unlink(missing_ok=True)
        raise RuntimeError(error.stderr.strip() or str(error)) from error

    return {
        "applied": True,
        "delivery_rate": clamped_rate,
        "audio_path": str(source),
        "filter": filter_chain,
    }
