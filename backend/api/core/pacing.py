"""
Helpers for speaker delivery-rate shaping and scene pacing defaults.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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

SCENE_SPEECH_RATE_MULTIPLIERS: Dict[str, float] = {
    "relaxed": 0.90,
    "balanced": 1.00,
    "snappy": 1.08,
    "tense": 1.12,
}
BASE_TARGET_WORDS_PER_MINUTE = 168.0
PACING_GRACE_RATIO = 1.12
PACING_MAX_RATIO = 2.40
PACING_MIN_SCORE = 0.08


def read_wav_duration_seconds(audio_path: str) -> Optional[float]:
    """Return the duration of a wav file, or ``None`` when unavailable."""
    path = Path(audio_path)
    if not path.is_file():
        return None

    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return None
            return round(wav_file.getnframes() / float(frame_rate), 3)
    except (wave.Error, OSError):
        return None


def _count_words(text: Optional[str]) -> int:
    """Count spoken words in a line of dialogue."""
    return len(re.findall(r"[A-Za-z0-9']+", str(text or "")))


def _estimate_pause_seconds(text: Optional[str]) -> float:
    """Estimate pause load from punctuation cues in the script text."""
    content = str(text or "")
    ellipsis_count = content.count("...")
    normalized = content.replace("...", " ")
    return round(
        (normalized.count(",") * 0.10)
        + (normalized.count(";") * 0.18)
        + (normalized.count(":") * 0.18)
        + (normalized.count("?") * 0.20)
        + (normalized.count("!") * 0.16)
        + (ellipsis_count * 0.34)
        + (normalized.count(".") * 0.18),
        3,
    )


def estimate_target_duration_seconds(
    text: Optional[str],
    *,
    delivery_rate: float = 1.0,
    scene_pacing_profile: Optional[Any] = None,
) -> float:
    """Estimate a natural target duration for a generated line."""
    normalized_profile = normalize_scene_pacing_profile(scene_pacing_profile)
    word_count = max(1, _count_words(text))
    effective_wpm = (
        BASE_TARGET_WORDS_PER_MINUTE
        * SCENE_SPEECH_RATE_MULTIPLIERS.get(normalized_profile, 1.0)
        * clamp_delivery_rate(delivery_rate)
    )
    base_seconds = (word_count / max(effective_wpm, 1.0)) * 60.0
    pause_seconds = _estimate_pause_seconds(text)
    return round(max(0.55, base_seconds + pause_seconds), 3)


def _short_line_tolerance(word_count: int) -> float:
    """Return extra timing tolerance for short conversational lines."""
    if word_count <= 0:
        return 0.0
    return min(0.08, max(0.0, (10 - min(word_count, 10)) * 0.02))


def _score_pacing_ratio(pacing_ratio: float, word_count: int) -> float:
    """Score timing mismatch with a softer, more human-friendly curve."""
    safe_ratio = max(float(pacing_ratio), 1e-6)
    ratio_distance = abs(math.log(safe_ratio))
    grace_distance = math.log(PACING_GRACE_RATIO)
    max_distance = math.log(PACING_MAX_RATIO)
    softened_distance = max(0.0, ratio_distance - grace_distance)
    score_span = max(0.1, max_distance - grace_distance)
    raw_score = 1.0 - min(softened_distance / score_span, 1.0)
    short_line_bonus = _short_line_tolerance(word_count)
    return round(min(1.0, PACING_MIN_SCORE + (raw_score * (1.0 - PACING_MIN_SCORE)) + short_line_bonus), 3)


def assess_line_pacing(
    text: Optional[str],
    audio_path: str,
    *,
    delivery_rate: float = 1.0,
    scene_pacing_profile: Optional[Any] = None,
    quality_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Score how naturally the rendered duration matches the line's pacing cues."""
    duration_seconds = read_wav_duration_seconds(audio_path)
    word_count = _count_words(text)
    pause_seconds = _estimate_pause_seconds(text)
    target_duration_seconds = estimate_target_duration_seconds(
        text,
        delivery_rate=delivery_rate,
        scene_pacing_profile=scene_pacing_profile,
    )

    if duration_seconds is None or target_duration_seconds <= 0:
        pacing_score = 0.5
        pacing_label = "unknown"
        pacing_ratio = None
        pacing_notes = ["Rendered duration could not be measured yet."]
    else:
        pacing_ratio = duration_seconds / target_duration_seconds
        pacing_score = _score_pacing_ratio(pacing_ratio, word_count)
        pacing_notes: List[str] = []
        short_line_flex = _short_line_tolerance(word_count)
        too_fast_threshold = 0.68 - short_line_flex
        slightly_fast_threshold = 0.86 - short_line_flex
        slightly_slow_threshold = 1.20 + short_line_flex
        too_slow_threshold = 1.50 + short_line_flex

        if pacing_ratio < too_fast_threshold:
            pacing_label = "too_fast"
            pacing_notes.append("Sounds rushed for the line length.")
        elif pacing_ratio < slightly_fast_threshold:
            pacing_label = "slightly_fast"
            pacing_notes.append("Leans a little fast.")
        elif pacing_ratio > too_slow_threshold:
            pacing_label = "too_slow"
            pacing_notes.append("Drags compared with the script length.")
        elif pacing_ratio > slightly_slow_threshold:
            pacing_label = "slightly_slow"
            pacing_notes.append("Leans a little slow.")
        else:
            pacing_label = "balanced"
            pacing_notes.append("Pacing sits in the expected range.")

        if pause_seconds >= 0.25 and pacing_ratio < 0.90:
            pacing_notes.append("Punctuation suggests stronger pauses than the audio delivered.")
        if word_count <= 4 and pacing_ratio > 1.20:
            pacing_notes.append("Very short line is landing long.")

    safe_quality = None if quality_score is None else max(0.0, min(1.0, float(quality_score)))
    review_score = pacing_score if safe_quality is None else round((safe_quality * 0.8) + (pacing_score * 0.2), 3)

    return {
        "duration_seconds": duration_seconds,
        "expected_duration_seconds": target_duration_seconds,
        "pacing_ratio": None if pacing_ratio is None else round(pacing_ratio, 3),
        "pacing_score": pacing_score,
        "pacing_label": pacing_label,
        "pacing_notes": pacing_notes,
        "review_score": review_score,
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
