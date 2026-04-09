"""
Helpers for export-time conversation mixing with optional overlap plans.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from .pacing import (
    DEFAULT_SCENE_GAP_MS,
    DEFAULT_SCENE_PACING_PROFILE,
    normalize_scene_pacing_profile,
    resolve_scene_pacing,
)


ALLOWED_START_MODES = {
    "after_previous",
    "overlap_previous",
    "simultaneous",
    "hard_cut_previous",
}
DEFAULT_TARGET_LEVEL_DBFS = -19.0
DEFAULT_PEAK_LIMIT_DBFS = -1.0
MAX_LEVELING_BOOST_DB = 12.0
MAX_LEVELING_CUT_DB = 18.0
DEFAULT_EXPORT_BITRATE_KBPS = 192


def _load_audio_segment(audio_file: str) -> AudioSegment:
    with open(audio_file, "rb") as audio_handle:
        return AudioSegment.from_file(audio_handle, format="wav")


def _resolve_output_format(output_path: str, output_format: Optional[str] = None) -> str:
    if output_format:
        return str(output_format).strip().lower()

    suffix = Path(output_path).suffix.lower().lstrip(".")
    return suffix or "wav"


def _export_segment(
    audio: AudioSegment,
    output_path: str,
    output_format: Optional[str] = None,
    output_bitrate_kbps: Optional[int] = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_format = _resolve_output_format(output_path, output_format)
    export_kwargs: Dict[str, Any] = {}
    if output_bitrate_kbps and resolved_format == "mp3":
        export_kwargs["bitrate"] = f"{int(output_bitrate_kbps)}k"

    exported_file = audio.export(str(output), format=resolved_format, **export_kwargs)
    close_export = getattr(exported_file, "close", None)
    if callable(close_export):
        close_export()


def _safe_dbfs(value: float) -> Optional[float]:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _match_segment_loudness(
    segment: AudioSegment,
    target_level_dbfs: float = DEFAULT_TARGET_LEVEL_DBFS,
    peak_limit_dbfs: float = DEFAULT_PEAK_LIMIT_DBFS,
) -> Tuple[AudioSegment, Dict[str, Any]]:
    current_level = _safe_dbfs(segment.dBFS)
    current_peak = _safe_dbfs(segment.max_dBFS)
    metadata: Dict[str, Any] = {
        "source_level_dbfs": current_level,
        "source_peak_dbfs": current_peak,
        "applied_gain_db": 0.0,
        "target_level_dbfs": float(target_level_dbfs),
        "peak_limit_dbfs": float(peak_limit_dbfs),
        "skipped": False,
        "reason": None,
    }

    if current_level is None or current_peak is None:
        metadata["skipped"] = True
        metadata["reason"] = "silent_or_invalid"
        return segment, metadata

    desired_gain = float(target_level_dbfs) - current_level
    desired_gain = max(-MAX_LEVELING_CUT_DB, min(MAX_LEVELING_BOOST_DB, desired_gain))

    allowed_boost = float(peak_limit_dbfs) - current_peak
    applied_gain = min(desired_gain, allowed_boost)

    if math.isclose(applied_gain, 0.0, abs_tol=0.05):
        adjusted = segment
        applied_gain = 0.0
    else:
        adjusted = segment.apply_gain(applied_gain)

    metadata["applied_gain_db"] = round(applied_gain, 3)
    metadata["result_level_dbfs"] = _safe_dbfs(adjusted.dBFS)
    metadata["result_peak_dbfs"] = _safe_dbfs(adjusted.max_dBFS)
    return adjusted, metadata


def _apply_final_peak_protection(
    audio: AudioSegment,
    peak_limit_dbfs: float = DEFAULT_PEAK_LIMIT_DBFS,
) -> Tuple[AudioSegment, Dict[str, Any]]:
    current_peak = _safe_dbfs(audio.max_dBFS)
    metadata: Dict[str, Any] = {
        "peak_limit_dbfs": float(peak_limit_dbfs),
        "source_peak_dbfs": current_peak,
        "applied_gain_db": 0.0,
        "skipped": False,
        "reason": None,
    }

    if current_peak is None:
        metadata["skipped"] = True
        metadata["reason"] = "silent_or_invalid"
        return audio, metadata

    if current_peak <= float(peak_limit_dbfs):
        metadata["result_peak_dbfs"] = current_peak
        return audio, metadata

    applied_gain = float(peak_limit_dbfs) - current_peak
    adjusted = audio.apply_gain(applied_gain)
    metadata["applied_gain_db"] = round(applied_gain, 3)
    metadata["result_peak_dbfs"] = _safe_dbfs(adjusted.max_dBFS)
    return adjusted, metadata


def _trim_final_mix_silence(
    audio: AudioSegment,
    *,
    trim_leading: bool,
    trim_trailing: bool,
    silence_threshold_dbfs: float,
    min_silence_len_ms: int,
) -> Tuple[AudioSegment, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "trim_leading_silence": bool(trim_leading),
        "trim_trailing_silence": bool(trim_trailing),
        "silence_threshold_dbfs": float(silence_threshold_dbfs),
        "min_silence_len_ms": int(min_silence_len_ms),
        "leading_trim_ms": 0,
        "trailing_trim_ms": 0,
        "skipped": False,
        "reason": None,
    }

    if not trim_leading and not trim_trailing:
        metadata["skipped"] = True
        metadata["reason"] = "disabled"
        return audio, metadata

    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=max(20, int(min_silence_len_ms)),
        silence_thresh=float(silence_threshold_dbfs),
    )
    if not nonsilent_ranges:
        metadata["skipped"] = True
        metadata["reason"] = "no_nonsilent_audio"
        return audio, metadata

    first_start = max(0, int(nonsilent_ranges[0][0]))
    last_end = min(len(audio), int(nonsilent_ranges[-1][1]))

    trim_start_ms = first_start if trim_leading else 0
    trim_end_ms = last_end if trim_trailing else len(audio)

    if trim_end_ms <= trim_start_ms:
        metadata["skipped"] = True
        metadata["reason"] = "invalid_trim_window"
        return audio, metadata

    trimmed = audio[trim_start_ms:trim_end_ms]
    metadata["leading_trim_ms"] = trim_start_ms
    metadata["trailing_trim_ms"] = max(0, len(audio) - trim_end_ms)
    metadata["result_duration_ms"] = len(trimmed)
    return trimmed, metadata


def _apply_final_fades(
    audio: AudioSegment,
    *,
    fade_in_ms: int,
    fade_out_ms: int,
) -> Tuple[AudioSegment, Dict[str, Any]]:
    applied = audio
    safe_fade_in_ms = max(0, min(int(fade_in_ms), len(audio)))
    safe_fade_out_ms = max(0, min(int(fade_out_ms), len(audio)))

    if safe_fade_in_ms > 0:
        applied = applied.fade_in(safe_fade_in_ms)
    if safe_fade_out_ms > 0:
        applied = applied.fade_out(safe_fade_out_ms)

    return applied, {
        "fade_in_ms": safe_fade_in_ms,
        "fade_out_ms": safe_fade_out_ms,
        "applied": bool(safe_fade_in_ms or safe_fade_out_ms),
    }


def _extract_plan_payload(plan_text: str) -> Dict[str, Any]:
    stripped = (plan_text or "").strip()
    if not stripped:
        return {}

    fence_match = re.search(r"```(?:ya?ml|json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    payload = yaml.safe_load(stripped)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Overlap plan must parse to an object with scene/lines keys")
    return payload


def _line_id_to_index(raw_id: Any) -> Optional[int]:
    if raw_id is None:
        return None
    if isinstance(raw_id, int):
        return raw_id

    text = str(raw_id).strip()
    if text.isdigit():
        return int(text)

    match = re.fullmatch(r"[Ll](\d+)", text)
    if match:
        return int(match.group(1)) - 1

    return None


def parse_overlap_plan(plan_text: str, total_lines: Optional[int] = None) -> Dict[str, Any]:
    payload = _extract_plan_payload(plan_text)
    scene = payload.get("scene", {}) or {}
    line_entries = payload.get("lines", []) or []

    if not isinstance(scene, dict):
        raise ValueError("scene must be an object")
    if not isinstance(line_entries, list):
        raise ValueError("lines must be a list")

    default_gap_ms = int(scene.get("default_gap_ms", 0) or 0)
    default_duck_db = float(scene.get("default_duck_db", -4) or -4)
    overlap_policy = str(scene.get("overlap_policy", "explicit_only") or "explicit_only")

    normalized_lines: Dict[int, Dict[str, Any]] = {}
    for entry in line_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each line plan entry must be an object")

        line_index = _line_id_to_index(entry.get("line_index"))
        if line_index is None:
            line_index = _line_id_to_index(entry.get("id"))
        if line_index is None:
            raise ValueError(f"Could not resolve line id from overlap plan entry: {entry}")
        if total_lines is not None and (line_index < 0 or line_index >= total_lines):
            raise ValueError(f"Line index {line_index} is outside the conversation range")

        start_mode = str(entry.get("start_mode", "after_previous") or "after_previous")
        if start_mode not in ALLOWED_START_MODES:
            raise ValueError(f"Unsupported start_mode: {start_mode}")

        normalized_lines[line_index] = {
            "id": entry.get("id") or f"L{line_index + 1:02d}",
            "start_mode": start_mode,
            "allow_overlap": bool(entry.get("allow_overlap", False)),
            "gap_after_ms": int(entry.get("gap_after_ms", entry.get("gap_before_ms", default_gap_ms)) or 0),
            "overlap_prev_ms": int(entry.get("overlap_prev_ms", 0) or 0),
            "duck_prev_db": float(entry.get("duck_prev_db", default_duck_db) or 0),
            "fade_in_ms": int(entry.get("fade_in_ms", 0) or 0),
            "notes": entry.get("notes", ""),
        }

    return {
        "scene": {
            "title": scene.get("title", ""),
            "overlap_policy": overlap_policy,
            "default_gap_ms": default_gap_ms,
            "default_duck_db": default_duck_db,
        },
        "lines": normalized_lines,
    }


def _estimate_punctuation_pause_ms(text: Optional[str], scene_pacing: Dict[str, int]) -> int:
    stripped = str(text or "").strip()
    if not stripped:
        return 0
    if stripped.endswith("...") or stripped.endswith("…"):
        return int(scene_pacing.get("ellipsis_pause_ms", 0) or 0)
    if stripped.endswith("?"):
        return int(scene_pacing.get("question_pause_ms", 0) or 0)
    if stripped.endswith("!"):
        return int(scene_pacing.get("exclamation_pause_ms", 0) or 0)
    if stripped.endswith(".") or stripped.endswith(";") or stripped.endswith(":"):
        return int(scene_pacing.get("statement_pause_ms", 0) or 0)
    return 0


def build_mix_timeline(
    audio_segments: List[AudioSegment],
    mix_plan: Optional[Dict[str, Any]] = None,
    *,
    line_texts: Optional[List[str]] = None,
    scene_pacing_profile: str = DEFAULT_SCENE_PACING_PROFILE,
    scene_gap_ms: int = DEFAULT_SCENE_GAP_MS,
    respect_punctuation_pauses: bool = True,
) -> List[Dict[str, Any]]:
    if not audio_segments:
        return []

    scene = (mix_plan or {}).get("scene", {})
    line_rules = (mix_plan or {}).get("lines", {})
    resolved_scene_pacing = resolve_scene_pacing(scene_pacing_profile)
    default_gap_ms = int(
        scene.get(
            "default_gap_ms",
            scene_gap_ms if scene_gap_ms is not None else resolved_scene_pacing["default_gap_ms"],
        ) or 0
    )

    timeline: List[Dict[str, Any]] = []
    previous_start_ms = 0
    previous_end_ms = 0

    for index, segment in enumerate(audio_segments):
        duration_ms = len(segment)
        rule = line_rules.get(index, {})
        start_mode = rule.get("start_mode", "after_previous")
        allow_overlap = bool(rule.get("allow_overlap", False))
        gap_after_ms = int(rule.get("gap_after_ms", default_gap_ms) or 0)
        overlap_prev_ms = max(0, int(rule.get("overlap_prev_ms", 0) or 0))
        punctuation_pause_ms = (
            _estimate_punctuation_pause_ms(
                line_texts[index - 1] if line_texts and index - 1 < len(line_texts) else "",
                resolved_scene_pacing,
            )
            if index > 0 and respect_punctuation_pauses
            else 0
        )

        if index == 0:
            start_ms = 0
            applied_mode = "after_previous"
        elif start_mode == "simultaneous" and allow_overlap:
            start_ms = previous_start_ms
            applied_mode = "simultaneous"
        elif start_mode in {"overlap_previous", "hard_cut_previous"} and allow_overlap:
            start_ms = max(0, previous_end_ms - overlap_prev_ms)
            applied_mode = start_mode
        else:
            start_ms = previous_end_ms + gap_after_ms + punctuation_pause_ms
            applied_mode = "after_previous"

        end_ms = start_ms + duration_ms
        timeline.append({
            "line_index": index,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "start_mode": applied_mode,
            "duck_prev_db": float(rule.get("duck_prev_db", scene.get("default_duck_db", 0)) or 0),
            "fade_in_ms": max(0, int(rule.get("fade_in_ms", 0) or 0)),
            "overlap_applied": start_ms < previous_end_ms if index > 0 else False,
            "punctuation_pause_ms": punctuation_pause_ms,
        })

        previous_start_ms = start_ms
        previous_end_ms = end_ms

    return timeline


def mix_audio_files(
    audio_files: List[str],
    output_path: str,
    overlap_plan_text: Optional[str] = None,
    output_format: Optional[str] = None,
    output_bitrate_kbps: int = DEFAULT_EXPORT_BITRATE_KBPS,
    normalize_segments: bool = True,
    target_level_dbfs: float = DEFAULT_TARGET_LEVEL_DBFS,
    peak_limit_dbfs: float = DEFAULT_PEAK_LIMIT_DBFS,
    normalize_final_mix: bool = True,
    trim_leading_silence: bool = True,
    trim_trailing_silence: bool = True,
    trim_silence_threshold_dbfs: float = -42.0,
    trim_min_silence_len_ms: int = 120,
    fade_in_ms: int = 0,
    fade_out_ms: int = 60,
    scene_pacing_profile: str = DEFAULT_SCENE_PACING_PROFILE,
    scene_gap_ms: int = DEFAULT_SCENE_GAP_MS,
    respect_punctuation_pauses: bool = True,
    line_texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not audio_files:
        raise ValueError("No audio files provided for mixing")

    raw_audio_segments = [_load_audio_segment(audio_file) for audio_file in audio_files]
    normalization_details = []
    audio_segments: List[AudioSegment] = []

    for audio_file, segment in zip(audio_files, raw_audio_segments):
        prepared_segment = segment
        if normalize_segments:
            prepared_segment, segment_metadata = _match_segment_loudness(
                segment,
                target_level_dbfs=target_level_dbfs,
                peak_limit_dbfs=peak_limit_dbfs,
            )
        else:
            segment_metadata = {
                "source_level_dbfs": _safe_dbfs(segment.dBFS),
                "source_peak_dbfs": _safe_dbfs(segment.max_dBFS),
                "applied_gain_db": 0.0,
                "target_level_dbfs": float(target_level_dbfs),
                "peak_limit_dbfs": float(peak_limit_dbfs),
                "skipped": True,
                "reason": "disabled",
                "result_level_dbfs": _safe_dbfs(segment.dBFS),
                "result_peak_dbfs": _safe_dbfs(segment.max_dBFS),
            }

        segment_metadata["audio_file"] = str(audio_file)
        normalization_details.append(segment_metadata)
        audio_segments.append(prepared_segment)

    mix_plan = parse_overlap_plan(overlap_plan_text, len(audio_segments)) if overlap_plan_text else None
    normalized_scene_pacing_profile = normalize_scene_pacing_profile(scene_pacing_profile)
    timeline = build_mix_timeline(
        audio_segments,
        mix_plan,
        line_texts=line_texts,
        scene_pacing_profile=normalized_scene_pacing_profile,
        scene_gap_ms=scene_gap_ms,
        respect_punctuation_pauses=respect_punctuation_pauses,
    )
    scene_metadata = resolve_scene_pacing(normalized_scene_pacing_profile)
    scene_metadata.update((mix_plan or {}).get("scene", {}))
    scene_metadata["profile"] = normalized_scene_pacing_profile
    scene_metadata["default_gap_ms"] = int(scene_metadata.get("default_gap_ms", scene_gap_ms) or scene_gap_ms)
    scene_metadata["respect_punctuation_pauses"] = bool(respect_punctuation_pauses)

    total_duration_ms = max((item["end_ms"] for item in timeline), default=0)
    if total_duration_ms <= 0:
        raise ValueError("Calculated mix duration was invalid")

    base = AudioSegment.silent(duration=total_duration_ms, frame_rate=audio_segments[0].frame_rate)
    base = base.set_channels(audio_segments[0].channels).set_sample_width(audio_segments[0].sample_width)

    mixed = base
    for segment, placement in zip(audio_segments, timeline):
        prepared_segment = segment
        if placement["fade_in_ms"] > 0:
            prepared_segment = prepared_segment.fade_in(placement["fade_in_ms"])

        duck_prev_db = placement["duck_prev_db"] if placement["overlap_applied"] else 0
        mixed = mixed.overlay(
            prepared_segment,
            position=placement["start_ms"],
            gain_during_overlay=duck_prev_db,
        )

    final_mix_metadata = {
        "source_peak_dbfs": _safe_dbfs(mixed.max_dBFS),
        "applied_gain_db": 0.0,
        "peak_limit_dbfs": float(peak_limit_dbfs),
        "skipped": True,
        "reason": "disabled",
    }
    if normalize_final_mix:
        mixed, final_mix_metadata = _apply_final_peak_protection(
            mixed,
            peak_limit_dbfs=peak_limit_dbfs,
        )

    mixed, silence_trim_metadata = _trim_final_mix_silence(
        mixed,
        trim_leading=trim_leading_silence,
        trim_trailing=trim_trailing_silence,
        silence_threshold_dbfs=trim_silence_threshold_dbfs,
        min_silence_len_ms=trim_min_silence_len_ms,
    )
    mixed, fade_metadata = _apply_final_fades(
        mixed,
        fade_in_ms=fade_in_ms,
        fade_out_ms=fade_out_ms,
    )

    resolved_output_format = _resolve_output_format(output_path, output_format)
    applied_output_bitrate_kbps = int(output_bitrate_kbps) if resolved_output_format == "mp3" else None
    _export_segment(
        mixed,
        output_path,
        output_format=resolved_output_format,
        output_bitrate_kbps=applied_output_bitrate_kbps,
    )

    return {
        "success": Path(output_path).exists(),
        "output_path": str(Path(output_path)),
        "output_format": resolved_output_format,
        "timeline": timeline,
        "plan_applied": bool(mix_plan),
        "scene": scene_metadata,
        "duration_ms": len(mixed),
        "normalization": {
            "normalize_segments": bool(normalize_segments),
            "normalize_final_mix": bool(normalize_final_mix),
            "target_level_dbfs": float(target_level_dbfs),
            "peak_limit_dbfs": float(peak_limit_dbfs),
            "segment_adjustments": normalization_details,
            "final_mix_adjustment": final_mix_metadata,
        },
        "finishing": {
            "trim_leading_silence": bool(trim_leading_silence),
            "trim_trailing_silence": bool(trim_trailing_silence),
            "trim_silence_threshold_dbfs": float(trim_silence_threshold_dbfs),
            "trim_min_silence_len_ms": int(trim_min_silence_len_ms),
            "silence_trim": silence_trim_metadata,
            "fades": fade_metadata,
            "output_bitrate_kbps": applied_output_bitrate_kbps,
        },
    }


def mix_audio_files_at_positions(
    placements: List[Dict[str, Any]],
    output_path: str,
    output_format: Optional[str] = None,
    output_bitrate_kbps: int = DEFAULT_EXPORT_BITRATE_KBPS,
    total_duration_ms: Optional[int] = None,
    duck_overlaps: bool = False,
    duck_amount_db: float = 0.0,
    duck_fade_ms: int = 0,
    normalize_segments: bool = True,
    target_level_dbfs: float = DEFAULT_TARGET_LEVEL_DBFS,
    peak_limit_dbfs: float = DEFAULT_PEAK_LIMIT_DBFS,
    normalize_final_mix: bool = True,
    trim_leading_silence: bool = True,
    trim_trailing_silence: bool = True,
    trim_silence_threshold_dbfs: float = -42.0,
    trim_min_silence_len_ms: int = 120,
    fade_in_ms: int = 0,
    fade_out_ms: int = 60,
) -> Dict[str, Any]:
    if not placements:
        raise ValueError("No timeline placements provided for mixing")

    loaded_segments: List[Dict[str, Any]] = []
    max_end_ms = 0
    normalization_details = []

    for placement in placements:
        audio_path = placement.get("audio_path")
        if not audio_path:
            raise ValueError(f"Timeline placement is missing an audio_path: {placement}")

        start_ms = max(0, int(round(float(placement.get("start_ms", 0) or 0))))
        volume = float(placement.get("volume", 1.0) or 1.0)
        segment = _load_audio_segment(str(audio_path))

        if normalize_segments:
            segment, segment_metadata = _match_segment_loudness(
                segment,
                target_level_dbfs=target_level_dbfs,
                peak_limit_dbfs=peak_limit_dbfs,
            )
        else:
            segment_metadata = {
                "source_level_dbfs": _safe_dbfs(segment.dBFS),
                "source_peak_dbfs": _safe_dbfs(segment.max_dBFS),
                "applied_gain_db": 0.0,
                "target_level_dbfs": float(target_level_dbfs),
                "peak_limit_dbfs": float(peak_limit_dbfs),
                "skipped": True,
                "reason": "disabled",
                "result_level_dbfs": _safe_dbfs(segment.dBFS),
                "result_peak_dbfs": _safe_dbfs(segment.max_dBFS),
            }

        if volume <= 0:
            segment = segment - 120
        elif not math.isclose(volume, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            segment = segment.apply_gain(20 * math.log10(volume))

        segment_metadata["audio_path"] = str(audio_path)
        segment_metadata["track_volume"] = volume
        segment_metadata["result_level_dbfs"] = _safe_dbfs(segment.dBFS)
        segment_metadata["result_peak_dbfs"] = _safe_dbfs(segment.max_dBFS)
        normalization_details.append(segment_metadata)

        end_ms = start_ms + len(segment)
        max_end_ms = max(max_end_ms, end_ms)

        loaded_segments.append(
            {
                **placement,
                "audio_path": str(audio_path),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": len(segment),
                "segment": segment,
                "duck_applied_db": 0.0,
                "overlaps_existing": False,
            }
        )

    mix_duration_ms = max(max_end_ms, int(total_duration_ms or 0))
    if mix_duration_ms <= 0:
        raise ValueError("Calculated timeline mix duration was invalid")

    first_segment = loaded_segments[0]["segment"]
    mixed = AudioSegment.silent(duration=mix_duration_ms, frame_rate=first_segment.frame_rate)
    mixed = mixed.set_channels(first_segment.channels).set_sample_width(first_segment.sample_width)

    sorted_segments = sorted(loaded_segments, key=lambda item: (item["start_ms"], item.get("track_id", ""), item.get("segment_id", "")))

    active_segments: List[Dict[str, Any]] = []
    for placement in sorted_segments:
        prepared_segment = placement["segment"]
        active_segments = [item for item in active_segments if item["end_ms"] > placement["start_ms"]]
        overlaps_existing = bool(active_segments)

        if duck_overlaps and overlaps_existing and duck_amount_db > 0:
            prepared_segment = prepared_segment.apply_gain(-abs(float(duck_amount_db)))
            if duck_fade_ms > 0:
                prepared_segment = prepared_segment.fade_in(min(int(duck_fade_ms), len(prepared_segment)))
            placement["duck_applied_db"] = abs(float(duck_amount_db))
            placement["overlaps_existing"] = True

        mixed = mixed.overlay(prepared_segment, position=placement["start_ms"])
        active_segments.append(placement)

    final_mix_metadata = {
        "source_peak_dbfs": _safe_dbfs(mixed.max_dBFS),
        "applied_gain_db": 0.0,
        "peak_limit_dbfs": float(peak_limit_dbfs),
        "skipped": True,
        "reason": "disabled",
    }
    if normalize_final_mix:
        mixed, final_mix_metadata = _apply_final_peak_protection(
            mixed,
            peak_limit_dbfs=peak_limit_dbfs,
        )

    mixed, silence_trim_metadata = _trim_final_mix_silence(
        mixed,
        trim_leading=trim_leading_silence,
        trim_trailing=trim_trailing_silence,
        silence_threshold_dbfs=trim_silence_threshold_dbfs,
        min_silence_len_ms=trim_min_silence_len_ms,
    )
    mixed, fade_metadata = _apply_final_fades(
        mixed,
        fade_in_ms=fade_in_ms,
        fade_out_ms=fade_out_ms,
    )

    resolved_output_format = _resolve_output_format(output_path, output_format)
    applied_output_bitrate_kbps = int(output_bitrate_kbps) if resolved_output_format == "mp3" else None
    _export_segment(
        mixed,
        output_path,
        output_format=resolved_output_format,
        output_bitrate_kbps=applied_output_bitrate_kbps,
    )

    return {
        "success": Path(output_path).exists(),
        "output_path": str(Path(output_path)),
        "output_format": resolved_output_format,
        "duration_ms": len(mixed),
        "timeline": [
            {
                key: value
                for key, value in placement.items()
                if key != "segment"
            }
            for placement in loaded_segments
        ],
        "clip_count": len(loaded_segments),
        "normalization": {
            "normalize_segments": bool(normalize_segments),
            "normalize_final_mix": bool(normalize_final_mix),
            "target_level_dbfs": float(target_level_dbfs),
            "peak_limit_dbfs": float(peak_limit_dbfs),
            "segment_adjustments": normalization_details,
            "final_mix_adjustment": final_mix_metadata,
        },
        "finishing": {
            "trim_leading_silence": bool(trim_leading_silence),
            "trim_trailing_silence": bool(trim_trailing_silence),
            "trim_silence_threshold_dbfs": float(trim_silence_threshold_dbfs),
            "trim_min_silence_len_ms": int(trim_min_silence_len_ms),
            "silence_trim": silence_trim_metadata,
            "fades": fade_metadata,
            "output_bitrate_kbps": applied_output_bitrate_kbps,
        },
    }
