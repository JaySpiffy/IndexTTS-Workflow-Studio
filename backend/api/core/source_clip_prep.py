"""
Source clip diagnostics and preparation helpers.

These helpers power the speaker prep workflow for source clips before they are
promoted into reusable speaker prompts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence, detect_nonsilent

from .audio_processing import (
    DEEPFILTERNET_AVAILABLE,
    NOISEREDUCE_AVAILABLE,
    apply_deepfilter_noise_reduction,
    nr,
    separate_vocals,
)


IDEAL_DURATION_MIN_SECONDS = 8.0
IDEAL_DURATION_MAX_SECONDS = 20.0
ACCEPTABLE_DURATION_MIN_SECONDS = 5.0
ACCEPTABLE_DURATION_MAX_SECONDS = 35.0
AUTO_ACCEPT_SCORE = 85
AUTO_PREP_SCORE = 60


def _round_seconds(value: float | int | None) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 2)


def _measure_edge_silence_ms(segment: AudioSegment, from_start: bool, chunk_ms: int = 10, silence_threshold_dbfs: float = -40.0) -> int:
    if len(segment) <= 0:
        return 0

    edge_silence = 0
    cursor = 0
    total_ms = len(segment)

    while cursor < total_ms:
        start = cursor if from_start else max(total_ms - cursor - chunk_ms, 0)
        end = min(start + chunk_ms, total_ms)
        chunk = segment[start:end]
        if len(chunk) == 0:
            break
        chunk_level = chunk.dBFS if chunk.rms else float("-inf")
        if chunk_level > silence_threshold_dbfs:
            break
        edge_silence += len(chunk)
        cursor += chunk_ms

    return edge_silence


def _finite_dbfs(value: float | int | None) -> Optional[float]:
    if value is None:
        return None
    try:
        db_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isinf(db_value) or math.isnan(db_value):
        return None
    return db_value


def _bounded_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def _segment_to_float32_samples(segment: AudioSegment) -> Tuple[np.ndarray, int]:
    sample_width = segment.sample_width
    dtype_map = {
        1: np.int8,
        2: np.int16,
        4: np.int32,
    }
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels > 1:
        samples = samples.reshape((-1, segment.channels))

    scale = float(2 ** (8 * sample_width - 1))
    samples = np.clip(samples / scale, -1.0, 1.0)
    return samples, segment.frame_rate


def _load_audio_segment(path: str | Path) -> AudioSegment:
    audio_path = Path(path)
    audio_format = audio_path.suffix.lstrip(".") or None
    with audio_path.open("rb") as audio_file:
        return AudioSegment.from_file(audio_file, format=audio_format)


def _float32_samples_to_segment(samples: np.ndarray, sample_rate: int, channels: int = 1) -> AudioSegment:
    clipped = np.clip(samples, -1.0, 1.0)
    if channels > 1 and clipped.ndim == 1:
        clipped = np.repeat(clipped[:, None], channels, axis=1)
    if channels == 1 and clipped.ndim > 1:
        clipped = np.mean(clipped, axis=1)

    int_samples = (clipped * 32767.0).astype(np.int16)
    raw = int_samples.tobytes()
    return AudioSegment(
        data=raw,
        sample_width=2,
        frame_rate=sample_rate,
        channels=channels,
    )


def _find_best_trim_window_ms(
    segment: AudioSegment,
    *,
    silence_threshold_dbfs: float = -42.0,
    min_window_ms: int = int(IDEAL_DURATION_MIN_SECONDS * 1000),
    max_window_ms: int = int(IDEAL_DURATION_MAX_SECONDS * 1000),
    step_ms: int = 500,
) -> Tuple[int, Optional[int], List[str]]:
    notes: List[str] = []
    total_ms = len(segment)
    if total_ms <= 0:
        return 0, None, notes

    nonsilent_ranges = detect_nonsilent(
        segment,
        min_silence_len=250,
        silence_thresh=silence_threshold_dbfs,
    )

    if not nonsilent_ranges:
        return 0, None, notes

    content_start_ms = nonsilent_ranges[0][0]
    content_end_ms = nonsilent_ranges[-1][1]
    active_duration_ms = max(0, content_end_ms - content_start_ms)

    if active_duration_ms <= max_window_ms:
        suggested_end = content_end_ms if content_end_ms < total_ms - 50 else None
        if content_start_ms > 0:
            notes.append("Trim the silence before the first spoken words.")
        if suggested_end is not None and content_end_ms < total_ms:
            notes.append("Trim the silence after the final spoken words.")
        return content_start_ms, suggested_end, notes

    search_start_ms = content_start_ms
    search_end_ms = max(content_start_ms, content_end_ms - max_window_ms)

    best_score = None
    best_window = (content_start_ms, min(content_start_ms + max_window_ms, content_end_ms))

    for candidate_start in range(search_start_ms, search_end_ms + 1, step_ms):
        candidate_end = min(candidate_start + max_window_ms, content_end_ms)
        candidate = segment[candidate_start:candidate_end]
        if len(candidate) < min_window_ms:
            continue

        candidate_silences = detect_silence(
            candidate,
            min_silence_len=250,
            silence_thresh=silence_threshold_dbfs,
        )
        candidate_silence_ms = sum((end - start) for start, end in candidate_silences)
        candidate_silence_ratio = candidate_silence_ms / max(len(candidate), 1)
        candidate_level = _finite_dbfs(candidate.dBFS)
        level_score = candidate_level if candidate_level is not None else -80.0
        density_score = 1.0 - candidate_silence_ratio
        score = (density_score * 100.0) + level_score

        if best_score is None or score > best_score:
            best_score = score
            best_window = (candidate_start, candidate_end)

    best_start, best_end = best_window
    notes.append("Keep the prompt inside the strongest 8 to 20 second spoken window.")
    if best_start > 0:
        notes.append("Trim the silence before the chosen prompt window.")
    if best_end < total_ms:
        notes.append("Trim the silence and weaker trailing material after the chosen prompt window.")

    return best_start, best_end, notes


def analyze_source_clip(audio_path: str | Path) -> Dict[str, Any]:
    path = Path(audio_path)
    segment = _load_audio_segment(path)

    duration_seconds = round(len(segment) / 1000.0, 2)
    level_dbfs = _finite_dbfs(segment.dBFS)
    peak_dbfs = _finite_dbfs(segment.max_dBFS)
    silence_threshold = -42.0
    silence_ranges = detect_silence(segment, min_silence_len=250, silence_thresh=silence_threshold)
    total_silence_ms = int(sum((end - start) for start, end in silence_ranges))
    silence_percent = round((total_silence_ms / max(len(segment), 1)) * 100.0, 1)
    leading_silence_ms = _measure_edge_silence_ms(segment, from_start=True, silence_threshold_dbfs=silence_threshold)
    trailing_silence_ms = _measure_edge_silence_ms(segment, from_start=False, silence_threshold_dbfs=silence_threshold)
    active_speech_seconds = round(duration_seconds * max(0.0, (100.0 - silence_percent) / 100.0), 2) if duration_seconds else 0.0

    dynamic_range_db = None
    if level_dbfs is not None and peak_dbfs is not None:
        dynamic_range_db = round(max(0.0, peak_dbfs - level_dbfs), 2)

    recommendations: List[str] = []
    warnings: List[str] = []
    hard_fail_reasons: List[str] = []
    repairable_issues: List[str] = []

    input_extension = path.suffix.lower()
    lossy_formats = {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
    is_lossy_source = input_extension in lossy_formats

    perceptual_quality = 100.0
    acoustic_clarity = 100.0
    structural_integrity = 100.0
    format_and_dynamics = 100.0

    if duration_seconds < ACCEPTABLE_DURATION_MIN_SECONDS:
        structural_integrity -= 45
        warnings.append("Clip is very short for cloning.")
        recommendations.append("Aim for a cleaner 8 to 20 second sample if possible.")
        hard_fail_reasons.append("Less than 5 seconds of total prompt audio.")
    elif duration_seconds < IDEAL_DURATION_MIN_SECONDS:
        structural_integrity -= 15
        recommendations.append("A slightly longer sample would usually clone more consistently.")
        repairable_issues.append("Short prompt length may limit tonal and emotional stability.")
    elif duration_seconds > ACCEPTABLE_DURATION_MAX_SECONDS:
        structural_integrity -= 20
        warnings.append("Clip is long enough that cleanup and trimming are recommended.")
        recommendations.append("Trim the clip down to the strongest 8 to 20 seconds.")
        repairable_issues.append("Prompt is long enough that trimming is recommended.")
    elif duration_seconds > IDEAL_DURATION_MAX_SECONDS:
        structural_integrity -= 8
        recommendations.append("Consider trimming this closer to the strongest 8 to 20 second section.")
        repairable_issues.append("Prompt is usable, but a tighter focus window would usually clone more consistently.")

    if segment.channels > 1:
        format_and_dynamics -= 8
        recommendations.append("Convert the clip to mono before using it as a speaker prompt.")
        repairable_issues.append("Stereo prompt should be downmixed to mono.")

    if leading_silence_ms > 350:
        acoustic_clarity -= 10
        recommendations.append("Trim the leading silence so the voice starts sooner.")
        repairable_issues.append("Leading silence is wasting prompt space.")

    if trailing_silence_ms > 500:
        acoustic_clarity -= 8
        recommendations.append("Trim the trailing silence after the last spoken phrase.")
        repairable_issues.append("Trailing silence should be trimmed.")

    if silence_percent > 25.0:
        acoustic_clarity -= 20
        warnings.append("Large silent sections reduce prompt quality.")
        recommendations.append("Trim long pauses and dead air out of the sample.")
        repairable_issues.append("Large silent sections reduce the amount of usable speech.")
    elif silence_percent > 15.0:
        acoustic_clarity -= 8
        recommendations.append("This clip has a fair amount of silence. Trimming should help.")
        repairable_issues.append("Moderate silence should be trimmed down.")

    if level_dbfs is not None and level_dbfs < -28.0:
        perceptual_quality -= 15
        warnings.append("Overall level is very quiet.")
        recommendations.append("Normalize the clip before promoting it to a speaker.")
        repairable_issues.append("Low average level should be normalized.")
    elif level_dbfs is not None and level_dbfs < -24.0:
        perceptual_quality -= 8
        recommendations.append("A modest normalization pass would help this clip.")
        repairable_issues.append("Average level is low for stable cloning.")

    if peak_dbfs is not None and peak_dbfs > -0.3:
        perceptual_quality -= 25
        format_and_dynamics -= 20
        warnings.append("Peak level is very hot and may be clipped.")
        recommendations.append("Reduce clipping or use a cleaner source sample.")
        hard_fail_reasons.append("Peak level is so hot that clipping is likely.")
    elif peak_dbfs is not None and peak_dbfs > -1.0:
        perceptual_quality -= 10
        format_and_dynamics -= 8
        recommendations.append("Leave a little more headroom. Targeting around -1 dBFS is safer.")
        repairable_issues.append("Peak headroom is tight and should be normalized.")

    if dynamic_range_db is not None and dynamic_range_db < 5.0:
        perceptual_quality -= 10
        format_and_dynamics -= 8
        recommendations.append("This clip is heavily compressed. A more natural source can clone better.")
        repairable_issues.append("Heavy compression may flatten the timbre and micro-dynamics.")

    if segment.frame_rate < 16000:
        acoustic_clarity -= 20
        format_and_dynamics -= 15
        warnings.append("Sample rate is low for a modern cloning prompt.")
        hard_fail_reasons.append("Sample rate is below 16 kHz.")
    elif segment.frame_rate < 22050:
        acoustic_clarity -= 12
        format_and_dynamics -= 10
        recommendations.append("A cleaner source recorded at 22.05 kHz or above will usually clone better.")
        repairable_issues.append("Lower sample rate reduces high-frequency identity detail.")

    if segment.sample_width != 2:
        format_and_dynamics -= 8
        repairable_issues.append("Clip should be converted to 16-bit PCM WAV for consistency.")

    if is_lossy_source:
        perceptual_quality -= 8
        format_and_dynamics -= 15
        warnings.append("Lossy source formats can blur detail that helps speaker matching.")
        recommendations.append("Prefer an uncompressed WAV if you can get one.")
        repairable_issues.append("Lossy compression may soften the speaker's micro-timbre.")

    if active_speech_seconds < ACCEPTABLE_DURATION_MIN_SECONDS:
        structural_integrity -= 15
        repairable_issues.append("Usable voiced content is shorter than the total file suggests.")

    perceptual_quality_score = _bounded_score(perceptual_quality)
    acoustic_clarity_score = _bounded_score(acoustic_clarity)
    structural_integrity_score = _bounded_score(structural_integrity)
    format_and_dynamics_score = _bounded_score(format_and_dynamics)
    score = _bounded_score(
        (perceptual_quality_score * 0.4)
        + (acoustic_clarity_score * 0.3)
        + (structural_integrity_score * 0.2)
        + (format_and_dynamics_score * 0.1)
    )

    if score >= AUTO_ACCEPT_SCORE and not hard_fail_reasons:
        readiness = "excellent"
        quality_gate_status = "accept"
    elif score >= 70 and not hard_fail_reasons:
        readiness = "good"
        quality_gate_status = "prep"
    elif score >= AUTO_PREP_SCORE and not hard_fail_reasons:
        readiness = "fair"
        quality_gate_status = "prep"
    else:
        readiness = "poor"
        quality_gate_status = "reject"

    if not recommendations:
        recommendations.append("This clip already looks healthy for cloning.")

    suggested_start_ms, suggested_end_ms, suggested_prep_reasons = _find_best_trim_window_ms(
        segment,
        silence_threshold_dbfs=silence_threshold,
    )
    suggested_start_seconds = _round_seconds(suggested_start_ms / 1000.0) or 0.0
    suggested_end_seconds = None if suggested_end_ms is None else _round_seconds(suggested_end_ms / 1000.0)
    effective_end_ms = len(segment) if suggested_end_ms is None else suggested_end_ms
    suggested_duration_seconds = _round_seconds(
        (effective_end_ms / 1000.0) - suggested_start_seconds
    )

    should_normalize = False
    if peak_dbfs is None:
        should_normalize = True
    elif peak_dbfs < -2.0 or peak_dbfs > -0.6:
        should_normalize = True
    elif level_dbfs is not None and level_dbfs < -24.0:
        should_normalize = True

    suggested_prep = {
        "start_time": suggested_start_seconds,
        "end_time": suggested_end_seconds,
        "duration_seconds": suggested_duration_seconds,
        "convert_to_mono": segment.channels > 1,
        "normalize_audio": should_normalize,
        "target_peak_dbfs": -1.0,
        "use_noise_reduction": level_dbfs is not None and level_dbfs < -26.0 and not hard_fail_reasons,
        "noise_reduction_backend": "deepfilter" if DEEPFILTERNET_AVAILABLE else "classic",
        "use_vocal_separation": False,
        "reasons": suggested_prep_reasons,
    }

    quality_gate = {
        "status": quality_gate_status,
        "label": {
            "accept": "Ready for cloning",
            "prep": "Prep recommended",
            "reject": "Needs a better source",
        }[quality_gate_status],
        "auto_accept": quality_gate_status == "accept",
        "auto_prep_eligible": quality_gate_status == "prep",
        "manual_override_required": quality_gate_status == "reject",
    }

    if quality_gate_status == "accept":
        recommended_action_summary = "This clip is strong enough to use as-is, though the suggested trim may still help."
    elif quality_gate_status == "prep":
        recommended_action_summary = "Run the suggested prep recipe before promoting this clip into the speaker library."
    else:
        recommended_action_summary = "This clip should usually be replaced with a cleaner source rather than forced through heavy repair."

    return {
        "filename": path.name,
        "path": str(path),
        "input_extension": input_extension,
        "is_lossy_source": is_lossy_source,
        "duration_seconds": duration_seconds,
        "active_speech_seconds": active_speech_seconds,
        "channels": segment.channels,
        "sample_rate_hz": segment.frame_rate,
        "sample_width_bytes": segment.sample_width,
        "level_dbfs": level_dbfs,
        "peak_dbfs": peak_dbfs,
        "dynamic_range_db": dynamic_range_db,
        "silence_percent": silence_percent,
        "leading_silence_ms": leading_silence_ms,
        "trailing_silence_ms": trailing_silence_ms,
        "clone_readiness_score": score,
        "clone_readiness_label": readiness,
        "ready_for_cloning": quality_gate_status == "accept",
        "quality_gate": quality_gate,
        "recommended_action_summary": recommended_action_summary,
        "hard_fail_reasons": hard_fail_reasons,
        "repairable_issues": repairable_issues,
        "score_breakdown": {
            "perceptual_quality": perceptual_quality_score,
            "acoustic_clarity": acoustic_clarity_score,
            "structural_integrity": structural_integrity_score,
            "format_and_dynamics": format_and_dynamics_score,
        },
        "warnings": warnings,
        "recommendations": recommendations,
        "suggested_prep": suggested_prep,
    }


def prepare_source_clip(
    source_path: str | Path,
    output_path: str | Path,
    *,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    convert_to_mono: bool = True,
    normalize_audio: bool = True,
    target_peak_dbfs: float = -1.0,
    use_noise_reduction: bool = False,
    noise_reduction_strength: float = 0.35,
    noise_reduction_backend: str = "auto",
    use_vocal_separation: bool = False,
) -> Dict[str, Any]:
    source = Path(source_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    segment = _load_audio_segment(source)
    original_metrics = analyze_source_clip(source)
    processing_notes: List[str] = []

    total_duration_ms = len(segment)
    if start_time is not None or end_time is not None:
        start_ms = max(0, int((start_time or 0.0) * 1000))
        end_ms = total_duration_ms if end_time is None else min(total_duration_ms, int(end_time * 1000))
        if end_ms <= start_ms:
            raise ValueError("Trim end time must be greater than trim start time.")
        segment = segment[start_ms:end_ms]
        processing_notes.append(f"Trimmed clip to {round(len(segment) / 1000.0, 2)} seconds.")

    if convert_to_mono and segment.channels > 1:
        segment = segment.set_channels(1)
        processing_notes.append("Converted clip to mono.")

    # Run optional DSP using float32 arrays.
    if use_vocal_separation or use_noise_reduction:
        if segment.channels > 1:
            segment = segment.set_channels(1)
            processing_notes.append("Converted clip to mono for cleanup processing.")

        sample_array, sample_rate = _segment_to_float32_samples(segment)
        if sample_array.ndim > 1:
            sample_array = np.mean(sample_array, axis=1)

        if use_vocal_separation:
            separated = separate_vocals(sample_array, sample_rate)
            if separated is not None:
                sample_array = np.asarray(separated, dtype=np.float32)
                processing_notes.append("Applied vocal separation to emphasize the voice.")
            else:
                processing_notes.append("Vocal separation was requested but no model was available.")

        if use_noise_reduction:
            requested_backend = (noise_reduction_backend or "auto").strip().lower()
            resolved_backend = requested_backend
            if requested_backend == "auto":
                resolved_backend = "deepfilter" if DEEPFILTERNET_AVAILABLE else "classic"

            if resolved_backend == "deepfilter":
                deepfilter_output = apply_deepfilter_noise_reduction(
                    sample_array,
                    sample_rate,
                    noise_reduction_strength=noise_reduction_strength,
                )
                if deepfilter_output is not None:
                    sample_array = deepfilter_output.astype(np.float32)
                    processing_notes.append("Applied DeepFilterNet speech cleanup.")
                elif NOISEREDUCE_AVAILABLE and nr is not None:
                    sample_array = nr.reduce_noise(
                        y=sample_array,
                        sr=sample_rate,
                        prop_decrease=noise_reduction_strength,
                    ).astype(np.float32)
                    processing_notes.append("DeepFilterNet cleanup was unavailable, so classic noise reduction was used instead.")
                else:
                    processing_notes.append("DeepFilterNet cleanup was requested but no cleanup backend was available.")
            elif NOISEREDUCE_AVAILABLE and nr is not None:
                sample_array = nr.reduce_noise(
                    y=sample_array,
                    sr=sample_rate,
                    prop_decrease=noise_reduction_strength,
                ).astype(np.float32)
                processing_notes.append("Applied classic noise reduction.")
            else:
                processing_notes.append("Classic noise reduction was requested but the dependency is not available.")

        segment = _float32_samples_to_segment(sample_array, sample_rate, channels=1)

    if segment.sample_width != 2:
        segment = segment.set_sample_width(2)

    if normalize_audio:
        peak_dbfs = _finite_dbfs(segment.max_dBFS)
        if peak_dbfs is not None:
            gain_db = max(-24.0, min(24.0, float(target_peak_dbfs) - peak_dbfs))
            if abs(gain_db) > 0.1:
                segment = segment.apply_gain(gain_db)
                processing_notes.append(f"Normalized clip by {round(gain_db, 2)} dB toward {target_peak_dbfs} dBFS.")

    export_handle = segment.export(destination, format="wav")
    try:
        export_handle.close()
    except Exception:
        pass

    prepared_metrics = analyze_source_clip(destination)

    return {
        "source_filename": source.name,
        "output_filename": destination.name,
        "output_path": str(destination),
        "processing_notes": processing_notes,
        "applied_options": {
            "start_time": start_time,
            "end_time": end_time,
            "convert_to_mono": convert_to_mono,
            "normalize_audio": normalize_audio,
            "target_peak_dbfs": target_peak_dbfs,
            "use_noise_reduction": use_noise_reduction,
            "noise_reduction_strength": noise_reduction_strength,
            "noise_reduction_backend": noise_reduction_backend,
            "use_vocal_separation": use_vocal_separation,
        },
        "before": original_metrics,
        "after": prepared_metrics,
    }
