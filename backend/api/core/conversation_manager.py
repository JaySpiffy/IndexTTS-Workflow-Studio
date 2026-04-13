"""
Conversation Manager module for standalone FastAPI implementation.
Handles multi-speaker conversation generation and management without Gradio dependencies.
"""

import os
import uuid
import time
import random
from typing import Dict, Any, List, Optional, Tuple, Generator
from pathlib import Path

from .app_paths import SPEAKERS_DIR, TEMP_CONVERSATION_SEGMENTS_DIR
from .pacing import (
    apply_delivery_rate_to_file,
    assess_line_pacing,
    build_speaker_pacing_map,
    resolve_speaker_delivery_rate,
)

# Try to import pydub for audio concatenation
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None


SEED_MODULUS = 2 ** 32
DEFAULT_FIXED_BASE_SEED = 1234
DEFAULT_QUALITY_GATE_MIN_QUALITY_SCORE = 0.48
DEFAULT_QUALITY_GATE_MIN_PACING_SCORE = 0.45

class ConversationManager:
    """Manages multi-speaker conversation generation."""
    
    def __init__(self, tts_core, cmd_args):
        """
        Initialize the conversation manager.
        
        Args:
            tts_core: TTSCore instance
            cmd_args: Command line arguments
        """
        self.tts_core = tts_core
        self.cmd_args = cmd_args
        self.temp_dir = TEMP_CONVERSATION_SEGMENTS_DIR
        self.speakers_dir = SPEAKERS_DIR
        print(f"DEBUG: ConversationManager initialized with speakers_dir: {self.speakers_dir}")
        print(f"DEBUG: ConversationManager speakers_dir: {self.speakers_dir}")
        
        # Keep a strong reference to prevent garbage collection
        self._tts_reference = tts_core.tts
        
        # Track generated files for cleanup
        self._generated_files = set()
        
        # Stop mechanism
        self._should_stop = False
        
    def cleanup_generated_files(self, preserve_audio=False):
        """Clean up generated files, optionally preserving audio files."""
        import shutil
        try:
            if preserve_audio:
                # Only clear the tracking set, don't delete actual files
                self._generated_files.clear()
                return True
            else:
                # Full cleanup - delete all files
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                    self.temp_dir.mkdir(parents=True, exist_ok=True)
                self._generated_files.clear()
                return True
        except Exception as e:
            print(f"Error cleaning up generated files: {e}")
            return False
            
    def cleanup_memory(self):
        """Clean up GPU memory and other resources."""
        import gc
        import torch
        try:
            tts_instance = getattr(self.tts_core, 'tts', None) or self.tts_core
            if hasattr(tts_instance, "release_unused_memory"):
                tts_instance.release_unused_memory(clear_prompt_cache=True)
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return True
        except Exception as e:
            print(f"Error cleaning up memory: {e}")
            return False
            
    def reset_conversation(self, preserve_audio=False):
        """Reset conversation state and clean up resources.
        
        Args:
            preserve_audio: If True, keeps audio files for timeline editing
        """
        self.cleanup_generated_files(preserve_audio=preserve_audio)
        self.cleanup_memory()
        return True
    
    def stop_generation(self):
        """Stop the current generation process."""
        self._should_stop = True
        return True
        
    def reset_stop_flag(self):
        """Reset the stop flag for new generations."""
        self._should_stop = False
        return True

    @staticmethod
    def _normalize_seed_strategy(seed_strategy: Optional[str]) -> str:
        """Normalize unknown seed strategy values back to the default."""
        allowed = {
            "fully_random",
            "random_base_sequential",
            "fixed_base_sequential",
            "fixed_base_reused_list",
            "random_base_reused_list",
        }
        normalized = str(seed_strategy or "fully_random").strip().lower()
        return normalized if normalized in allowed else "fully_random"

    @staticmethod
    def _normalize_seed_value(seed_value: Optional[Any], fallback: int = DEFAULT_FIXED_BASE_SEED) -> int:
        """Coerce arbitrary seed values into an unsigned 32-bit integer."""
        try:
            numeric_seed = int(seed_value)
        except (TypeError, ValueError):
            numeric_seed = int(fallback)
        return numeric_seed % SEED_MODULUS

    @staticmethod
    def _normalize_score_threshold(value: Optional[Any], fallback: float) -> float:
        """Clamp arbitrary score thresholds to the supported 0..1 range."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = float(fallback)
        return max(0.0, min(1.0, numeric_value))

    @classmethod
    def _evaluate_quality_gate(
        cls,
        *,
        similarity_score: float,
        robotic_score: float,
        quality_score: float,
        pacing_score: Optional[float],
        similarity_threshold: float,
        robotic_threshold: float,
    ) -> Dict[str, Any]:
        """Return whether a generated version is safe to auto-select."""
        normalized_similarity_threshold = cls._normalize_score_threshold(similarity_threshold, 0.60)
        normalized_robotic_threshold = cls._normalize_score_threshold(robotic_threshold, 0.70)
        min_quality_score = max(
            DEFAULT_QUALITY_GATE_MIN_QUALITY_SCORE,
            round(normalized_similarity_threshold * 0.8, 3),
        )
        min_pacing_score = DEFAULT_QUALITY_GATE_MIN_PACING_SCORE
        failures: List[str] = []

        try:
            safe_similarity = float(similarity_score)
        except (TypeError, ValueError):
            safe_similarity = -1.0
        try:
            safe_robotic = float(robotic_score)
        except (TypeError, ValueError):
            safe_robotic = 1.0
        try:
            safe_quality = float(quality_score)
        except (TypeError, ValueError):
            safe_quality = -1.0
        try:
            safe_pacing = None if pacing_score is None else float(pacing_score)
        except (TypeError, ValueError):
            safe_pacing = None

        if safe_similarity < normalized_similarity_threshold:
            failures.append(
                f"Similarity {safe_similarity:.2f} is below the {normalized_similarity_threshold:.2f} gate."
            )
        if safe_robotic > normalized_robotic_threshold:
            failures.append(
                f"Robotic score {safe_robotic:.2f} is above the {normalized_robotic_threshold:.2f} gate."
            )
        if safe_quality < min_quality_score:
            failures.append(
                f"Quality {safe_quality:.2f} is below the {min_quality_score:.2f} floor."
            )
        if safe_pacing is not None and safe_pacing < min_pacing_score:
            failures.append(
                f"Pacing {safe_pacing:.2f} is below the {min_pacing_score:.2f} floor."
            )

        return {
            "meets_quality_gate": not failures,
            "quality_gate_failures": failures,
        }

    @classmethod
    def _apply_quality_gate_metadata(
        cls,
        version_result: Dict[str, Any],
        *,
        similarity_threshold: float,
        robotic_threshold: float,
    ) -> Dict[str, Any]:
        """Annotate a version payload with quality-gate metadata."""
        gate_result = cls._evaluate_quality_gate(
            similarity_score=version_result.get("similarity_score", -1.0),
            robotic_score=version_result.get("robotic_score", 1.0),
            quality_score=version_result.get("quality_score", -1.0),
            pacing_score=version_result.get("pacing_score"),
            similarity_threshold=similarity_threshold,
            robotic_threshold=robotic_threshold,
        )
        version_result.update(gate_result)
        return version_result

    @staticmethod
    def _build_version_result(
        *,
        audio_path: str,
        similarity_score: float,
        robotic_score: float,
        quality_score: float,
        speaker_filename: str,
        text: str,
        emotion_vectors: Optional[List[float]],
        emotion_control_method: Optional[int],
        emotion_reference_filename: Optional[str],
        emotion_weight: Optional[float],
        emotion_text: Optional[str],
        seed: Optional[int],
        seed_origin: str,
        seed_strategy: str,
        delivery_rate: float,
        duration_seconds: Optional[float],
        expected_duration_seconds: Optional[float],
        pacing_score: Optional[float],
        pacing_label: Optional[str],
        pacing_notes: Optional[List[str]],
        review_score: Optional[float],
    ) -> Dict[str, Any]:
        """Create a consistent version result payload."""
        return {
            "audio_path": audio_path,
            "audio_filename": Path(audio_path).name,
            "similarity_score": similarity_score,
            "robotic_score": robotic_score,
            "quality_score": quality_score,
            "speaker_filename": speaker_filename,
            "text": text,
            "emotion_vectors": emotion_vectors or [],
            "emotion_control_method": emotion_control_method,
            "emotion_reference_filename": emotion_reference_filename,
            "emotion_weight": emotion_weight,
            "emotion_text": emotion_text,
            "seed": seed,
            "seed_origin": seed_origin,
            "seed_strategy": seed_strategy,
            "delivery_rate": delivery_rate,
            "duration_seconds": duration_seconds,
            "expected_duration_seconds": expected_duration_seconds,
            "pacing_score": pacing_score,
            "pacing_label": pacing_label,
            "pacing_notes": pacing_notes or [],
            "review_score": review_score,
        }

    def _generate_unique_random_seed(
        self,
        used_seeds: Optional[set],
        rng: Optional[random.Random] = None,
    ) -> int:
        """Generate a unique 32-bit seed for the current run."""
        generator = rng or random.SystemRandom()
        while True:
            candidate = generator.randrange(SEED_MODULUS)
            if used_seeds is None or candidate not in used_seeds:
                if used_seeds is not None:
                    used_seeds.add(candidate)
                return candidate

    def _resolve_line_seeds(
        self,
        *,
        seed_strategy: Optional[str],
        num_versions: int,
        line_index: int,
        resolved_base_seed: Optional[int],
        reused_seed_list: Optional[List[int]],
        used_seeds: Optional[set],
    ) -> List[int]:
        """Resolve the version seeds for one line of conversation output."""
        normalized_strategy = self._normalize_seed_strategy(seed_strategy)
        normalized_num_versions = max(1, int(num_versions))
        normalized_base_seed = (
            None if resolved_base_seed is None else self._normalize_seed_value(resolved_base_seed)
        )
        normalized_reused_seed_list = [
            self._normalize_seed_value(seed)
            for seed in (reused_seed_list or [])[:normalized_num_versions]
        ]

        if normalized_strategy == "fully_random":
            return [
                self._generate_unique_random_seed(used_seeds)
                for _ in range(normalized_num_versions)
            ]

        if normalized_strategy in {"random_base_sequential", "fixed_base_sequential"}:
            if normalized_base_seed is None:
                normalized_base_seed = self._generate_unique_random_seed(None)
            line_offset = line_index * normalized_num_versions * 10
            return [
                (normalized_base_seed + line_offset + version_index) % SEED_MODULUS
                for version_index in range(normalized_num_versions)
            ]

        if len(normalized_reused_seed_list) == normalized_num_versions:
            return normalized_reused_seed_list

        fallback_base_seed = self._normalize_seed_value(normalized_base_seed)
        return [
            (fallback_base_seed + version_index) % SEED_MODULUS
            for version_index in range(normalized_num_versions)
        ]
    
    def generate_conversation(
        self,
        parsed_script: List[Dict[str, str]],
        versions_per_line: int,
        similarity_threshold: float,
        robotic_threshold: float,  # New parameter
        auto_regen_attempts: int,
        emo_control_method: int,
        emo_ref_path: Optional[str],
        emo_weight: float,
        emo_random: bool,
        vec1: float, vec2: float, vec3: float, vec4: float,
        vec5: float, vec6: float, vec7: float, vec8: float,
        emo_text: str,
        do_sample_convo: bool,
        top_p_convo: float,
        top_k_convo: int,
        temperature_convo: float,
        length_penalty_convo: float,
        num_beams_convo: int,
        repetition_penalty_convo: float,
        max_mel_tokens_convo: int,
        max_text_tokens_per_segment_convo: int,
        speaker_pacing: Optional[List[Dict[str, Any]]] = None,
        scene_pacing_profile: str = "balanced",
        seed_strategy: str = "fully_random",
        fixed_base_seed: Optional[int] = DEFAULT_FIXED_BASE_SEED,
        resolved_base_seed: Optional[int] = None,
        reused_seed_list: Optional[List[int]] = None,
        progress: Optional[Any] = None
    ) -> Generator[Tuple[str, str, int, Optional[List[Dict]], Optional[str]], None, None]:
        """
        Generate conversation based on parsed script with emotion control.
        
        Args:
            parsed_script: Parsed conversation script
            versions_per_line: Number of versions per line
            similarity_threshold: Similarity threshold for auto-regeneration
            robotic_threshold: Robotic speech threshold (0.0-1.0)
            auto_regen_attempts: Number of auto-regeneration attempts
            emo_control_method: Emotion control method
            emo_ref_path: Emotion reference audio path
            emo_weight: Emotion weight
            emo_random: Whether to use random sampling
            vec1-vec8: Emotion vector components
            emo_text: Emotion description text
            **kwargs: Additional generation parameters
            progress: Progress callback
        
        Yields:
            Tuple: (status_log, progress_bar_html, progress_value, line_results, final_output_path)
        """
        from .file_utils import prepare_temp_dir
        from .audio_processing import analyze_speaker_similarity, speaker_similarity_model, analyze_speaker_similarity_with_quality
        
        if not parsed_script:
            yield "Please parse the script first!", "", 0, None, None
            return
        
        # Convert versions_per_line to integer
        try:
            num_versions = int(versions_per_line)
        except (ValueError, TypeError):
            yield f"Error: Invalid number of versions '{versions_per_line}'", "", 0, None, None
            return
        
        # Prepare temp directory
        if not prepare_temp_dir(self.temp_dir):
            yield f"Error: Failed to prepare temp directory {self.temp_dir}", "", 0, None, None
            return
        
        status_log = [f"Using temp directory: {self.temp_dir}"]
        yield "\n".join(status_log), self._update_progress_bar(0), 0, None, None
        
        # Process emotion parameters
        emo_vector, emo_ref_path, emo_weight = self._process_emotion_control(
            emo_control_method, emo_ref_path, emo_weight, 
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8
        )
        
        if emo_text == "" or emo_text is None:
            # For emotion from text method, if no emotion text is provided,
            # we'll use the main text for emotion analysis
            if emo_control_method == 3:
                # We'll handle this in the infer function by using the main text
                pass
            else:
                emo_text = None
        
        status_log.append("Starting Conversation Generation...")
        yield "\n".join(status_log), self._update_progress_bar(0), 0, None, None
        
        results = []
        total_lines = len(parsed_script)
        normalized_seed_strategy = self._normalize_seed_strategy(seed_strategy)
        normalized_fixed_base_seed = self._normalize_seed_value(fixed_base_seed)
        normalized_resolved_base_seed = (
            None if resolved_base_seed is None else self._normalize_seed_value(resolved_base_seed)
        )
        normalized_reused_seed_list = [
            self._normalize_seed_value(seed)
            for seed in (reused_seed_list or [])[:num_versions]
        ]
        used_generation_seeds: set = set()
        speaker_pacing_map = build_speaker_pacing_map(speaker_pacing)
        
        # Reset stop flag at start of generation
        self.reset_stop_flag()
        
        # Process each line automatically
        iterator = parsed_script
            
        for i, line_data in enumerate(iterator):
            line_work_units = max(1, int(num_versions) * max(1, int(auto_regen_attempts) + 1))
            completed_work_units = 0
            line_progress = self._calculate_line_progress(
                i,
                total_lines,
                completed_work_units=0,
                total_work_units=line_work_units
            )
        
            # Check if generation should be stopped
            if self._should_stop:
                status_log.append("\n🛑 Generation stopped by user request.")
                self.cleanup_memory()
                yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
                return
            speaker_file = line_data['speaker_filename']
            text = line_data['text']
            (
                line_emo_control_method,
                line_emo_ref_path,
                line_emo_weight,
                line_emo_vector,
                line_emo_text,
            ) = self._resolve_line_emotion_settings(
                line_data,
                emo_control_method,
                emo_ref_path,
                emo_weight,
                emo_vector,
                emo_text,
            )
            
            status_log.append(f"\nGenerating Line {i+1}/{total_lines} ({speaker_file})...")
            yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
            
            line_success = False
            best_score = -1.0
            delivery_rate = resolve_speaker_delivery_rate(speaker_file, speaker_pacing_map)
            line_version_seeds = self._resolve_line_seeds(
                seed_strategy=normalized_seed_strategy,
                num_versions=num_versions,
                line_index=i,
                resolved_base_seed=(
                    normalized_fixed_base_seed
                    if normalized_resolved_base_seed is None
                    else normalized_resolved_base_seed
                ),
                reused_seed_list=normalized_reused_seed_list,
                used_seeds=used_generation_seeds,
            )
            
            # Try multiple versions for this line
            for attempt in range(num_versions):
                try:
                    current_seed = line_version_seeds[attempt]
                    # Create unique output path for this attempt
                    output_filename = f"line{i:03d}_spk-{Path(speaker_file).stem}_v{attempt+1:02d}.wav"
                    audio_path = str(self.temp_dir / output_filename)
                    self._generated_files.add(audio_path)
                    
                    # Generate audio for this line - use full path to speaker file
                    speaker_full_path = str(self.speakers_dir / speaker_file)
                    print(f"DEBUG: Using speaker path: {speaker_full_path}")
                    # Handle both direct TTS instance and wrapped TTS
                    tts_instance = getattr(self.tts_core, 'tts', None) or self.tts_core
                    infer_kwargs = {
                        "verbose": False,
                        "max_text_tokens_per_segment": int(max_text_tokens_per_segment_convo),
                        "use_random": emo_random,
                        "emo_audio_prompt": line_emo_ref_path,
                        "emo_alpha": line_emo_weight,
                        "emo_vector": line_emo_vector,
                        "use_emo_text": (line_emo_control_method == 3),
                        "emo_text": line_emo_text,
                        "do_sample": bool(do_sample_convo),
                        "top_p": float(top_p_convo),
                        "top_k": int(top_k_convo) if int(top_k_convo) > 0 else None,
                        "temperature": float(temperature_convo),
                        "length_penalty": float(length_penalty_convo),
                        "num_beams": int(num_beams_convo),
                        "repetition_penalty": float(repetition_penalty_convo),
                        "max_mel_tokens": int(max_mel_tokens_convo),
                        "seed": current_seed,
                    }
                    self._infer_with_live_progress(
                        tts_instance,
                        (speaker_full_path, text, audio_path),
                        infer_kwargs,
                        progress_callback=progress,
                        line_index=i,
                        total_lines=total_lines,
                        speaker_file=speaker_file,
                        completed_work_units=completed_work_units,
                        total_work_units=line_work_units,
                    )
                    if abs(delivery_rate - 1.0) > 0.005:
                        apply_delivery_rate_to_file(audio_path, delivery_rate)
                    
                    # Calculate enhanced quality score with robotic detection
                    quality_result = analyze_speaker_similarity_with_quality(
                        speaker_similarity_model, speaker_full_path, audio_path
                    )
                    similarity_score = quality_result['similarity']
                    robotic_score = quality_result['robotic_score']
                    quality_score = quality_result['quality_score']
                    pacing_result = assess_line_pacing(
                        text,
                        audio_path,
                        delivery_rate=delivery_rate,
                        scene_pacing_profile=scene_pacing_profile,
                        quality_score=quality_score,
                    )
                    completed_work_units += 1
                    line_progress = self._calculate_line_progress(
                        i,
                        total_lines,
                        completed_work_units=completed_work_units,
                        total_work_units=line_work_units
                    )
                    
                    status_log.append(f"  Attempt {attempt+1}: Similarity {similarity_score:.2f}, Robotic {robotic_score:.2f}, Quality {quality_score:.2f}")
                    yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
                    
                    if quality_score > best_score:
                        best_score = quality_score
                    
                    version_result = self._apply_quality_gate_metadata(
                        self._build_version_result(
                            audio_path=audio_path,
                            similarity_score=similarity_score,
                            robotic_score=robotic_score,
                            quality_score=quality_score,
                            speaker_filename=speaker_file,
                            text=text,
                            emotion_vectors=line_emo_vector,
                            emotion_control_method=line_emo_control_method,
                            emotion_reference_filename=line_emo_ref_path,
                            emotion_weight=line_emo_weight,
                            emotion_text=line_emo_text,
                            seed=current_seed,
                            seed_origin="initial",
                            seed_strategy=normalized_seed_strategy,
                            delivery_rate=delivery_rate,
                            duration_seconds=pacing_result["duration_seconds"],
                            expected_duration_seconds=pacing_result["expected_duration_seconds"],
                            pacing_score=pacing_result["pacing_score"],
                            pacing_label=pacing_result["pacing_label"],
                            pacing_notes=pacing_result["pacing_notes"],
                            review_score=pacing_result["review_score"],
                        ),
                        similarity_threshold=similarity_threshold,
                        robotic_threshold=robotic_threshold,
                    )

                    # Always add the result to the list, regardless of threshold
                    results.append(version_result)
                    
                    # Check if quality meets threshold (considering both similarity and robotic score)
                    if version_result["meets_quality_gate"]:
                        line_success = True
                        status_log.append(f"  ✅ Version {attempt+1} meets quality threshold (similarity: {similarity_score:.2f}, robotic: {robotic_score:.2f})")
                        yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
                        # Don't break - continue generating all versions
                    else:
                        status_log.append(f"  ⚠️ Version {attempt+1} below threshold (similarity: {similarity_score:.2f}, robotic: {robotic_score:.2f})")
                        yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
                        
                        # Auto-regenerate if below threshold
                        if auto_regen_attempts > 0:
                            status_log.append(f"  🔄 Auto-regenerating version {attempt+1}...")
                            yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
                            
                            for regen_attempt in range(auto_regen_attempts):
                                try:
                                    regen_seed = self._generate_unique_random_seed(used_generation_seeds)
                                    regen_filename = f"line{i:03d}_spk-{Path(speaker_file).stem}_v{attempt+1:02d}_regen{regen_attempt+1}.wav"
                                    regen_audio_path = str(self.temp_dir / regen_filename)
                                    self._generated_files.add(regen_audio_path)
                                    
                                    # Preserve the caller's sampling mode so retries do not
                                    # drift further away from the reference voice.
                                    tts_instance = getattr(self.tts_core, 'tts', None) or self.tts_core
                                    regen_infer_kwargs = {
                                        "verbose": False,
                                        "max_text_tokens_per_segment": int(max_text_tokens_per_segment_convo),
                                        "use_random": emo_random,
                                        "emo_audio_prompt": line_emo_ref_path,
                                        "emo_alpha": line_emo_weight,
                                        "emo_vector": line_emo_vector,
                                        "use_emo_text": (line_emo_control_method == 3),
                                        "emo_text": line_emo_text,
                                        "do_sample": bool(do_sample_convo),
                                        "top_p": float(top_p_convo),
                                        "top_k": int(top_k_convo) if int(top_k_convo) > 0 else None,
                                        "temperature": float(temperature_convo),
                                        "length_penalty": float(length_penalty_convo),
                                        "num_beams": int(num_beams_convo),
                                        "repetition_penalty": float(repetition_penalty_convo),
                                        "max_mel_tokens": int(max_mel_tokens_convo),
                                        "seed": regen_seed,
                                    }
                                    self._infer_with_live_progress(
                                        tts_instance,
                                        (speaker_full_path, text, regen_audio_path),
                                        regen_infer_kwargs,
                                        progress_callback=progress,
                                        line_index=i,
                                        total_lines=total_lines,
                                        speaker_file=speaker_file,
                                        completed_work_units=completed_work_units,
                                        total_work_units=line_work_units,
                                        prefix="Regenerating Line"
                                    )
                                    if abs(delivery_rate - 1.0) > 0.005:
                                        apply_delivery_rate_to_file(regen_audio_path, delivery_rate)
                                    
                                    # Check new quality with robotic detection
                                    regen_quality = analyze_speaker_similarity_with_quality(
                                        speaker_similarity_model, speaker_full_path, regen_audio_path
                                    )
                                    regen_similarity = regen_quality['similarity']
                                    regen_robotic = regen_quality['robotic_score']
                                    regen_quality_score = regen_quality['quality_score']
                                    regen_pacing = assess_line_pacing(
                                        text,
                                        regen_audio_path,
                                        delivery_rate=delivery_rate,
                                        scene_pacing_profile=scene_pacing_profile,
                                        quality_score=regen_quality_score,
                                    )
                                    completed_work_units += 1
                                    line_progress = self._calculate_line_progress(
                                        i,
                                        total_lines,
                                        completed_work_units=completed_work_units,
                                        total_work_units=line_work_units
                                    )
                                    
                                    status_log.append(f"    Regen attempt {regen_attempt+1}: Similarity {regen_similarity:.2f}, Robotic {regen_robotic:.2f}, Quality {regen_quality_score:.2f}")
                                    yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
                                    
                                    if regen_quality_score > quality_score:
                                        regen_version_result = self._apply_quality_gate_metadata(
                                            self._build_version_result(
                                                audio_path=regen_audio_path,
                                                similarity_score=regen_similarity,
                                                robotic_score=regen_robotic,
                                                quality_score=regen_quality_score,
                                                speaker_filename=speaker_file,
                                                text=text,
                                                emotion_vectors=line_emo_vector,
                                                emotion_control_method=line_emo_control_method,
                                                emotion_reference_filename=line_emo_ref_path,
                                                emotion_weight=line_emo_weight,
                                                emotion_text=line_emo_text,
                                                seed=regen_seed,
                                                seed_origin="auto_regen",
                                                seed_strategy=normalized_seed_strategy,
                                                delivery_rate=delivery_rate,
                                                duration_seconds=regen_pacing["duration_seconds"],
                                                expected_duration_seconds=regen_pacing["expected_duration_seconds"],
                                                pacing_score=regen_pacing["pacing_score"],
                                                pacing_label=regen_pacing["pacing_label"],
                                                pacing_notes=regen_pacing["pacing_notes"],
                                                review_score=regen_pacing["review_score"],
                                            ),
                                            similarity_threshold=similarity_threshold,
                                            robotic_threshold=robotic_threshold,
                                        )
                                        # Replace with better version
                                        results[-1] = regen_version_result
                                        similarity_score = regen_similarity
                                        robotic_score = regen_robotic
                                        quality_score = regen_quality_score
                                        status_log.append(f"    ✅ Improved to quality {regen_quality_score:.2f}")
                                        yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
                                        
                                        if regen_version_result["meets_quality_gate"]:
                                            line_success = True
                                            status_log.append(f"    ✅ Regen version meets quality threshold!")
                                            yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
                                            break
                                
                                except Exception as e:
                                    status_log.append(f"    ❌ Regen attempt {regen_attempt+1} failed: {str(e)}")
                                    yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
                        
                except Exception as e:
                    status_log.append(f"  ❌ Attempt {attempt+1} failed: {str(e)}")
                    yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
            
            line_progress = self._calculate_line_progress(i, total_lines, line_completed=True)
            # All versions are now collected, log the status
            if line_success:
                status_log.append(f"  ✅ Line completed with {num_versions} versions")
            else:
                status_log.append(f"  ⚠️ Line completed with {num_versions} versions (none met threshold)")
            yield "\n".join(status_log), self._update_progress_bar(line_progress), line_progress, None, None
        
        # Reorganize results by line with versions
        line_results = self._organize_results_by_line(results)
        
        # Update progress bar to 100% completion
        status_log.append(f"\n✅ Generation complete! Processed {len(line_results)} lines with {sum(len(line['versions']) for line in line_results)} total versions.")
        
        # Don't automatically concatenate - let the user choose when to concatenate
        final_output_path = None
        status_log.append(f"\n✅ Generation complete! Use the buttons below to concatenate selected versions.")
        
        self.cleanup_memory()
        yield "\n".join(status_log), self._update_progress_bar(100), 100, line_results, final_output_path
    
    def _process_emotion_control(
        self,
        emo_control_method: int,
        emo_ref_path: Optional[str],
        emo_weight: float,
        vec1: float, vec2: float, vec3: float, vec4: float,
        vec5: float, vec6: float, vec7: float, vec8: float
    ) -> Tuple[Optional[List[float]], Optional[str], float]:
        """Process emotion control parameters."""
        # Debug logging for emotion control
        print(f"DEBUG: _process_emotion_control called")
        print(f"DEBUG: emo_control_method: {emo_control_method}")
        print(f"DEBUG: emo_ref_path: {emo_ref_path}")
        print(f"DEBUG: emo_weight: {emo_weight}")
        print(f"DEBUG: vec1-vec8: [{vec1}, {vec2}, {vec3}, {vec4}, {vec5}, {vec6}, {vec7}, {vec8}]")
        
        emo_vector = None
        
        if emo_control_method == 0:  # emotion from speaker
            emo_ref_path = None
            emo_weight = 1.0
            print(f"DEBUG: Using emotion from speaker - emo_ref_path set to None, emo_weight set to 1.0")
        elif emo_control_method == 1:  # emotion from reference audio
            print(f"DEBUG: Using emotion from reference audio - keeping provided emo_ref_path and emo_weight")
            pass  # emo_weight remains as provided
        elif emo_control_method == 2:  # emotion from custom vectors
            emo_vector = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            print(f"DEBUG: Using emotion from vectors - created emo_vector: {emo_vector}")
            print(f"DEBUG: Sum of emotion vectors: {sum(emo_vector)}")
            if sum(emo_vector) > 1.5:
                print(f"DEBUG: Emotion vector sum exceeds 1.5, raising ValueError")
                raise ValueError("The sum of emotion vectors cannot exceed 1.5")
        elif emo_control_method == 3:  # emotion from text
            print(f"DEBUG: Using emotion from text - keeping provided emo_text")
            pass
        else:
            print(f"DEBUG: Unknown emotion control method: {emo_control_method}")
        
        print(f"DEBUG: Returning emo_vector: {emo_vector}, emo_ref_path: {emo_ref_path}, emo_weight: {emo_weight}")
        return emo_vector, emo_ref_path, emo_weight

    def _resolve_line_emotion_settings(
        self,
        line_data: Dict[str, Any],
        default_emo_control_method: int,
        default_emo_ref_path: Optional[str],
        default_emo_weight: float,
        default_emo_vector: Optional[List[float]],
        default_emo_text: Optional[str]
    ) -> Tuple[int, Optional[str], float, Optional[List[float]], Optional[str]]:
        """
        Resolve the effective emotion settings for a single line.

        Line-level values take precedence so frontend timeline edits are not
        silently dropped when the conversation is generated.
        """
        line_emo_control_method = default_emo_control_method
        line_emo_ref_path = default_emo_ref_path
        line_emo_weight = default_emo_weight
        line_emo_vector = list(default_emo_vector) if default_emo_vector is not None else None
        line_emo_text = default_emo_text

        if line_data.get("emotion_weight") is not None:
            line_emo_weight = float(line_data["emotion_weight"])

        line_emotion_vectors = line_data.get("emotion_vectors") or line_data.get("emo_vector") or []
        if line_emotion_vectors:
            if len(line_emotion_vectors) > 8:
                raise ValueError("Maximum 8 emotion vector components allowed per line")
            padded_vectors = list(line_emotion_vectors[:8]) + [0.0] * max(0, 8 - len(line_emotion_vectors))
            if sum(abs(x) for x in padded_vectors) > 1.5:
                raise ValueError("The sum of per-line emotion vectors cannot exceed 1.5")
            return 2, None, line_emo_weight, padded_vectors, None

        line_control_method = line_data.get("emotion_control_method")
        if hasattr(line_control_method, "value"):
            line_control_method = line_control_method.value
        if isinstance(line_control_method, str):
            line_emo_control_method = {
                "from_speaker": 0,
                "from_reference": 1,
                "from_vectors": 2,
                "from_text": 3,
            }.get(line_control_method, line_emo_control_method)

        if line_data.get("emotion_reference_filename"):
            line_emo_ref_path = line_data["emotion_reference_filename"]
        if line_data.get("emotion_text") is not None:
            line_emo_text = line_data.get("emotion_text")

        return (
            line_emo_control_method,
            line_emo_ref_path,
            line_emo_weight,
            line_emo_vector,
            line_emo_text,
        )
    
    def _organize_results_by_line(self, results: List[Dict]) -> List[Dict]:
        """Organize results by line with versions."""
        line_results = []
        current_line_index = -1
        current_line_data = None
        
        for result in results:
            # Extract line number from filename
            filename = os.path.basename(result['audio_path'])
            if filename.startswith('line') and '_spk-' in filename:
                try:
                    line_num = int(filename.split('_')[0][4:])
                    if line_num != current_line_index:
                        # New line
                        if current_line_data:
                            line_results.append(current_line_data)
                        current_line_index = line_num
                        current_line_data = {
                            'line_index': line_num,
                            'speaker_filename': result['speaker_filename'],
                            'text': result['text'],
                            'emotion_vectors': result.get('emotion_vectors', []),
                            'emotion_control_method': result.get('emotion_control_method'),
                            'emotion_reference_filename': result.get('emotion_reference_filename'),
                            'emotion_weight': result.get('emotion_weight'),
                            'emotion_text': result.get('emotion_text'),
                            'versions': []
                        }
                    # Add this version to the current line
                    current_line_data['versions'].append(result)
                except (ValueError, IndexError):
                    # Fallback: treat each result as a separate line
                    line_results.append({
                        'line_index': len(line_results),
                        'speaker_filename': result['speaker_filename'],
                        'text': result['text'],
                        'emotion_vectors': result.get('emotion_vectors', []),
                        'emotion_control_method': result.get('emotion_control_method'),
                        'emotion_reference_filename': result.get('emotion_reference_filename'),
                        'emotion_weight': result.get('emotion_weight'),
                        'emotion_text': result.get('emotion_text'),
                        'versions': [result]
                    })
        
        # Add the last line
        if current_line_data:
            line_results.append(current_line_data)
        
        # If no proper line structure found, create one result per item
        if not line_results:
            for i, result in enumerate(results):
                line_results.append({
                    'line_index': i,
                    'speaker_filename': result.get('speaker_filename', 'Unknown'),
                    'text': result.get('text', 'No text'),
                    'emotion_vectors': result.get('emotion_vectors', []),
                    'emotion_control_method': result.get('emotion_control_method'),
                    'emotion_reference_filename': result.get('emotion_reference_filename'),
                    'emotion_weight': result.get('emotion_weight'),
                    'emotion_text': result.get('emotion_text'),
                    'versions': [result]
                })
        
        return line_results
    
    def _create_final_conversation(self, line_results: List[Dict]) -> str:
        """Create final conversation audio from best versions."""
        audio_segments = []
        
        for line_data in line_results:
            if line_data['versions']:
                # Use the best version for each line
                best_index = self._get_highest_score_index(line_data['versions'])
                best_audio = line_data['versions'][best_index]['audio_path']
                
                if os.path.exists(best_audio):
                    try:
                        audio_segments.append(AudioSegment.from_wav(best_audio))
                    except Exception as e:
                        print(f"Error loading {best_audio}: {e}")
                        continue
        
        if audio_segments:
            try:
                final_audio = sum(audio_segments)
                final_output_path = str(self.temp_dir / f"final_conversation_{uuid.uuid4().hex[:8]}.wav")
                final_audio.export(final_output_path, format="wav")
                return final_output_path
            except Exception as e:
                print(f"Error during concatenation/export: {e}")
                import traceback
                traceback.print_exc()
        
        return None
    
    def _get_highest_score_index(self, line_results: List[Dict]) -> int:
        """Find the index of the highest quality result for a line."""
        if not line_results:
            return 0

        gated_indices = [
            index for index, result in enumerate(line_results)
            if result.get("meets_quality_gate", False)
        ]
        candidate_indices = gated_indices or list(range(len(line_results)))
        highest_index = candidate_indices[0]
        highest_score = -1.0

        for i in candidate_indices:
            result = line_results[i]
            # Use quality_score if available, otherwise fallback to similarity_score
            score = result.get('quality_score', result.get('similarity_score', -1.0))
            if score > highest_score:
                highest_score = score
                highest_index = i
        
        return highest_index

    def _calculate_line_progress(
        self,
        line_index: int,
        total_lines: int,
        completed_work_units: float = 0.0,
        total_work_units: float = 1.0,
        line_completed: bool = False
    ) -> float:
        """Return smoother per-line progress so long generations don't look frozen."""
        safe_total_lines = max(1, int(total_lines))
        line_start = (line_index / safe_total_lines) * 100.0
        line_end = ((line_index + 1) / safe_total_lines) * 100.0

        if line_completed:
            return round(line_end, 2)

        safe_total_units = max(1.0, float(total_work_units))
        safe_completed_units = max(0.0, min(float(completed_work_units), safe_total_units))
        line_span = max(0.0, line_end - line_start)
        working_span = line_span * 0.9
        base_offset = min(max(line_span * 0.08, 1.0), 4.0)

        progress = line_start + base_offset
        if working_span > base_offset:
            progress += (safe_completed_units / safe_total_units) * (working_span - base_offset)

        return round(min(line_end - 0.1, progress), 2)

    def _emit_live_progress(
        self,
        progress_callback,
        progress_value: float,
        step_message: str
    ) -> None:
        """Safely report live progress updates to the caller."""
        if not callable(progress_callback):
            return

        try:
            bounded_progress = max(0.0, min(99.5, float(progress_value)))
            progress_callback(bounded_progress, step_message)
        except Exception as progress_error:
            print(f"DEBUG: Live progress callback failed: {progress_error}")

    def _build_generation_step_message(
        self,
        line_index: int,
        total_lines: int,
        speaker_file: str,
        desc: Optional[str] = None,
        prefix: str = "Generating Line"
    ) -> str:
        """Build a user-facing step label for line generation progress."""
        step_message = f"{prefix} {line_index + 1}/{max(1, int(total_lines))} ({speaker_file})"
        if desc:
            step_message = f"{step_message} - {desc}"
        return step_message

    def _infer_with_live_progress(
        self,
        tts_instance,
        infer_args: Tuple[Any, ...],
        infer_kwargs: Dict[str, Any],
        *,
        progress_callback,
        line_index: int,
        total_lines: int,
        speaker_file: str,
        completed_work_units: float,
        total_work_units: float,
        prefix: str = "Generating Line"
    ):
        """
        Run a TTS inference call while forwarding model-level progress updates
        as conversation-level progress percentages.
        """
        previous_callback = getattr(tts_instance, "progress_callback", None)
        supports_callback = hasattr(tts_instance, "progress_callback")

        def _handle_infer_progress(value, desc=None):
            try:
                intra_attempt_progress = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                intra_attempt_progress = 0.0

            absolute_progress = self._calculate_line_progress(
                line_index,
                total_lines,
                completed_work_units=float(completed_work_units) + intra_attempt_progress,
                total_work_units=total_work_units
            )
            step_message = self._build_generation_step_message(
                line_index,
                total_lines,
                speaker_file,
                desc=desc,
                prefix=prefix
            )
            self._emit_live_progress(progress_callback, absolute_progress, step_message)

        if supports_callback:
            tts_instance.progress_callback = _handle_infer_progress

        self._emit_live_progress(
            progress_callback,
            self._calculate_line_progress(
                line_index,
                total_lines,
                completed_work_units=completed_work_units,
                total_work_units=total_work_units
            ),
            self._build_generation_step_message(
                line_index,
                total_lines,
                speaker_file,
                desc="starting inference...",
                prefix=prefix
            )
        )

        try:
            return tts_instance.infer(*infer_args, **infer_kwargs)
        finally:
            if supports_callback:
                tts_instance.progress_callback = previous_callback
    
    def _update_progress_bar(self, progress_percent: float) -> str:
        """Generate HTML for progress bar."""
        return f"""
        <div id="progress-container" style="width: 100%; background: #f0f0f0; border-radius: 5px; margin: 10px 0;">
            <div id="progress-bar" style="width: {progress_percent}%; height: 20px; background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); border-radius: 5px; transition: width 0.3s ease;">
                <div style="text-align: center; color: white; font-weight: bold; line-height: 20px;">{progress_percent}%</div>
            </div>
        </div>
        """
    
    def regenerate_line(
        self,
        line_index: int,
        line_data: Dict[str, Any],
        num_versions: int,
        similarity_threshold: float,
        robotic_threshold: float,
        auto_regen_attempts: int,
        emo_control_method: int,
        emo_ref_path: Optional[str],
        emo_weight: float,
        emo_random: bool,
        vec1: float, vec2: float, vec3: float, vec4: float,
        vec5: float, vec6: float, vec7: float, vec8: float,
        emo_text: str,
        do_sample_convo: bool,
        top_p_convo: float,
        top_k_convo: int,
        temperature_convo: float,
        length_penalty_convo: float,
        num_beams_convo: int,
        repetition_penalty_convo: float,
        max_mel_tokens_convo: int,
        max_text_tokens_per_segment_convo: int,
        speaker_pacing: Optional[List[Dict[str, Any]]] = None,
        seed_strategy: str = "fully_random",
        fixed_base_seed: Optional[int] = DEFAULT_FIXED_BASE_SEED,
        resolved_base_seed: Optional[int] = None,
        reused_seed_list: Optional[List[int]] = None,
        progress: Optional[Any] = None
    ) -> Generator[Tuple[str, str, int, Optional[List[Dict]]], None, None]:
        """
        Regenerate a specific line with new versions.
        
        Args:
            line_index: Index of the line to regenerate
            line_data: Original line data containing speaker and text
            num_versions: Number of new versions to generate
            similarity_threshold: Similarity threshold for auto-regeneration
            robotic_threshold: Robotic speech threshold (0.0-1.0)
            auto_regen_attempts: Number of auto-regeneration attempts
            emo_control_method: Emotion control method
            emo_ref_path: Emotion reference audio path
            emo_weight: Emotion weight
            emo_random: Whether to use random sampling
            vec1-vec8: Emotion vector components
            emo_text: Emotion description text
            do_sample_convo: Whether to perform sampling
            top_p_convo: Top-p sampling parameter
            top_k_convo: Top-k sampling parameter
            temperature_convo: Temperature parameter
            length_penalty_convo: Length penalty parameter
            num_beams_convo: Number of beams
            repetition_penalty_convo: Repetition penalty parameter
            max_mel_tokens_convo: Maximum mel tokens
            max_text_tokens_per_segment_convo: Maximum text tokens per segment
            progress: Progress callback
        
        Yields:
            Tuple: (status_log, progress_bar_html, progress_value, new_versions)
        """
        from .file_utils import prepare_temp_dir
        from .audio_processing import analyze_speaker_similarity_with_quality, speaker_similarity_model
        
        status_log = [f"Regenerating Line {line_index + 1}..."]
        yield "\n".join(status_log), self._update_progress_bar(0), 0, None
        
        # Process emotion parameters
        emo_vector, emo_ref_path, emo_weight = self._process_emotion_control(
            emo_control_method, emo_ref_path, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8
        )
        
        if emo_text == "" or emo_text is None:
            if emo_control_method == 3:
                pass
            else:
                emo_text = None
        
        speaker_file = line_data['speaker_filename']
        text = line_data['text']
        (
            line_emo_control_method,
            line_emo_ref_path,
            line_emo_weight,
            line_emo_vector,
            line_emo_text,
        ) = self._resolve_line_emotion_settings(
            line_data,
            emo_control_method,
            emo_ref_path,
            emo_weight,
            emo_vector,
            emo_text,
        )
        
        status_log.append(f"Speaker: {speaker_file}")
        status_log.append(f"Text: {text}")
        yield "\n".join(status_log), self._update_progress_bar(10), 10, None
        
        new_versions = []
        regen_run_id = uuid.uuid4().hex[:8]
        normalized_seed_strategy = self._normalize_seed_strategy(seed_strategy)
        normalized_fixed_base_seed = self._normalize_seed_value(fixed_base_seed)
        normalized_resolved_base_seed = (
            None if resolved_base_seed is None else self._normalize_seed_value(resolved_base_seed)
        )
        normalized_reused_seed_list = [
            self._normalize_seed_value(seed)
            for seed in (reused_seed_list or [])[:num_versions]
        ]
        used_generation_seeds: set = set()
        speaker_pacing_map = build_speaker_pacing_map(speaker_pacing)
        delivery_rate = resolve_speaker_delivery_rate(speaker_file, speaker_pacing_map)
        line_version_seeds = self._resolve_line_seeds(
            seed_strategy=normalized_seed_strategy,
            num_versions=num_versions,
            line_index=line_index,
            resolved_base_seed=(
                normalized_fixed_base_seed
                if normalized_resolved_base_seed is None
                else normalized_resolved_base_seed
            ),
            reused_seed_list=normalized_reused_seed_list,
            used_seeds=used_generation_seeds,
        )
        
        # Generate new versions
        print(f"DEBUG: regenerate_line called with num_versions={num_versions}")
        status_log.append(f"DEBUG: Will generate {num_versions} versions")
        yield "\n".join(status_log), self._update_progress_bar(10), 10, None
        
        for attempt in range(num_versions):
            print(f"DEBUG: Starting regeneration attempt {attempt+1}/{num_versions}")
            status_log.append(f"DEBUG: Starting regeneration attempt {attempt+1}/{num_versions}")
            yield "\n".join(status_log), self._update_progress_bar(10), 10, None
            try:
                current_seed = line_version_seeds[attempt]
                # Create unique output path for this attempt
                output_filename = (
                    f"line{line_index:03d}_spk-{Path(speaker_file).stem}_"
                    f"regen{attempt+1:02d}_{regen_run_id}.wav"
                )
                audio_path = str(self.temp_dir / output_filename)
                self._generated_files.add(audio_path)
                
                # Update progress
                progress_percent = 10 + (attempt / num_versions) * 70
                status_log.append(f"Generating version {attempt+1}/{num_versions}...")
                yield "\n".join(status_log), self._update_progress_bar(progress_percent), progress_percent, None
                
                # Generate audio for this line - use full path to speaker file
                speaker_full_path = str(self.speakers_dir / speaker_file)
                print(f"DEBUG: Using speaker path for regeneration: {speaker_full_path}")
                # Handle both direct TTS instance and wrapped TTS
                tts_instance = getattr(self.tts_core, 'tts', None) or self.tts_core
                infer_kwargs = {
                    "verbose": False,
                    "max_text_tokens_per_segment": int(max_text_tokens_per_segment_convo),
                    "use_random": emo_random,
                    "emo_audio_prompt": line_emo_ref_path,
                    "emo_alpha": line_emo_weight,
                    "emo_vector": line_emo_vector,
                    "use_emo_text": (line_emo_control_method == 3),
                    "emo_text": line_emo_text,
                    "do_sample": bool(do_sample_convo),
                    "top_p": float(top_p_convo),
                    "top_k": int(top_k_convo) if int(top_k_convo) > 0 else None,
                    "temperature": float(temperature_convo),
                    "length_penalty": float(length_penalty_convo),
                    "num_beams": int(num_beams_convo),
                    "repetition_penalty": float(repetition_penalty_convo),
                    "max_mel_tokens": int(max_mel_tokens_convo),
                    "seed": current_seed,
                }
                self._infer_with_live_progress(
                    tts_instance,
                    (speaker_full_path, text, audio_path),
                    infer_kwargs,
                    progress_callback=progress,
                    line_index=line_index,
                    total_lines=1,
                    speaker_file=speaker_file,
                    completed_work_units=float(attempt),
                    total_work_units=max(1.0, float(num_versions)),
                    prefix="Regenerating Line"
                )
                if abs(delivery_rate - 1.0) > 0.005:
                    apply_delivery_rate_to_file(audio_path, delivery_rate)
                
                # Calculate enhanced quality score with robotic detection
                quality_result = analyze_speaker_similarity_with_quality(
                    speaker_similarity_model, speaker_full_path, audio_path
                )
                similarity_score = quality_result['similarity']
                robotic_score = quality_result['robotic_score']
                quality_score = quality_result['quality_score']
                
                status_log.append(f"  Version {attempt+1}: Similarity {similarity_score:.2f}, Robotic {robotic_score:.2f}, Quality {quality_score:.2f}")
                yield "\n".join(status_log), self._update_progress_bar(progress_percent), progress_percent, None
                
                # Add the result to the list
                new_versions.append(self._build_version_result(
                    audio_path=audio_path,
                    similarity_score=similarity_score,
                    robotic_score=robotic_score,
                    quality_score=quality_score,
                    speaker_filename=speaker_file,
                    text=text,
                    emotion_vectors=line_emo_vector,
                    emotion_control_method=line_emo_control_method,
                    emotion_reference_filename=line_emo_ref_path,
                    emotion_weight=line_emo_weight,
                    emotion_text=line_emo_text,
                    seed=current_seed,
                    seed_origin="manual_regen",
                    seed_strategy=normalized_seed_strategy,
                    delivery_rate=delivery_rate,
                ))
                
                # Check if quality meets threshold
                if similarity_score >= similarity_threshold and robotic_score <= robotic_threshold:
                    status_log.append(f"  ✅ Version {attempt+1} meets quality threshold!")
                else:
                    status_log.append(f"  ⚠️ Version {attempt+1} below threshold (similarity: {similarity_score:.2f}, robotic: {robotic_score:.2f})")
                    
                    # Auto-regenerate if below threshold
                    if auto_regen_attempts > 0:
                        status_log.append(f"  🔄 Auto-regenerating version {attempt+1}...")
                        yield "\n".join(status_log), self._update_progress_bar(progress_percent), progress_percent, None
                        
                        for regen_attempt in range(auto_regen_attempts):
                            try:
                                regen_seed = self._generate_unique_random_seed(used_generation_seeds)
                                regen_filename = (
                                    f"line{line_index:03d}_spk-{Path(speaker_file).stem}_"
                                    f"regen{attempt+1:02d}_{regen_run_id}_auto{regen_attempt+1}.wav"
                                )
                                regen_audio_path = str(self.temp_dir / regen_filename)
                                self._generated_files.add(regen_audio_path)
                                
                                # Preserve the caller's sampling mode so retries do not
                                # drift further away from the reference voice.
                                tts_instance = getattr(self.tts_core, 'tts', None) or self.tts_core
                                regen_infer_kwargs = {
                                    "verbose": False,
                                    "max_text_tokens_per_segment": int(max_text_tokens_per_segment_convo),
                                    "use_random": emo_random,
                                    "emo_audio_prompt": line_emo_ref_path,
                                    "emo_alpha": line_emo_weight,
                                    "emo_vector": line_emo_vector,
                                    "use_emo_text": (line_emo_control_method == 3),
                                    "emo_text": line_emo_text,
                                    "do_sample": bool(do_sample_convo),
                                    "top_p": float(top_p_convo),
                                    "top_k": int(top_k_convo) if int(top_k_convo) > 0 else None,
                                    "temperature": float(temperature_convo),
                                    "length_penalty": float(length_penalty_convo),
                                    "num_beams": int(num_beams_convo),
                                    "repetition_penalty": float(repetition_penalty_convo),
                                    "max_mel_tokens": int(max_mel_tokens_convo),
                                    "seed": regen_seed,
                                }
                                self._infer_with_live_progress(
                                    tts_instance,
                                    (speaker_full_path, text, regen_audio_path),
                                    regen_infer_kwargs,
                                    progress_callback=progress,
                                    line_index=line_index,
                                    total_lines=1,
                                    speaker_file=speaker_file,
                                    completed_work_units=float(attempt + regen_attempt + 1),
                                    total_work_units=max(1.0, float(num_versions + auto_regen_attempts)),
                                    prefix="Regenerating Line"
                                )
                                if abs(delivery_rate - 1.0) > 0.005:
                                    apply_delivery_rate_to_file(regen_audio_path, delivery_rate)
                                
                                # Check new quality with robotic detection
                                regen_quality = analyze_speaker_similarity_with_quality(
                                    speaker_similarity_model, speaker_full_path, regen_audio_path
                                )
                                regen_similarity = regen_quality['similarity']
                                regen_robotic = regen_quality['robotic_score']
                                regen_quality_score = regen_quality['quality_score']
                                
                                status_log.append(f"    Auto-regen attempt {regen_attempt+1}: Similarity {regen_similarity:.2f}, Robotic {regen_robotic:.2f}, Quality {regen_quality_score:.2f}")
                                yield "\n".join(status_log), self._update_progress_bar(progress_percent), progress_percent, None
                                
                                if regen_quality_score > quality_score:
                                    # Replace with better version
                                    new_versions[-1] = self._build_version_result(
                                        audio_path=regen_audio_path,
                                        similarity_score=regen_similarity,
                                        robotic_score=regen_robotic,
                                        quality_score=regen_quality_score,
                                        speaker_filename=speaker_file,
                                        text=text,
                                        emotion_vectors=line_emo_vector,
                                        emotion_control_method=line_emo_control_method,
                                        emotion_reference_filename=line_emo_ref_path,
                                        emotion_weight=line_emo_weight,
                                        emotion_text=line_emo_text,
                                        seed=regen_seed,
                                        seed_origin="auto_regen",
                                        seed_strategy=normalized_seed_strategy,
                                        delivery_rate=delivery_rate,
                                    )
                                    quality_score = regen_quality_score
                                    status_log.append(f"    ✅ Improved to quality {regen_quality_score:.2f}")
                                    yield "\n".join(status_log), self._update_progress_bar(progress_percent), progress_percent, None
                                    
                                    if regen_similarity >= similarity_threshold and regen_robotic <= robotic_threshold:
                                        status_log.append(f"    ✅ Auto-regen version meets quality threshold!")
                                        yield "\n".join(status_log), self._update_progress_bar(progress_percent), progress_percent, None
                                        break
                                
                            except Exception as e:
                                status_log.append(f"    ❌ Auto-regen attempt {regen_attempt+1} failed: {str(e)}")
                                yield "\n".join(status_log), self._update_progress_bar(progress_percent), progress_percent, None
                
            except Exception as e:
                status_log.append(f"  ❌ Version {attempt+1} failed: {str(e)}")
                yield "\n".join(status_log), self._update_progress_bar(progress_percent), progress_percent, None
        
        # Update progress to 100% completion
        status_log.append(f"✅ Regeneration complete! Generated {len(new_versions)} new versions for Line {line_index + 1}.")
        self.cleanup_memory()
        yield "\n".join(status_log), self._update_progress_bar(100), 100, new_versions
