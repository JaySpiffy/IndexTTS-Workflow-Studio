"""
Audio Processing Service for IndexTTS2 API.
Handles audio analysis, similarity analysis, and processing operations.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..core.app_paths import (
    OUTPUT_DIR,
    SOURCE_CLIPS_DIR,
    SPEAKERS_DIR,
    TEMP_CONVERSATION_SEGMENTS_DIR,
)
from ..core.source_clip_prep import analyze_source_clip, prepare_source_clip
from ..exceptions import AudioProcessingError, SimilarityAnalysisError, ModelNotLoadedError, ValidationError
from ..config import settings


class AudioProcessingService:
    """Service for handling audio processing operations."""
    
    def __init__(self):
        """Initialize audio processing service."""
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            SOURCE_CLIPS_DIR,
            SPEAKERS_DIR,
            TEMP_CONVERSATION_SEGMENTS_DIR,
            OUTPUT_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def find_audio_file(self, filename: str) -> Path:
        """
        Find audio file in various directories.
        
        Args:
            filename: Name of the audio file
        
        Returns:
            Path: Path to the audio file
        
        Raises:
            FileNotFoundError: If file not found
        """
        if not filename.endswith('.wav'):
            filename += '.wav'
        
        # Search in multiple directories
        search_dirs = [
            SPEAKERS_DIR,
            SOURCE_CLIPS_DIR,
            TEMP_CONVERSATION_SEGMENTS_DIR,
            OUTPUT_DIR,
        ]
        
        for directory in search_dirs:
            file_path = directory / filename
            if file_path.exists():
                return file_path
        
        raise FileNotFoundError(f"Audio file not found: {filename}")
    
    def analyze_speaker_similarity(
        self,
        reference_filename: str,
        generated_filename: str
    ) -> Dict[str, Any]:
        """
        Analyze speaker similarity between reference and generated audio.
        
        Args:
            reference_filename: Reference audio filename
            generated_filename: Generated audio filename
        
        Returns:
            Dict: Analysis result
        """
        try:
            # Import here to avoid circular imports
            from api.core.audio_processing import (
                speaker_similarity_model,
                analyze_speaker_similarity_with_quality
            )
            
            if speaker_similarity_model is None:
                raise ModelNotLoadedError("Speaker similarity model not loaded")
            
            # Find audio files
            reference_path = self.find_audio_file(reference_filename)
            generated_path = self.find_audio_file(generated_filename)
            
            # Perform analysis
            result = analyze_speaker_similarity_with_quality(
                speaker_similarity_model,
                str(reference_path),
                str(generated_path)
            )
            
            return {
                "success": True,
                "reference_filename": reference_filename,
                "generated_filename": generated_filename,
                "similarity_score": result['similarity'],
                "robotic_score": result['robotic_score'],
                "quality_score": result['quality_score'],
                "meets_similarity_threshold": result['similarity'] >= settings.similarity_threshold,
                "meets_robotic_threshold": result['robotic_score'] <= settings.robotic_threshold,
                "overall_quality_acceptable": result['quality_score'] >= (settings.similarity_threshold * 0.8),
                "thresholds": {
                    "similarity": settings.similarity_threshold,
                    "robotic": settings.robotic_threshold
                }
            }
            
        except Exception as e:
            if isinstance(e, (AudioProcessingError, ModelNotLoadedError)):
                raise
            raise SimilarityAnalysisError(f"Failed to analyze speaker similarity: {str(e)}")
    
    def batch_similarity_analysis(
        self,
        reference_filename: str,
        generated_filenames: List[str]
    ) -> Dict[str, Any]:
        """
        Perform batch similarity analysis on multiple generated audio files.
        
        Args:
            reference_filename: Reference audio filename
            generated_filenames: List of generated audio filenames
        
        Returns:
            Dict: Batch analysis results
        """
        if not generated_filenames:
            raise ValidationError("At least one generated filename must be provided")
        
        if len(generated_filenames) > 50:
            raise ValidationError("Too many files for batch analysis (max 50)")
        
        try:
            # Import here to avoid circular imports
            from api.core.audio_processing import speaker_similarity_model
            
            if speaker_similarity_model is None:
                raise ModelNotLoadedError("Speaker similarity model not loaded")
            
            # Find reference file
            reference_path = self.find_audio_file(reference_filename)
            
            results = []
            
            for filename in generated_filenames:
                try:
                    analysis = self.analyze_speaker_similarity(reference_filename, filename)
                    results.append(analysis)
                    
                except Exception as e:
                    # Add error result for this file
                    results.append({
                        "success": False,
                        "reference_filename": reference_filename,
                        "generated_filename": filename,
                        "similarity_score": -1.0,
                        "robotic_score": 1.0,
                        "quality_score": 0.0,
                        "error": str(e)
                    })
            
            # Calculate batch statistics
            valid_results = [r for r in results if r.get("success", False)]
            
            if valid_results:
                avg_similarity = sum(r["similarity_score"] for r in valid_results) / len(valid_results)
                avg_robotic = sum(r["robotic_score"] for r in valid_results) / len(valid_results)
                avg_quality = sum(r["quality_score"] for r in valid_results) / len(valid_results)
                
                # Count files meeting thresholds
                similarity_ok = sum(1 for r in valid_results if r["meets_similarity_threshold"])
                robotic_ok = sum(1 for r in valid_results if r["meets_robotic_threshold"])
                quality_ok = sum(1 for r in valid_results if r["overall_quality_acceptable"])
            else:
                avg_similarity = avg_robotic = avg_quality = 0.0
                similarity_ok = robotic_ok = quality_ok = 0
            
            return {
                "success": True,
                "reference_filename": reference_filename,
                "total_files": len(results),
                "valid_analyses": len(valid_results),
                "results": results,
                "batch_statistics": {
                    "average_similarity": avg_similarity,
                    "average_robotic_score": avg_robotic,
                    "average_quality_score": avg_quality,
                    "files_meeting_similarity_threshold": similarity_ok,
                    "files_meeting_robotic_threshold": robotic_ok,
                    "files_with_acceptable_quality": quality_ok
                }
            }
            
        except Exception as e:
            if isinstance(e, (AudioProcessingError, ModelNotLoadedError)):
                raise
            raise SimilarityAnalysisError(f"Failed to perform batch similarity analysis: {str(e)}")
    
    def detect_robotic_speech(self, filename: str) -> Dict[str, Any]:
        """
        Detect robotic speech characteristics in an audio file.
        
        Args:
            filename: Name of the audio file to analyze
        
        Returns:
            Dict: Robotic speech detection result
        """
        try:
            # Import here to avoid circular imports
            from api.core.audio_processing import detect_robotic_speech
            
            audio_path = self.find_audio_file(filename)
            
            # Perform detection
            robotic_score = detect_robotic_speech(str(audio_path))
            
            # Determine quality level
            if robotic_score < 0.3:
                quality_level = "natural"
            elif robotic_score < 0.7:
                quality_level = "moderately robotic"
            else:
                quality_level = "highly robotic"
            
            return {
                "success": True,
                "filename": filename,
                "robotic_score": robotic_score,
                "is_robotic": robotic_score > 0.5,
                "quality_level": quality_level,
                "threshold": settings.robotic_threshold,
                "meets_threshold": robotic_score <= settings.robotic_threshold
            }
            
        except Exception as e:
            if isinstance(e, (AudioProcessingError, FileNotFoundError)):
                raise
            raise SimilarityAnalysisError(f"Failed to detect robotic speech: {str(e)}")
    
    def get_audio_quality_metrics(self, filename: str) -> Dict[str, Any]:
        """
        Get detailed audio quality metrics for a file.
        
        Args:
            filename: Name of the audio file to analyze
        
        Returns:
            Dict: Audio quality metrics
        """
        try:
            # Import here to avoid circular imports
            from api.core.audio_processing import check_audio_quality
            
            audio_path = self.find_audio_file(filename)
            
            # Get basic quality info
            quality_info = check_audio_quality(str(audio_path))
            
            # Get robotic speech detection
            try:
                robotic_result = self.detect_robotic_speech(filename)
                robotic_score = robotic_result["robotic_score"]
            except:
                robotic_score = None
            
            # Parse quality info if it's a string
            if isinstance(quality_info, str):
                # Parse the string into key-value pairs
                metrics = {}
                for line in quality_info.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metrics[key.strip()] = value.strip()
            else:
                metrics = quality_info
            
            # Add file information
            file_stat = audio_path.stat()
            
            return {
                "success": True,
                "filename": filename,
                "file_size_bytes": file_stat.st_size,
                "file_size_kb": round(file_stat.st_size / 1024, 1),
                "audio_metrics": metrics,
                "robotic_score": robotic_score,
                "analysis_timestamp": file_stat.st_mtime
            }
            
        except Exception as e:
            if isinstance(e, (AudioProcessingError, FileNotFoundError)):
                raise
            raise AudioProcessingError(f"Failed to get audio quality metrics: {str(e)}")
    
    def compare_audio_versions(
        self,
        reference_filename: str,
        version_filenames: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple audio versions against a reference.
        
        Args:
            reference_filename: Reference audio filename
            version_filenames: List of version filenames to compare
        
        Returns:
            Dict: Comparison results
        """
        if not version_filenames:
            raise ValidationError("At least one version filename must be provided")
        
        try:
            comparison_results = []
            
            for filename in version_filenames:
                try:
                    analysis = self.analyze_speaker_similarity(reference_filename, filename)
                    comparison_results.append({
                        "filename": filename,
                        "similarity_score": analysis["similarity_score"],
                        "robotic_score": analysis["robotic_score"],
                        "quality_score": analysis["quality_score"],
                        "rank": None,  # Will be calculated below
                        "success": True
                    })
                    
                except Exception as e:
                    comparison_results.append({
                        "filename": filename,
                        "similarity_score": -1.0,
                        "robotic_score": 1.0,
                        "quality_score": 0.0,
                        "rank": None,
                        "success": False,
                        "error": str(e)
                    })
            
            # Rank results by quality score
            valid_results = [r for r in comparison_results if r.get("success", False)]
            valid_results.sort(key=lambda x: x["quality_score"], reverse=True)
            
            for i, result in enumerate(valid_results):
                result["rank"] = i + 1
            
            # Find best version
            best_version = valid_results[0] if valid_results else None
            
            return {
                "success": True,
                "reference_filename": reference_filename,
                "total_versions": len(version_filenames),
                "valid_comparisons": len(valid_results),
                "best_version": best_version,
                "comparison_results": comparison_results,
                "thresholds": {
                    "similarity": settings.similarity_threshold,
                    "robotic": settings.robotic_threshold
                }
            }
            
        except Exception as e:
            if isinstance(e, (AudioProcessingError, ValidationError)):
                raise
            raise SimilarityAnalysisError(f"Failed to compare audio versions: {str(e)}")
    
    def get_speaker_model_status(self) -> Dict[str, Any]:
        """
        Get the status of the speaker similarity model.
        
        Returns:
            Dict: Model status
        """
        try:
            # Import here to avoid circular imports
            from api.core.audio_processing import speaker_similarity_model, SPEECHBRAIN_AVAILABLE
            
            return {
                "success": True,
                "model_loaded": speaker_similarity_model is not None,
                "speechbrain_available": SPEECHBRAIN_AVAILABLE,
                "similarity_threshold": settings.similarity_threshold,
                "robotic_threshold": settings.robotic_threshold,
                "auto_regen_attempts": settings.auto_regen_attempts
            }
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to get model status: {str(e)}")
    
    def extract_audio_from_video(self, video_filename: str, output_name: str) -> Dict[str, Any]:
        """
        Extract audio from video file.
        
        Args:
            video_filename: Name of the video file
            output_name: Output audio filename
        
        Returns:
            Dict: Extraction result
        """
        try:
            # Import here to avoid circular imports
            from api.core.audio_processing import extract_audio_from_video
            
            # Check if video file exists
            video_path = SOURCE_CLIPS_DIR / video_filename
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_filename}")
            
            # Extract audio
            result_message = extract_audio_from_video(
                str(video_path), 
                output_name
            )
            
            if "Error:" in result_message:
                raise AudioProcessingError(result_message)
            
            return {
                "success": True,
                "message": result_message,
                "video_filename": video_filename,
                "output_name": output_name
            }
            
        except Exception as e:
            if isinstance(e, (AudioProcessingError, FileNotFoundError)):
                raise
            raise AudioProcessingError(f"Failed to extract audio: {str(e)}")
    
    def trim_audio_segment(
        self,
        original_filename: str,
        output_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Trim audio segment from existing audio file.
        
        Args:
            original_filename: Original audio filename
            output_name: Output audio filename
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
        
        Returns:
            Dict: Trimming result
        """
        try:
            # Check if original audio file exists
            audio_path = SOURCE_CLIPS_DIR / original_filename
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {original_filename}")
            output_filename = output_name.strip()
            if not output_filename:
                raise ValidationError("Output name cannot be empty")
            if not output_filename.lower().endswith(".wav"):
                output_filename += ".wav"

            output_path = SOURCE_CLIPS_DIR / output_filename
            result = prepare_source_clip(
                audio_path,
                output_path,
                start_time=start_time,
                end_time=end_time,
                convert_to_mono=False,
                normalize_audio=False,
                use_noise_reduction=False,
                use_vocal_separation=False,
            )
            
            return {
                "success": True,
                "message": f"Trimmed audio to {output_filename}",
                "original_filename": original_filename,
                "output_name": output_filename,
                "start_time": start_time,
                "end_time": end_time,
                "processing_notes": result["processing_notes"],
                "before": result["before"],
                "after": result["after"],
            }
            
        except Exception as e:
            if isinstance(e, (AudioProcessingError, FileNotFoundError, ValidationError)):
                raise
            raise AudioProcessingError(f"Failed to trim audio: {str(e)}")
    
    def batch_process_source_clips(
        self,
        use_noise_reduction: bool = False,
        use_vocal_separation: bool = False,
        normalization_strength: float = 0.5,
        noise_reduction_strength: float = 0.5
    ) -> Dict[str, Any]:
        """
        Process all source clips through vocal separation and normalization.
        
        Args:
            use_noise_reduction: Whether to apply noise reduction
            use_vocal_separation: Whether to use vocal separation
            normalization_strength: Strength of normalization
            noise_reduction_strength: Strength of noise reduction
        
        Returns:
            Dict: Processing result
        """
        try:
            # Import here to avoid circular imports
            from api.core.audio_processing import process_all_source_clips
            
            # Create generator function
            progress_generator = process_all_source_clips(
                use_noise_reduction=use_noise_reduction,
                use_vocal_separation=use_vocal_separation,
                normalization_strength=normalization_strength,
                noise_reduction_strength=noise_reduction_strength
            )
            
            # Collect all progress updates
            progress_updates = []
            for update in progress_generator:
                progress_updates.append(update)
            
            return {
                "success": True,
                "message": "Batch processing completed",
                "progress_updates": progress_updates,
                "processing_parameters": {
                    "use_noise_reduction": use_noise_reduction,
                    "use_vocal_separation": use_vocal_separation,
                    "normalization_strength": normalization_strength,
                    "noise_reduction_strength": noise_reduction_strength
                }
            }
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to batch process source clips: {str(e)}")
    
    def refresh_speaker_lists(self) -> Dict[str, Any]:
        """
        Refresh and return formatted lists of source clips and speakers.
        
        Returns:
            Dict: Refresh result with lists
        """
        try:
            # Import here to avoid circular imports
            from api.core.audio_processing import refresh_speaker_lists
            
            source_list, speaker_list = refresh_speaker_lists()
            
            return {
                "success": True,
                "message": "Speaker lists refreshed successfully",
                "source_clips": source_list,
                "speakers": speaker_list
            }
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to refresh speaker lists: {str(e)}")

    def get_source_clip_diagnostics(self, filename: str) -> Dict[str, Any]:
        """
        Return cloning-focused diagnostics for a source clip.
        """
        try:
            clip_path = SOURCE_CLIPS_DIR / filename
            if not clip_path.exists():
                raise FileNotFoundError(f"Source clip not found: {filename}")

            diagnostics = analyze_source_clip(clip_path)

            return {
                "success": True,
                "filename": filename,
                "diagnostics": diagnostics,
            }
        except Exception as e:
            if isinstance(e, (AudioProcessingError, FileNotFoundError)):
                raise
            raise AudioProcessingError(f"Failed to analyze source clip: {str(e)}")

    def prepare_source_clip(
        self,
        source_filename: str,
        output_name: Optional[str],
        *,
        target_category: str = "speakers",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        convert_to_mono: bool = True,
        normalize_audio: bool = True,
        target_peak_dbfs: float = -1.0,
        use_noise_reduction: bool = False,
        noise_reduction_strength: float = 0.35,
        use_vocal_separation: bool = False,
    ) -> Dict[str, Any]:
        """
        Prepare a source clip and write the result back to source_clips or speakers.
        """
        try:
            source_path = SOURCE_CLIPS_DIR / source_filename
            if not source_path.exists():
                raise FileNotFoundError(f"Source clip not found: {source_filename}")

            normalized_category = (target_category or "speakers").strip().lower()
            if normalized_category not in {"source_clips", "speakers"}:
                raise ValidationError("Target category must be either 'source_clips' or 'speakers'")

            output_filename = (output_name or Path(source_filename).stem).strip()
            if not output_filename:
                raise ValidationError("Output name cannot be empty")
            if not output_filename.lower().endswith(".wav"):
                output_filename += ".wav"

            output_dir = SPEAKERS_DIR if normalized_category == "speakers" else SOURCE_CLIPS_DIR
            output_path = output_dir / output_filename

            result = prepare_source_clip(
                source_path,
                output_path,
                start_time=start_time,
                end_time=end_time,
                convert_to_mono=convert_to_mono,
                normalize_audio=normalize_audio,
                target_peak_dbfs=target_peak_dbfs,
                use_noise_reduction=use_noise_reduction,
                noise_reduction_strength=noise_reduction_strength,
                use_vocal_separation=use_vocal_separation,
            )

            return {
                "success": True,
                "message": f"Prepared {source_filename} -> {output_filename}",
                "target_category": normalized_category,
                **result,
            }
        except Exception as e:
            if isinstance(e, (AudioProcessingError, FileNotFoundError, ValidationError)):
                raise
            raise AudioProcessingError(f"Failed to prepare source clip: {str(e)}")
