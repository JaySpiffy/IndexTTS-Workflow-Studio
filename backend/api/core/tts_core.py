"""
TTS Core module for standalone FastAPI implementation.
Handles the core TTS generation functionality and emotion control without Gradio dependencies.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from .app_paths import OUTPUT_DIR


class TTSCore:
    """Core TTS generation functionality for IndexTTS2."""
    
    def __init__(self, tts_model, cmd_args):
        """
        Initialize the TTS core.
        
        Args:
            tts_model: IndexTTS2 model instance
            cmd_args: Command line arguments
        """
        self.tts = tts_model
        self.cmd_args = cmd_args
        self.progress_callback = None
        self.emotion_timeline_enabled = False
        self.current_emotion_timeline = None
    
    def generate_single(
        self,
        emo_control_method: int,
        prompt: str,
        text: str,
        emo_ref_path: Optional[str],
        emo_weight: float,
        vec1: float, vec2: float, vec3: float, vec4: float,
        vec5: float, vec6: float, vec7: float, vec8: float,
        emo_text: str,
        emo_random: bool,
        max_text_tokens_per_segment: int = 120,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 0.8,
        length_penalty: float = 0.0,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        max_mel_tokens: int = 1500
    ) -> Dict[str, Any]:
        """
        Generate single TTS output with emotion control.
        
        Args:
            emo_control_method: Emotion control method (0-3)
            prompt: Speaker prompt audio path
            text: Text to synthesize
            emo_ref_path: Emotion reference audio path
            emo_weight: Emotion weight
            vec1-vec8: Emotion vector components
            emo_text: Emotion description text
            emo_random: Whether to use random sampling
            max_text_tokens_per_segment: Max tokens per segment
            **kwargs: Additional generation parameters
        
        Returns:
            Dict: Generation result with audio path and metadata
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(OUTPUT_DIR / f"spk_{int(time.time())}.wav")
        
        # Prepare generation parameters
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        
        # Process emotion control parameters
        emo_vector, emo_ref_path, emo_weight = self._process_emotion_control(
            emo_control_method, emo_ref_path, emo_weight, 
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8
        )
        
        # Process emotion text
        if emo_text == "" or emo_text is None:
            # For emotion from text method, if no emotion text is provided,
            # we'll use the main text for emotion analysis
            if emo_control_method == 3:
                # We'll handle this in the infer function by using the main text
                pass
            else:
                emo_text = None
        
        print(f"[DEBUG] Emo control mode:{emo_control_method},weight:{emo_weight},vec:{emo_vector}")
        print(f"[DEBUG] Emotion text before processing: {emo_text}")
        print(f"[DEBUG] use_emo_text will be set to: {emo_control_method == 3}")
        
        # Generate audio
        output = self.tts.infer(
            spk_audio_prompt=prompt,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_ref_path,
            emo_alpha=emo_weight,
            emo_vector=emo_vector,
            use_emo_text=(emo_control_method == 3),
            emo_text=emo_text,
            use_random=emo_random,
            verbose=self.cmd_args.verbose,
            max_text_tokens_per_segment=int(max_text_tokens_per_segment),
            **kwargs
        )
        
        # Return a dict for API usage
        return {"audio_path": output, "visible": True}
    
    def generate_with_emotion_timeline(
        self,
        prompt: str,
        text: str,
        emotion_timeline: List[Dict[str, Any]],
        segment_duration: float,
        emotion_control_method: int = 2,  # Default to custom vectors
        emo_ref_path: Optional[str] = None,
        emo_weight: float = 1.0,
        use_random_sampling: bool = False,
        max_text_tokens_per_segment: int = 120,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 0.8,
        length_penalty: float = 0.0,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        max_mel_tokens: int = 1500
    ) -> Dict[str, Any]:
        """
        Generate TTS with emotion timeline support.
        
        Args:
            prompt: Speaker prompt audio path
            text: Text to synthesize
            emotion_timeline: List of emotion timeline entries with timestamps and vectors
            segment_duration: Duration of the segment in seconds
            emotion_control_method: Emotion control method (0-3)
            emo_ref_path: Emotion reference audio path
            emo_weight: Emotion weight
            use_random_sampling: Whether to use random sampling
            **kwargs: Additional generation parameters
        
        Returns:
            Dict: Generation result with audio path and metadata
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(OUTPUT_DIR / f"spk_emotion_timeline_{int(time.time())}.wav")
        
        # Prepare generation parameters
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        
        # Store emotion timeline for processing
        self.emotion_timeline_enabled = True
        self.current_emotion_timeline = emotion_timeline
        
        # For emotion timeline generation, we need to process the text in chunks
        # and apply different emotion vectors at different timestamps
        
        # Split text into chunks based on timeline entries
        text_chunks = self._split_text_by_emotion_timeline(text, emotion_timeline, segment_duration)
        
        # Generate audio for each chunk with appropriate emotion
        audio_segments = []
        current_time = 0.0
        
        for chunk in text_chunks:
            # Find the emotion vector for this chunk's timestamp
            emotion_vectors = self._get_emotion_at_timestamp(chunk["timestamp"], emotion_timeline)
            
            # Generate audio for this chunk
            chunk_result = self.generate_single(
                emo_control_method=emotion_control_method,
                prompt=prompt,
                text=chunk["text"],
                emo_ref_path=emo_ref_path,
                emo_weight=emo_weight,
                vec1=emotion_vectors[0] if len(emotion_vectors) > 0 else 0.0,
                vec2=emotion_vectors[1] if len(emotion_vectors) > 1 else 0.0,
                vec3=emotion_vectors[2] if len(emotion_vectors) > 2 else 0.0,
                vec4=emotion_vectors[3] if len(emotion_vectors) > 3 else 0.0,
                vec5=emotion_vectors[4] if len(emotion_vectors) > 4 else 0.0,
                vec6=emotion_vectors[5] if len(emotion_vectors) > 5 else 0.0,
                vec7=emotion_vectors[6] if len(emotion_vectors) > 6 else 0.0,
                vec8=emotion_vectors[7] if len(emotion_vectors) > 7 else 0.0,
                emo_text="",
                emo_random=use_random_sampling,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                **kwargs
            )
            
            if chunk_result and chunk_result.get("audio_path"):
                audio_segments.append(chunk_result["audio_path"])
            
            current_time += chunk.get("duration", segment_duration / len(text_chunks))
        
        # Concatenate audio segments
        if audio_segments:
            from .audio_processing import concatenate_audio_files
            concatenate_audio_files(audio_segments, output_path)
        
        # Reset emotion timeline state
        self.emotion_timeline_enabled = False
        self.current_emotion_timeline = None
        
        return {"audio_path": output_path, "visible": True}
    
    def _split_text_by_emotion_timeline(
        self,
        text: str,
        emotion_timeline: List[Dict[str, Any]],
        segment_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks based on emotion timeline entries.
        
        Args:
            text: Text to split
            emotion_timeline: Emotion timeline with timestamps
            segment_duration: Total duration of the segment
        
        Returns:
            List of text chunks with timestamps
        """
        if not emotion_timeline:
            # If no emotion timeline, return single chunk
            return [{"text": text, "timestamp": 0.0, "duration": segment_duration}]
        
        # Sort timeline by timestamp
        sorted_timeline = sorted(emotion_timeline, key=lambda x: x["timestamp"])
        
        # Calculate chunk durations based on timeline
        chunks = []
        text_words = text.split()
        total_words = len(text_words)
        
        # Create chunks based on timeline entries
        for i in range(len(sorted_timeline)):
            start_time = sorted_timeline[i]["timestamp"]
            end_time = sorted_timeline[i + 1]["timestamp"] if i + 1 < len(sorted_timeline) else segment_duration
            
            # Calculate proportion of text for this chunk
            chunk_duration = end_time - start_time
            chunk_proportion = chunk_duration / segment_duration
            chunk_word_count = max(1, int(total_words * chunk_proportion))
            
            # Get words for this chunk
            start_word = sum(int(total_words * (sorted_timeline[j]["timestamp"] / segment_duration)) for j in range(i))
            end_word = min(start_word + chunk_word_count, total_words)
            
            chunk_text = " ".join(text_words[start_word:end_word])
            
            chunks.append({
                "text": chunk_text,
                "timestamp": start_time,
                "duration": chunk_duration,
                "emotion_vectors": sorted_timeline[i]["emotion_vectors"]
            })
        
        return chunks
    
    def _get_emotion_at_timestamp(
        self,
        timestamp: float,
        emotion_timeline: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Get emotion vector at a specific timestamp.
        
        Args:
            timestamp: Timestamp in seconds
            emotion_timeline: Emotion timeline with timestamps
        
        Returns:
            Emotion vector at the specified timestamp
        """
        if not emotion_timeline:
            return [0.0] * 8
        
        # Sort timeline by timestamp
        sorted_timeline = sorted(emotion_timeline, key=lambda x: x["timestamp"])
        
        # Find the timeline entry that matches or is before the timestamp
        for entry in sorted_timeline:
            if entry["timestamp"] <= timestamp:
                return entry["emotion_vectors"]
        
        # If no entry found, return the first entry
        return sorted_timeline[0]["emotion_vectors"] if sorted_timeline else [0.0] * 8
    
    def set_emotion_timeline_enabled(self, enabled: bool):
        """
        Enable or disable emotion timeline processing.
        
        Args:
            enabled: Whether to enable emotion timeline processing
        """
        self.emotion_timeline_enabled = enabled
    
    def set_current_emotion_timeline(self, emotion_timeline: List[Dict[str, Any]]):
        """
        Set the current emotion timeline for processing.
        
        Args:
            emotion_timeline: Emotion timeline data
        """
        self.current_emotion_timeline = emotion_timeline
    
    def _process_emotion_control(
        self,
        emo_control_method: int,
        emo_ref_path: Optional[str],
        emo_weight: float,
        vec1: float, vec2: float, vec3: float, vec4: float,
        vec5: float, vec6: float, vec7: float, vec8: float
    ) -> Tuple[Optional[List[float]], Optional[str], float]:
        """
        Process emotion control parameters based on selected method.
        
        Args:
            emo_control_method: Selected emotion control method
            emo_ref_path: Emotion reference audio path
            emo_weight: Emotion weight
            vec1-vec8: Emotion vector components
        
        Returns:
            Tuple: (emo_vector, emo_ref_path, emo_weight)
        """
        emo_vector = None
        
        if emo_control_method == 0:  # emotion from speaker
            emo_ref_path = None  # remove external reference audio
            emo_weight = 1.0
        elif emo_control_method == 1:  # emotion from reference audio
            # emo_weight remains as provided
            pass
        elif emo_control_method == 2:  # emotion from custom vectors
            emo_vector = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            if sum(emo_vector) > 1.5:
                print("Warning: The sum of emotion vectors cannot exceed 1.5, please adjust and try again.")
                return None, None, emo_weight
        else:
            # don't use emotion vector inputs for other modes
            emo_vector = None
        
        return emo_vector, emo_ref_path, emo_weight
    
    def set_progress_callback(self, progress_callback):
        """
        Set progress callback for TTS generation.
        
        Args:
            progress_callback: Callback function for progress updates
        """
        self.progress_callback = progress_callback
    
    def validate_inputs(self, prompt: str, text: str) -> Tuple[bool, str]:
        """
        Validate TTS generation inputs.
        
        Args:
            prompt: Speaker prompt audio path
            text: Text to synthesize
        
        Returns:
            Tuple: (is_valid, error_message)
        """
        if not prompt:
            return False, "Please select a speaker prompt"
        
        if not text or not text.strip():
            return False, "Please enter text to synthesize"
        
        if not os.path.exists(prompt):
            return False, f"Speaker prompt file not found: {prompt}"
        
        return True, ""
