"""
TTS Service for IndexTTS2 API.
Handles core TTS generation functionality with emotion control.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..exceptions import TTSError, ValidationError, ModelNotLoadedError
from ..config import settings
from ..core.app_paths import OUTPUT_DIR, SPEAKERS_DIR
from ..core.pacing import apply_delivery_rate_to_file, clamp_delivery_rate


class TTSService:
    """Service for handling TTS generation operations."""
    
    def __init__(self, tts_core=None, cmd_args=None):
        """
        Initialize TTS service.
        
        Args:
            tts_core: TTSCore instance from webui
            cmd_args: Command line arguments
        """
        self.tts_core = tts_core
        self.cmd_args = cmd_args or self._create_default_cmd_args()
        
    def _create_default_cmd_args(self):
        """Create default command line arguments."""
        class DefaultArgs:
            def __init__(self):
                self.verbose = False
                
        return DefaultArgs()
    
    def generate_single_tts(
        self,
        speaker_filename: str,
        text: str,
        emotion_control_method: str = "from_speaker",
        emotion_reference_filename: Optional[str] = None,
        emotion_weight: float = 1.0,
        emotion_vectors: List[float] = None,
        emotion_text: Optional[str] = None,
        use_random_sampling: bool = False,
        max_text_tokens_per_segment: int = 120,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 0.8,
        length_penalty: float = 0.0,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        max_mel_tokens: int = 1500,
        seed: Optional[int] = None,
        delivery_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Generate single TTS output with emotion control.
        
        Args:
            speaker_filename: Speaker audio filename
            text: Text to synthesize
            emotion_control_method: Emotion control method
            emotion_reference_filename: Emotion reference audio filename
            emotion_weight: Emotion weight
            emotion_vectors: Emotion vector components
            emotion_text: Emotion description text
            use_random_sampling: Whether to use random sampling
            max_text_tokens_per_segment: Max tokens per segment
            **kwargs: Additional generation parameters
        
        Returns:
            Dict: Generation result with audio path and metadata
        """
        print(f"DEBUG: TTSService.generate_single_tts - tts_core is None: {self.tts_core is None}")
        if not self.tts_core:
            print("DEBUG: TTS model not loaded in TTSService, raising ModelNotLoadedError")
            raise ModelNotLoadedError("TTS model not loaded")
        
        print(f"DEBUG: TTSService.generate_single_tts - tts_core type: {type(self.tts_core)}")
        print(f"DEBUG: TTSService.generate_single_tts - tts_core has infer method: {hasattr(self.tts_core, 'infer')}")
        print(f"DEBUG: TTSService.generate_single_tts - tts_core has generate_single method: {hasattr(self.tts_core, 'generate_single')}")
        
        # Validate inputs
        validation_result = self.validate_tts_inputs(speaker_filename, text)
        if not validation_result["valid"]:
            raise ValidationError(validation_result["error"])
        
        # Create output path
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"spk_{int(time.time())}.wav"
        
        # Convert emotion control method to integer
        emo_control_method_map = {
            "from_speaker": 0,
            "from_reference": 1,
            "from_vectors": 2,
            "from_text": 3
        }
        emo_control_method_int = emo_control_method_map.get(emotion_control_method, 0)
        
        # Process emotion reference path
        emo_ref_path = None
        if emotion_reference_filename:
            emo_ref_path = str(SPEAKERS_DIR / emotion_reference_filename)
            if not Path(emo_ref_path).exists():
                raise ValidationError(f"Emotion reference file not found: {emotion_reference_filename}")
        
        # Process emotion vectors
        vec_args = [0.0] * 8
        if emotion_vectors:
            for i, vec in enumerate(emotion_vectors[:8]):
                vec_args[i] = vec
        
        # Add emotion text if needed (pass None instead of empty string for from_text method)
        emo_text = emotion_text
        
        tts_instance = getattr(self.tts_core, 'tts', None) or self.tts_core

        try:
            # Generate TTS using the TTSCore
            # Note: IndexTTS2 uses 'infer' method, not 'generate_single'
            print(f"DEBUG: Calling tts_core.infer with parameters...")
            result = tts_instance.infer(
                spk_audio_prompt=str(SPEAKERS_DIR / speaker_filename),
                text=text,
                output_path=str(output_path),
                emo_audio_prompt=emo_ref_path,
                emo_alpha=emotion_weight,
                emo_vector=emotion_vectors if emotion_vectors and any(v != 0 for v in emotion_vectors) else None,
                use_emo_text=emotion_control_method == "from_text",
                emo_text=emotion_text,
                use_random=use_random_sampling,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                length_penalty=length_penalty,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                max_mel_tokens=max_mel_tokens,
                seed=seed,
                verbose=False
            )
            print(f"DEBUG: tts_core.infer returned: {result}")
            
            # The infer method returns the output path directly
            actual_output_path = result if isinstance(result, str) else str(output_path)
            print(f"DEBUG: Actual output path: {actual_output_path}")

            applied_delivery_rate = clamp_delivery_rate(delivery_rate)
            if abs(applied_delivery_rate - 1.0) > 0.005:
                apply_delivery_rate_to_file(actual_output_path, applied_delivery_rate)
            
            return {
                "success": True,
                "audio_path": actual_output_path,
                "audio_filename": Path(actual_output_path).name,
                "speaker_filename": speaker_filename,
                "text": text,
                "emotion_control_method": emotion_control_method,
                "generation_parameters": {
                    "emotion_weight": emotion_weight,
                    "emotion_vectors": emotion_vectors,
                    "emotion_text": emotion_text,
                    "use_random_sampling": use_random_sampling,
                    "max_text_tokens_per_segment": max_text_tokens_per_segment,
                    "do_sample": do_sample,
                    "top_p": top_p,
                    "top_k": top_k,
                    "temperature": temperature,
                    "length_penalty": length_penalty,
                    "num_beams": num_beams,
                    "repetition_penalty": repetition_penalty,
                    "max_mel_tokens": max_mel_tokens,
                    "seed": seed
                    ,
                    "delivery_rate": applied_delivery_rate,
                }
            }
            
        except Exception as e:
            raise TTSError(f"TTS generation failed: {str(e)}")
        finally:
            try:
                if hasattr(tts_instance, "release_unused_memory"):
                    tts_instance.release_unused_memory(clear_prompt_cache=True)
            except Exception as cleanup_error:
                print(f"DEBUG: Failed to release TTS memory after single generation: {cleanup_error}")
    
    def validate_tts_inputs(self, speaker_filename: str, text: str) -> Dict[str, Any]:
        """
        Validate TTS generation inputs.
        
        Args:
            speaker_filename: Speaker audio filename
            text: Text to synthesize
        
        Returns:
            Dict: Validation result
        """
        if not speaker_filename:
            return {"valid": False, "error": "Speaker filename is required"}
        
        if not text or not text.strip():
            return {"valid": False, "error": "Text cannot be empty"}
        
        # Check if speaker file exists
        speaker_path = SPEAKERS_DIR / speaker_filename
        print(f"DEBUG: Looking for speaker file at: {speaker_path.absolute()}")
        print(f"DEBUG: Current working directory: {Path.cwd()}")
        print(f"DEBUG: Speaker file exists: {speaker_path.exists()}")
        if not speaker_path.exists():
            return {"valid": False, "error": f"Speaker file not found: {speaker_filename}"}
        
        # Check text length
        if len(text) > 10000:
            return {"valid": False, "error": "Text too long (max 10000 characters)"}
        
        return {"valid": True, "error": None}
    
    def validate_emotion_vectors(self, emotion_vectors: List[float]) -> Dict[str, Any]:
        """
        Validate emotion vector components.
        
        Args:
            emotion_vectors: List of emotion vector components
        
        Returns:
            Dict: Validation result
        """
        if not emotion_vectors:
            return {"valid": True, "error": None}
        
        if len(emotion_vectors) > 8:
            return {"valid": False, "error": "Maximum 8 emotion vector components allowed"}
        
        if sum(abs(x) for x in emotion_vectors) > 1.5:
            return {"valid": False, "error": "Sum of emotion vector components cannot exceed 1.5"}
        
        return {"valid": True, "error": None}
    
    def process_emotion_control_parameters(
        self,
        emotion_control_method: str,
        emotion_reference_filename: Optional[str],
        emotion_weight: float,
        emotion_vectors: List[float]
    ) -> Tuple[Optional[List[float]], Optional[str], float]:
        """
        Process emotion control parameters based on selected method.
        
        Args:
            emotion_control_method: Selected emotion control method
            emotion_reference_filename: Emotion reference audio filename
            emotion_weight: Emotion weight
            emotion_vectors: Emotion vector components
        
        Returns:
            Tuple: (emo_vector, emo_ref_path, emo_weight)
        """
        emo_vector = None
        emo_ref_path = None
        
        if emotion_control_method == "from_speaker":
            # Emotion from speaker
            emo_ref_path = None
            emo_weight = 1.0
        elif emotion_control_method == "from_reference":
            # Emotion from reference audio
            if emotion_reference_filename:
                emo_ref_path = str(SPEAKERS_DIR / emotion_reference_filename)
            # emo_weight remains as provided
        elif emotion_control_method == "from_vectors":
            # Emotion from custom vectors
            emo_vector = emotion_vectors[:8] if emotion_vectors else [0.0] * 8
            if sum(emo_vector) > 1.5:
                raise ValidationError("The sum of emotion vectors cannot exceed 1.5")
        elif emotion_control_method == "from_text":
            # Emotion from text description
            # emo_weight remains as provided
            pass
        
        return emo_vector, emo_ref_path, emo_weight
    
    def get_supported_emotion_methods(self) -> List[Dict[str, Any]]:
        """
        Get list of supported emotion control methods.
        
        Returns:
            List: Emotion control methods with descriptions
        """
        return [
            {
                "id": "from_speaker",
                "name": "From Speaker",
                "description": "Use emotion from the speaker audio prompt"
            },
            {
                "id": "from_reference",
                "name": "From Reference Audio",
                "description": "Use emotion from a separate reference audio file"
            },
            {
                "id": "from_vectors",
                "name": "From Custom Vectors",
                "description": "Use custom emotion vector parameters"
            },
            {
                "id": "from_text",
                "name": "From Text Description",
                "description": "Use emotion from text description"
            }
        ]
    
    def get_default_generation_parameters(self) -> Dict[str, Any]:
        """
        Get default TTS generation parameters.
        
        Returns:
            Dict: Default parameters
        """
        return {
            "max_text_tokens_per_segment": 120,
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 30,
            "temperature": 0.8,
            "length_penalty": 0.0,
            "num_beams": 3,
            "repetition_penalty": 10.0,
            "max_mel_tokens": 1500,
            "emotion_weight": 1.0,
            "use_random_sampling": False
        }
