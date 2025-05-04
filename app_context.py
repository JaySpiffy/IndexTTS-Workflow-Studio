# app_context.py
from typing import Optional, Any
import torch

# --- Import TTS Type Hint (if available) ---
try:
    from indextts.infer import IndexTTS
except ImportError:
    IndexTTS = Any # Fallback if IndexTTS cannot be imported

# --- Import Speaker Model Type Hint (if available) ---
try:
    # Assuming EncoderClassifier is the type, adjust if needed
    from speechbrain.pretrained import EncoderClassifier
except ImportError:
    EncoderClassifier = Any # Fallback

class AppContext:
    """Holds shared resources and configurations for the application."""
    def __init__(
        self,
        # Core Models
        tts_instance: Optional[IndexTTS] = None,
        speaker_similarity_model_instance: Optional[EncoderClassifier] = None,
        # Hardware
        device: Optional[torch.device] = None,
        # Library Availability Flags
        pydub_available: bool = False,
        pydub_silence_available: bool = False,
        pydub_compress_available: bool = False,
        scipy_available: bool = False,
        noisereduce_available: bool = False,
        speechbrain_available: bool = False,
        # Add other shared resources or config as needed
    ):
        self.tts = tts_instance
        self.speaker_similarity_model = speaker_similarity_model_instance
        self.device = device
        self.pydub_available = pydub_available
        self.pydub_silence_available = pydub_silence_available
        self.pydub_compress_available = pydub_compress_available
        self.scipy_available = scipy_available
        self.noisereduce_available = noisereduce_available
        self.speechbrain_available = speechbrain_available

        # You could also add constants here if they are widely used and
        # initialized/determined dynamically, though constants.py is fine too.
        # Example: self.temp_dir = TEMP_CONVO_MULTI_DIR
