"""
Configuration management for IndexTTS2 API.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Server settings
    host: str = Field(default="0.0.0.0", validation_alias="INDTEXTS_HOST")
    port: int = Field(default=8000, validation_alias="INDTEXTS_PORT")
    reload: bool = Field(default=False, validation_alias="INDTEXTS_RELOAD")
    log_level: str = Field(default="info", validation_alias="INDTEXTS_LOG_LEVEL")
    debug: bool = Field(default=False, validation_alias="INDTEXTS_DEBUG")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000", "http://127.0.0.1:8080", "http://localhost:8000"],
        validation_alias="INDTEXTS_CORS_ORIGINS"
    )
    
    # TTS settings
    use_gpu: bool = Field(default=True, validation_alias="INDTEXTS_USE_GPU")
    model_path: Optional[str] = Field(default=None, validation_alias="INDTEXTS_MODEL_PATH")
    device: str = Field(default="auto", validation_alias="INDTEXTS_DEVICE")
    fp16_mode: str = Field(default="auto", validation_alias="INDTEXTS_USE_FP16")
    use_deepspeed: bool = Field(default=True, validation_alias="INDTEXTS_USE_DEEPSPEED")
    use_cuda_kernel: bool = Field(default=False, validation_alias="INDTEXTS_USE_CUDA_KERNEL")
    
    # File storage settings
    upload_dir: str = Field(default="shared/audio/uploads", validation_alias="INDTEXTS_UPLOAD_DIR")
    output_dir: str = Field(default="shared/audio/outputs", validation_alias="INDTEXTS_OUTPUT_DIR")
    temp_dir: str = Field(default="shared/audio/temp", validation_alias="INDTEXTS_TEMP_DIR")
    speakers_dir: str = Field(default="shared/audio/speakers", validation_alias="INDTEXTS_SPEAKERS_DIR")
    source_clips_dir: str = Field(default="shared/audio/source_clips", validation_alias="INDTEXTS_SOURCE_CLIPS_DIR")
    temp_conversation_dir: str = Field(default="shared/audio/temp_conversation_segments", validation_alias="INDTEXTS_TEMP_CONVERSATION_DIR")
    project_saves_dir: str = Field(default="shared/data/project_saves", validation_alias="INDTEXTS_PROJECT_SAVES_DIR")
    timeline_projects_dir: str = Field(default="shared/data/timeline_projects", validation_alias="INDTEXTS_TIMELINE_PROJECTS_DIR")
    pretrained_models_dir: str = Field(default="shared/models/pretrained", validation_alias="INDTEXTS_PRETRAINED_MODELS_DIR")
    hf_cache_dir: str = Field(default="shared/models/checkpoints/hf_cache", validation_alias="INDTEXTS_HF_CACHE_DIR")
    max_file_size: int = Field(default=100 * 1024 * 1024, validation_alias="INDTEXTS_MAX_FILE_SIZE")  # 100MB
    
    # Audio processing settings
    similarity_threshold: float = Field(default=0.60, validation_alias="INDTEXTS_SIMILARITY_THRESHOLD")
    robotic_threshold: float = Field(default=0.70, validation_alias="INDTEXTS_ROBOTIC_THRESHOLD")
    auto_regen_attempts: int = Field(default=1, validation_alias="INDTEXTS_AUTO_REGEN_ATTEMPTS")
    similarity_backend: str = Field(default="speechbrain", validation_alias="INDTEXTS_SIMILARITY_BACKEND")
    
    # Conversation settings
    max_conversation_length: int = Field(default=1000, validation_alias="INDTEXTS_MAX_CONVERSATION_LENGTH")
    max_versions_per_line: int = Field(default=5, validation_alias="INDTEXTS_MAX_VERSIONS_PER_LINE")
    generation_worker_slots: int = Field(default=1, validation_alias="INDTEXTS_GENERATION_WORKER_SLOTS")
    generation_max_pending_tasks: int = Field(default=4, validation_alias="INDTEXTS_GENERATION_MAX_PENDING_TASKS")

    def resolve_device(self) -> Optional[str]:
        """
        Resolve the requested inference device.

        Returns:
            Optional[str]: ``None`` for automatic detection, otherwise an explicit
            device string such as ``cpu``, ``cuda``, ``cuda:1``, ``xpu``, or ``mps``.
        """
        requested = (self.device or "auto").strip().lower()

        if requested in {"", "auto"}:
            return None if self.use_gpu else "cpu"

        if requested == "gpu":
            return "cuda"

        return requested

    def resolve_fp16(self, resolved_device: Optional[str]) -> bool:
        """
        Resolve whether FP16 should be requested for the current runtime.

        Args:
            resolved_device: The explicit device string returned by
                :meth:`resolve_device`, or ``None`` when auto-detecting.

        Returns:
            bool: Whether FP16 should be requested from the TTS runtime.
        """
        mode = (self.fp16_mode or "auto").strip().lower()

        if mode in {"1", "true", "yes", "on"}:
            requested = True
        elif mode in {"0", "false", "no", "off"}:
            requested = False
        else:
            requested = self.use_gpu

        # The model already disables unsupported combinations internally, but we
        # clamp obvious CPU and MPS requests here to keep startup logs truthful.
        if resolved_device == "cpu" or resolved_device == "mps":
            return False

        return requested


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()


# Create global settings instance
settings = get_settings()


def ensure_directories():
    """
    Ensure all required directories exist.
    """
    directories = [
        settings.upload_dir,
        settings.output_dir,
        settings.temp_dir,
        settings.speakers_dir,
        settings.source_clips_dir,
        settings.temp_conversation_dir,
        settings.project_saves_dir,
        settings.timeline_projects_dir,
        settings.pretrained_models_dir,
        settings.hf_cache_dir,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Initialize directories on import
ensure_directories()
