"""
Canonical runtime paths for the Docker-first app layout.
"""

from pathlib import Path

from ..config import settings


UPLOAD_DIR = Path(settings.upload_dir)
OUTPUT_DIR = Path(settings.output_dir)
TEMP_DIR = Path(settings.temp_dir)
SPEAKERS_DIR = Path(settings.speakers_dir)
SOURCE_CLIPS_DIR = Path(settings.source_clips_dir)
TEMP_CONVERSATION_SEGMENTS_DIR = Path(settings.temp_conversation_dir)
PROJECT_SAVES_DIR = Path(settings.project_saves_dir)
TIMELINE_PROJECTS_DIR = Path(settings.timeline_projects_dir)
PRETRAINED_MODELS_DIR = Path(settings.pretrained_models_dir)
HF_CACHE_DIR = Path(settings.hf_cache_dir)
