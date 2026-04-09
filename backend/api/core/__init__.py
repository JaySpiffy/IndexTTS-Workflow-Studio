"""
Core modules for standalone FastAPI implementation.
Contains independent copies of webui functionality without Gradio dependencies.
"""

from .conversation_manager import ConversationManager
from .tts_core import TTSCore
from .file_utils import prepare_temp_dir, parse_conversation_script, list_speaker_files

__all__ = [
    'ConversationManager',
    'TTSCore',
    'prepare_temp_dir',
    'parse_conversation_script',
    'list_speaker_files'
]