"""
Custom exceptions for IndexTTS2 API.
"""

from typing import Any, Dict, Optional


class IndexTTSException(Exception):
    """
    Base exception class for IndexTTS2 API errors.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "GENERAL_ERROR",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(IndexTTSException):
    """
    Raised when request validation fails.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details=details
        )


class FileNotFoundError(IndexTTSException):
    """
    Raised when a requested file is not found.
    """
    
    def __init__(self, file_path: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"File not found: {file_path}",
            error_code="FILE_NOT_FOUND",
            status_code=404,
            details=details
        )


class ModelNotLoadedError(IndexTTSException):
    """
    Raised when TTS model is not loaded.
    """
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="TTS model is not loaded. Please load a model first.",
            error_code="MODEL_NOT_LOADED",
            status_code=503,
            details=details
        )


class AudioProcessingError(IndexTTSException):
    """
    Raised when audio processing fails.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Audio processing error: {message}",
            error_code="AUDIO_PROCESSING_ERROR",
            status_code=500,
            details=details
        )


class TTSError(IndexTTSException):
    """
    Raised when TTS generation fails.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"TTS generation error: {message}",
            error_code="TTS_ERROR",
            status_code=500,
            details=details
        )


class ConversationError(IndexTTSException):
    """
    Raised when conversation generation fails.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Conversation generation error: {message}",
            error_code="CONVERSATION_ERROR",
            status_code=500,
            details=details
        )


class SpeakerError(IndexTTSException):
    """
    Raised when speaker operations fail.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Speaker error: {message}",
            error_code="SPEAKER_ERROR",
            status_code=500,
            details=details
        )


class FileUploadError(IndexTTSException):
    """
    Raised when file upload fails.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"File upload error: {message}",
            error_code="FILE_UPLOAD_ERROR",
            status_code=500,
            details=details
        )


class SimilarityAnalysisError(IndexTTSException):
    """
    Raised when speaker similarity analysis fails.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Similarity analysis error: {message}",
            error_code="SIMILARITY_ANALYSIS_ERROR",
            status_code=500,
            details=details
        )


class TimelineError(IndexTTSException):
    """
    Raised when timeline operations fail.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Timeline error: {message}",
            error_code="TIMELINE_ERROR",
            status_code=500,
            details=details
        )