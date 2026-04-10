"""
Speaker tools endpoints for IndexTTS2 API.
Handles audio processing operations for speakers.
"""

import os
import asyncio
from pathlib import Path
from typing import List, Generator
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Request
from fastapi.responses import StreamingResponse

from ..models import (
    VideoExtractionRequest, AudioTrimRequest, BatchProcessingRequest,
    ProcessingProgress, BaseResponse, SourceClipPreparationRequest
)
from ..exceptions import (
    IndexTTSException, FileNotFoundError, ValidationError,
    AudioProcessingError, FileUploadError
)
from ..config import settings
from ..services import AudioProcessingService, FileService

router = APIRouter()


def get_audio_processing_service(request: Request) -> AudioProcessingService:
    """Get audio processing service from app state."""
    if not hasattr(request.app.state, 'audio_processing_service'):
        request.app.state.audio_processing_service = AudioProcessingService()
    
    return request.app.state.audio_processing_service


def get_file_service(request: Request) -> FileService:
    """Get file service from app state."""
    if not hasattr(request.app.state, 'file_service'):
        request.app.state.file_service = FileService()
    
    return request.app.state.file_service


@router.post("/extract-audio", response_model=BaseResponse)
async def extract_audio_from_video(request: VideoExtractionRequest, http_request: Request):
    """
    Extract audio from video file.
    
    Args:
        request: Video extraction request
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Extraction result
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        
        # Extract audio using service
        result = audio_service.extract_audio_from_video(
            video_filename=request.video_filename,
            output_name=request.output_name
        )
        
        return BaseResponse(message=result["message"])
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to extract audio: {str(e)}")


@router.post("/upload-video", response_model=BaseResponse)
async def upload_video_file(
    file: UploadFile = File(...),
    output_name: str = None,
    request: Request = None
):
    """
    Upload a video file for audio extraction.
    
    Args:
        file: Video file to upload
        output_name: Optional output name for extracted audio
        request: HTTP request object
        
    Returns:
        BaseResponse: Upload result
    """
    try:
        file_service = get_file_service(request)
        
        # Validate file type
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        if not any(file.filename.lower().endswith(ext) for ext in video_extensions):
            raise ValidationError("Unsupported video format. Supported formats: MP4, AVI, MOV, MKV, WebM")
        
        # Read file content
        content = await file.read()
        
        # Upload video to source_clips category
        result = file_service.upload_file_from_bytes(
            file_bytes=content,
            filename=file.filename,
            category="source_clips",
            custom_name=output_name,
            allowed_types=['video']
        )
        
        return BaseResponse(
            message=f"Video uploaded successfully: {result['filename']}",
            details={
                "filename": result["filename"],
                "size_kb": result["size_kb"],
                "file_type": result["file_type"]
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileUploadError(f"Failed to upload video: {str(e)}")


@router.post("/upload-source-clip", response_model=BaseResponse)
async def upload_source_clip(
    file: UploadFile = File(...),
    output_name: str = None,
    request: Request = None
):
    """
    Upload an audio file into the source_clips workspace.
    """
    try:
        file_service = get_file_service(request)

        audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
        if not any(file.filename.lower().endswith(ext) for ext in audio_extensions):
            raise ValidationError("Unsupported audio format. Supported formats: WAV, MP3, FLAC, M4A, OGG")

        content = await file.read()

        result = file_service.upload_file_from_bytes(
            file_bytes=content,
            filename=file.filename,
            category="source_clips",
            custom_name=output_name,
            allowed_types=["audio"],
        )

        return BaseResponse(
            message=f"Source clip uploaded successfully: {result['filename']}",
            details=result,
        )
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileUploadError(f"Failed to upload source clip: {str(e)}")


@router.post("/trim-audio", response_model=BaseResponse)
async def trim_audio(request: AudioTrimRequest, http_request: Request):
    """
    Trim audio segment from existing audio file.
    
    Args:
        request: Audio trimming request
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Trimming result
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        
        # Trim audio using service
        result = audio_service.trim_audio_segment(
            original_filename=request.original_filename,
            output_name=request.output_name,
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        return BaseResponse(message=result["message"])
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to trim audio: {str(e)}")


@router.post("/batch-process")
async def batch_process_source_clips(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    Process all source clips through vocal separation and normalization.
    
    Args:
        request: Batch processing request
        background_tasks: FastAPI background tasks
        http_request: HTTP request object
        
    Returns:
        StreamingResponse: Progress updates
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        
        async def generate_progress():
            """Generate progress updates."""
            try:
                # Process batch using service
                result = audio_service.batch_process_source_clips(
                    use_noise_reduction=request.options.use_noise_reduction,
                    use_vocal_separation=request.options.use_vocal_separation,
                    normalization_strength=request.options.normalization_strength,
                    noise_reduction_strength=request.options.noise_reduction_strength,
                    noise_reduction_backend=request.options.noise_reduction_backend,
                )
                
                # Stream progress updates
                for update in result["progress_updates"]:
                    yield f"data: {update}\n\n"
                    await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
                
                yield "data: PROCESSING_COMPLETE\n\n"
                
            except Exception as e:
                yield f"data: ERROR: {str(e)}\n\n"
        
        return StreamingResponse(
            generate_progress(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to start batch processing: {str(e)}")


@router.get("/refresh-lists", response_model=BaseResponse)
async def refresh_speaker_lists(http_request: Request):
    """
    Refresh and return formatted lists of source clips and speakers.
    
    Args:
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Refresh result with lists
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        
        # Refresh lists using service
        result = audio_service.refresh_speaker_lists()
        
        return BaseResponse(
            message=result["message"],
            details={
                "source_clips": result["source_clips"],
                "speakers": result["speakers"]
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to refresh speaker lists: {str(e)}")


@router.get("/check-audio-quality/{filename}")
async def check_audio_quality(filename: str, http_request: Request):
    """
    Check basic audio quality metrics for a file.
    
    Args:
        filename: Name of the audio file to check
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Quality information
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        
        # Get audio quality metrics using service
        result = audio_service.get_audio_quality_metrics(filename)
        
        return BaseResponse(
            message=f"Audio quality check completed for {filename}",
            details={
                "audio_metrics": result["audio_metrics"],
                "file_size_kb": result["size_kb"],
                "robotic_score": result.get("robotic_score")
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to check audio quality: {str(e)}")


@router.get("/source-clip-diagnostics/{filename}", response_model=BaseResponse)
async def source_clip_diagnostics(filename: str, http_request: Request):
    """
    Return cloning-focused diagnostics for a source clip.
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        result = audio_service.get_source_clip_diagnostics(filename)

        return BaseResponse(
            message=f"Diagnostics ready for {filename}",
            details=result,
        )
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to analyze source clip: {str(e)}")


@router.post("/prepare-source-clip", response_model=BaseResponse)
async def prepare_source_clip_endpoint(request: SourceClipPreparationRequest, http_request: Request):
    """
    Prepare a source clip and write it back to source_clips or speakers.
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        result = audio_service.prepare_source_clip(
            source_filename=request.source_filename,
            output_name=request.output_name,
            target_category=request.target_category,
            start_time=request.start_time,
            end_time=request.end_time,
            convert_to_mono=request.convert_to_mono,
            normalize_audio=request.normalize_audio,
            target_peak_dbfs=request.target_peak_dbfs,
            use_noise_reduction=request.use_noise_reduction,
            noise_reduction_strength=request.noise_reduction_strength,
            noise_reduction_backend=request.noise_reduction_backend,
            use_vocal_separation=request.use_vocal_separation,
        )

        return BaseResponse(
            message=result["message"],
            details=result,
        )
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to prepare source clip: {str(e)}")


@router.post("/detect-robotic/{filename}")
async def detect_robotic_speech(filename: str, http_request: Request):
    """
    Detect robotic speech characteristics in an audio file.
    
    Args:
        filename: Name of the audio file to analyze
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Robotic speech detection result
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        
        # Detect robotic speech using service
        result = audio_service.detect_robotic_speech(filename)
        
        return BaseResponse(
            message=result["message"],
            details={
                "robotic_score": result["robotic_score"],
                "is_robotic": result["is_robotic"],
                "quality_level": result["quality_level"],
                "threshold": result["threshold"],
                "meets_threshold": result["meets_threshold"]
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to detect robotic speech: {str(e)}")


@router.get("/list-source-clips")
async def list_source_clips(http_request: Request):
    """
    List all source clips available for processing.
    
    Args:
        http_request: HTTP request object
        
    Returns:
        BaseResponse: List of source clips
    """
    try:
        file_service = get_file_service(http_request)
        
        # Get files from source_clips category
        result = file_service.list_files("source_clips")
        
        return BaseResponse(
            message=f"Found {result['total_count']} source clips",
            details={
                "files": result["files"],
                "total_count": result["total_count"]
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to list source clips: {str(e)}")
