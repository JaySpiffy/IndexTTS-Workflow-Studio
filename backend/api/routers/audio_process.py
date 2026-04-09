"""
Audio processing endpoints for IndexTTS2 API.
Handles similarity analysis, robotic detection, audio quality analysis, and audio trimming.
"""

from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Request, Form
from pydantic import BaseModel

from ..core.app_paths import (
    OUTPUT_DIR,
    SPEAKERS_DIR,
    TEMP_CONVERSATION_SEGMENTS_DIR,
)
from ..models import (
    SimilarityAnalysisRequest, SimilarityAnalysisResponse,
    BatchSimilarityRequest, BatchSimilarityResponse, BaseResponse
)
from ..exceptions import (
    IndexTTSException, FileNotFoundError, ValidationError,
    SimilarityAnalysisError, ModelNotLoadedError
)
from ..config import settings
from ..services import AudioProcessingService

router = APIRouter()


def get_audio_processing_service(request: Request) -> AudioProcessingService:
    """Get audio processing service from app state."""
    if not hasattr(request.app.state, 'audio_processing_service'):
        request.app.state.audio_processing_service = AudioProcessingService()
    
    return request.app.state.audio_processing_service


def get_speaker_model(request: Request):
    """Get speaker similarity model from app state (legacy compatibility)."""
    # Import here to avoid circular imports
    from webui.audio_processing import speaker_similarity_model
    
    if speaker_similarity_model is None:
        raise ModelNotLoadedError("Speaker similarity model not loaded")
    
    return speaker_similarity_model


def find_audio_file(filename: str) -> Path:
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
        Path(settings.source_clips_dir),
        TEMP_CONVERSATION_SEGMENTS_DIR,
        OUTPUT_DIR,
    ]
    
    for directory in search_dirs:
        file_path = directory / filename
        if file_path.exists():
            return file_path
    
    raise FileNotFoundError(f"Audio file not found: {filename}")


@router.post("/similarity-analysis", response_model=SimilarityAnalysisResponse)
async def analyze_speaker_similarity(
    request: SimilarityAnalysisRequest,
    http_request: Request
):
    """
    Analyze speaker similarity between reference and generated audio.
    
    Args:
        request: Similarity analysis request
        http_request: HTTP request object
        
    Returns:
        SimilarityAnalysisResponse: Analysis result
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        
        # Analyze similarity using service
        result = audio_service.analyze_speaker_similarity(
            reference_filename=request.reference_filename,
            generated_filename=request.generated_filename
        )
        
        return SimilarityAnalysisResponse(
            similarity_score=result["similarity_score"],
            robotic_score=result["robotic_score"],
            quality_score=result["quality_score"],
            analysis_details={
                "reference_filename": result["reference_filename"],
                "generated_filename": result["generated_filename"],
                "similarity_threshold": settings.similarity_threshold,
                "robotic_threshold": settings.robotic_threshold,
                "meets_similarity_threshold": result["meets_similarity_threshold"],
                "meets_robotic_threshold": result["meets_robotic_threshold"],
                "overall_quality_acceptable": result["overall_quality_acceptable"]
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SimilarityAnalysisError(f"Failed to analyze speaker similarity: {str(e)}")


@router.post("/batch-similarity", response_model=BatchSimilarityResponse)
async def batch_similarity_analysis(
    request: BatchSimilarityRequest,
    http_request: Request
):
    """
    Perform batch similarity analysis on multiple generated audio files.
    
    Args:
        request: Batch similarity request
        http_request: HTTP request object
        
    Returns:
        BatchSimilarityResponse: Batch analysis results
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        
        # Perform batch analysis using service
        result = audio_service.batch_similarity_analysis(
            reference_filename=request.reference_filename,
            generated_filenames=request.generated_filenames
        )
        
        # Convert to response format
        results = []
        for item in result["results"]:
            if item.get("success", False):
                results.append(SimilarityAnalysisResponse(
                    similarity_score=item["similarity_score"],
                    robotic_score=item["robotic_score"],
                    quality_score=item["quality_score"],
                    analysis_details={
                        "reference_filename": item["reference_filename"],
                        "generated_filename": item["generated_filename"],
                        "meets_similarity_threshold": item["meets_similarity_threshold"],
                        "meets_robotic_threshold": item["meets_robotic_threshold"]
                    }
                ))
            else:
                results.append(SimilarityAnalysisResponse(
                    similarity_score=item.get("similarity_score", -1.0),
                    robotic_score=item.get("robotic_score", 1.0),
                    quality_score=item.get("quality_score", 0.0),
                    analysis_details={
                        "reference_filename": item.get("reference_filename"),
                        "generated_filename": item.get("generated_filename"),
                        "error": item.get("error")
                    }
                ))
        
        return BatchSimilarityResponse(
            results=results,
            message=result["message"],
            details=result["batch_statistics"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SimilarityAnalysisError(f"Failed to perform batch similarity analysis: {str(e)}")


@router.post("/robotic-detection/{filename}")
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
                "filename": result["filename"],
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
        raise SimilarityAnalysisError(f"Failed to detect robotic speech: {str(e)}")


@router.get("/audio-quality/{filename}")
async def get_audio_quality_metrics(filename: str, http_request: Request):
    """
    Get detailed audio quality metrics for a file.
    
    Args:
        filename: Name of the audio file to analyze
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Audio quality metrics
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        
        # Get audio quality metrics using service
        result = audio_service.get_audio_quality_metrics(filename)
        
        return BaseResponse(
            message=f"Audio quality analysis completed for {filename}",
            details={
                "filename": result["filename"],
                "file_size_bytes": result["file_size_bytes"],
                "file_size_kb": result["file_size_kb"],
                "audio_metrics": result["audio_metrics"],
                "robotic_score": result.get("robotic_score"),
                "analysis_timestamp": result.get("analysis_timestamp")
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SimilarityAnalysisError(f"Failed to get audio quality metrics: {str(e)}")


@router.get("/model-status")
async def get_speaker_model_status(http_request: Request):
    """
    Get the status of the speaker similarity model.
    
    Args:
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Model status
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        
        # Get model status using service
        result = audio_service.get_speaker_model_status()
        
        return BaseResponse(
            message=result["message"],
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SimilarityAnalysisError(f"Failed to get model status: {str(e)}")


@router.post("/compare-versions")
async def compare_audio_versions(
    reference_filename: str,
    version_filenames: List[str],
    http_request: Request
):
    """
    Compare multiple audio versions against a reference.
    
    Args:
        reference_filename: Reference audio file
        version_filenames: List of version files to compare
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Comparison results
    """
    try:
        audio_service = get_audio_processing_service(http_request)
        
        # Compare audio versions using service
        result = audio_service.compare_audio_versions(
            reference_filename=reference_filename,
            version_filenames=version_filenames
        )
        
        return BaseResponse(
            message=result["message"],
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SimilarityAnalysisError(f"Failed to compare audio versions: {str(e)}")


# Trim request model
class TrimRequest(BaseModel):
    audio_url: str
    start_time: float
    end_time: float
    output_name: str


@router.post("/trim")
async def trim_audio(
    audio_url: str = Form(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    output_name: str = Form(...),
    http_request: Request = None
):
    """
    Trim an audio file based on start and end time.
    
    Args:
        audio_url: URL of the audio file to trim
        start_time: Start time in seconds
        end_time: End time in seconds
        output_name: Name for the output file
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Trim result with URL to trimmed audio
    """
    try:
        # Parse the audio URL to get the file path
        if audio_url.startswith("http"):
            # Extract path from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(audio_url)
            audio_path = parsed_url.path
        else:
            # Handle relative URLs
            audio_path = audio_url
        
        # Convert to absolute path
        if audio_path.startswith("/api/assets/audio/"):
            filename = audio_path[len("/api/assets/audio/"):]
            audio_file = OUTPUT_DIR / filename
        elif audio_path.startswith("/api/conversation/assets/audio/"):
            filename = audio_path[len("/api/conversation/assets/audio/"):]
            audio_file = TEMP_CONVERSATION_SEGMENTS_DIR / filename
        else:
            # Try to find the file in various locations
            filename = Path(audio_path).name
            possible_paths = [
                OUTPUT_DIR / filename,
                TEMP_CONVERSATION_SEGMENTS_DIR / filename,
                SPEAKERS_DIR / filename
            ]
            
            audio_file = None
            for path in possible_paths:
                if path.exists():
                    audio_file = path
                    break
            
            if not audio_file:
                raise FileNotFoundError(f"Audio file not found: {filename}")
        
        # Create output directory if it doesn't exist
        output_dir = TEMP_CONVERSATION_SEGMENTS_DIR
        output_dir.mkdir(exist_ok=True)
        
        # Generate output filename
        from webui.file_utils import safe_filename
        output_filename = safe_filename(output_name)
        if not output_filename.endswith('.wav'):
            output_filename += '.wav'
        
        output_path = output_dir / output_filename
        
        # Import trim function
        from webui.audio_processing import trim_audio_file
        
        # Trim the audio file
        trim_audio_file(
            str(audio_file),
            str(output_path),
            start_time,
            end_time
        )
        
        # Return the URL to the trimmed audio
        trimmed_audio_url = f"/api/conversation/assets/audio/{output_filename}"
        
        return BaseResponse(
            message="Audio trimmed successfully",
            details={
                "trimmed_audio_url": trimmed_audio_url,
                "output_filename": output_filename,
                "original_filename": audio_file.name,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SimilarityAnalysisError(f"Failed to trim audio: {str(e)}")
