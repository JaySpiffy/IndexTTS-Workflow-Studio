"""
Speaker management endpoints for IndexTTS2 API.
"""

import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse

from ..models import (
    SpeakerInfo, SpeakerListResponse, SpeakerUploadResponse,
    BaseResponse, ErrorResponse
)
from ..exceptions import (
    IndexTTSException, FileNotFoundError, ValidationError,
    SpeakerError, FileUploadError
)
from ..config import settings
from ..services import SpeakerService

router = APIRouter()


def get_speaker_service(request: Request) -> SpeakerService:
    """Get speaker service from app state."""
    if not hasattr(request.app.state, 'speaker_service'):
        request.app.state.speaker_service = SpeakerService()
    
    return request.app.state.speaker_service


def get_speaker_info(speaker_path: Path) -> SpeakerInfo:
    """
    Get speaker information from file path.
    
    Args:
        speaker_path: Path to speaker file
        
    Returns:
        SpeakerInfo: Speaker information
    """
    if not speaker_path.exists():
        raise FileNotFoundError(str(speaker_path))
    
    stat = speaker_path.stat()
    name = speaker_path.stem
    
    return SpeakerInfo(
        filename=speaker_path.name,
        name=name,
        size_bytes=stat.st_size,
        size_kb=round(stat.st_size / 1024, 1)
    )


@router.get("/", response_model=SpeakerListResponse)
async def list_speakers(request: Request):
    """
    List all available speaker files.
    
    Args:
        request: HTTP request object
        
    Returns:
        SpeakerListResponse: List of speaker files
    """
    try:
        speaker_service = get_speaker_service(request)
        
        # Debug logging
        print(f"DEBUG: SpeakerService speakers_dir: {speaker_service.speakers_dir}")
        print(f"DEBUG: SpeakerService speakers_dir exists: {speaker_service.speakers_dir.exists()}")
        print(f"DEBUG: SpeakerService speakers_dir absolute: {speaker_service.speakers_dir.absolute()}")
        
        # Get speakers from service
        speakers_data = speaker_service.list_speakers()
        print(f"DEBUG: speaker_service.list_speakers() returned {len(speakers_data)} speakers")
        print(f"DEBUG: Speaker filenames returned: {[s['filename'] for s in speakers_data]}")
        print(f"DEBUG: Speaker names returned: {[s['name'] for s in speakers_data]}")
        
        # Convert to SpeakerInfo objects
        speakers = [
            SpeakerInfo(
                filename=speaker["filename"],
                name=speaker["name"],
                size_bytes=speaker["size_bytes"],
                size_kb=speaker["size_kb"]
            )
            for speaker in speakers_data
        ]
        
        print(f"DEBUG: Converted {len(speakers)} speakers to SpeakerInfo objects")
        
        return SpeakerListResponse(
            speakers=speakers,
            total_count=len(speakers),
            message=f"Found {len(speakers)} speaker files"
        )
        
    except Exception as e:
        print(f"DEBUG: Exception in list_speakers: {e}")
        import traceback
        traceback.print_exc()
        if isinstance(e, IndexTTSException):
            raise
        raise SpeakerError(f"Failed to list speakers: {str(e)}")


@router.get("/{speaker_name}", response_model=SpeakerInfo)
async def get_speaker_info(speaker_name: str, request: Request):
    """
    Get information about a specific speaker.
    
    Args:
        speaker_name: Name of the speaker file
        request: HTTP request object
        
    Returns:
        SpeakerInfo: Speaker information
    """
    try:
        speaker_service = get_speaker_service(request)
        
        # Get speaker info from service
        speaker_data = speaker_service.get_speaker_info(speaker_name)
        
        return SpeakerInfo(
            filename=speaker_data["filename"],
            name=speaker_data["name"],
            size_bytes=speaker_data["size_bytes"],
            size_kb=speaker_data["size_kb"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SpeakerError(f"Failed to get speaker info: {str(e)}")


@router.get("/{speaker_name}/audio")
async def get_speaker_audio(speaker_name: str, request: Request):
    """
    Download speaker audio file.
    
    Args:
        speaker_name: Name of the speaker file
        request: HTTP request object
        
    Returns:
        FileResponse: Audio file
    """
    try:
        speaker_service = get_speaker_service(request)
        
        # Get speaker info to get file path
        speaker_data = speaker_service.get_speaker_info(speaker_name)
        speaker_path = Path(speaker_data["path"])
        
        return FileResponse(
            path=speaker_path,
            filename=speaker_data["filename"],
            media_type="audio/wav"
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SpeakerError(f"Failed to get speaker audio: {str(e)}")


@router.post("/upload", response_model=SpeakerUploadResponse)
async def upload_speaker(
    file: UploadFile = File(...),
    name: str = None,
    request: Request = None
):
    """
    Upload a new speaker audio file.
    
    Args:
        file: Audio file to upload
        name: Optional name for the speaker
        request: HTTP request object
        
    Returns:
        SpeakerUploadResponse: Upload result
    """
    try:
        speaker_service = get_speaker_service(request)
        
        # Read file content
        content = await file.read()
        
        # Upload using service
        result = speaker_service.upload_file_from_bytes(
            file_bytes=content,
            filename=file.filename,
            custom_name=name,
            allowed_types=['audio']
        )
        
        # Convert to SpeakerInfo
        speaker_info = SpeakerInfo(
            filename=result["filename"],
            name=Path(result["filename"]).stem,
            size_bytes=result["size_bytes"],
            size_kb=result["size_kb"]
        )
        
        return SpeakerUploadResponse(
            speaker_info=speaker_info,
            message=f"Speaker uploaded successfully: {result['filename']}"
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileUploadError(f"Failed to upload speaker: {str(e)}")


@router.delete("/{speaker_name}", response_model=BaseResponse)
async def delete_speaker(speaker_name: str, request: Request):
    """
    Delete a speaker file.
    
    Args:
        speaker_name: Name of the speaker file to delete
        request: HTTP request object
        
    Returns:
        BaseResponse: Deletion result
    """
    try:
        speaker_service = get_speaker_service(request)
        
        # Delete speaker using service
        result = speaker_service.delete_speaker(speaker_name)
        
        return BaseResponse(message=result["message"])
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SpeakerError(f"Failed to delete speaker: {str(e)}")


@router.post("/{speaker_name}/validate", response_model=BaseResponse)
async def validate_speaker(speaker_name: str, request: Request):
    """
    Validate a speaker file for compatibility.
    
    Args:
        speaker_name: Name of the speaker file to validate
        request: HTTP request object
        
    Returns:
        BaseResponse: Validation result
    """
    try:
        speaker_service = get_speaker_service(request)
        
        # Validate speaker using service
        validation_result = speaker_service.validate_speaker(speaker_name)
        
        if validation_result["valid"]:
            return BaseResponse(
                message=f"Speaker file validation passed: {speaker_name}",
                details={
                    "validation_info": validation_result["info"],
                    "warnings": validation_result["warnings"]
                }
            )
        else:
            raise ValidationError(f"Speaker validation failed: {', '.join(validation_result['errors'])}")
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SpeakerError(f"Failed to validate speaker: {str(e)}")


@router.get("/statistics/overview", response_model=BaseResponse)
async def get_speaker_statistics(request: Request):
    """
    Get statistics about the speaker collection.
    
    Args:
        request: HTTP request object
        
    Returns:
        BaseResponse: Speaker statistics
    """
    try:
        speaker_service = get_speaker_service(request)
        
        # Get statistics from service
        stats = speaker_service.get_speaker_statistics()
        
        return BaseResponse(
            message="Speaker statistics retrieved",
            details=stats
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SpeakerError(f"Failed to get speaker statistics: {str(e)}")


@router.post("/search", response_model=BaseResponse)
async def search_speakers(query: str, request: Request):
    """
    Search for speakers by name.
    
    Args:
        query: Search query
        request: HTTP request object
        
    Returns:
        BaseResponse: Search results
    """
    try:
        speaker_service = get_speaker_service(request)
        
        # Search speakers using service
        results = speaker_service.search_speakers(query)
        
        return BaseResponse(
            message=f"Found {len(results)} speakers matching '{query}'",
            details={
                "query": query,
                "results": results,
                "total_count": len(results)
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SpeakerError(f"Failed to search speakers: {str(e)}")


@router.post("/batch-validate", response_model=BaseResponse)
async def batch_validate_speakers(speaker_names: List[str], request: Request):
    """
    Validate multiple speaker files.
    
    Args:
        speaker_names: List of speaker filenames to validate
        request: HTTP request object
        
    Returns:
        BaseResponse: Batch validation results
    """
    try:
        speaker_service = get_speaker_service(request)
        
        # Batch validate using service
        results = speaker_service.batch_validate_speakers(speaker_names)
        
        return BaseResponse(
            message=f"Validated {results['total_files']} speakers",
            details=results
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SpeakerError(f"Failed to batch validate speakers: {str(e)}")


@router.post("/{speaker_name}/copy-to-source-clips", response_model=BaseResponse)
async def copy_speaker_to_source_clips(speaker_name: str, request: Request):
    """
    Copy a speaker file to source clips directory.
    
    Args:
        speaker_name: Name of the speaker file to copy
        request: HTTP request object
        
    Returns:
        BaseResponse: Copy result
    """
    try:
        speaker_service = get_speaker_service(request)
        
        # Copy to source clips using service
        result = speaker_service.copy_speaker_to_source_clips(speaker_name)
        
        return BaseResponse(message=result["message"])
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SpeakerError(f"Failed to copy speaker to source clips: {str(e)}")


@router.post("/refresh-database", response_model=BaseResponse)
async def refresh_speaker_database(request: Request):
    """
    Refresh the speaker database and return updated lists.
    
    Args:
        request: HTTP request object
        
    Returns:
        BaseResponse: Refresh result
    """
    try:
        speaker_service = get_speaker_service(request)
        
        # Refresh database using service
        result = speaker_service.refresh_speaker_database()
        
        return BaseResponse(
            message=result["message"],
            details={
                "source_clips": result["source_clips"],
                "speakers": result["speakers"],
                "total_speakers": result["total_speakers"]
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise SpeakerError(f"Failed to refresh speaker database: {str(e)}")