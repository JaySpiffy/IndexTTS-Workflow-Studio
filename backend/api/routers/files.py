"""
File management endpoints for IndexTTS2 API.
Handles file uploads, downloads, and management.
"""

import os
import uuid
import mimetypes
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse

from ..models import (
    FileUploadResponse, FileListResponse, BaseResponse
)
from ..exceptions import (
    IndexTTSException, FileNotFoundError, ValidationError,
    FileUploadError
)
from ..config import settings
from ..services import FileService

router = APIRouter()


def get_file_service(request: Request) -> FileService:
    """Get file service from app state."""
    if not hasattr(request.app.state, 'file_service'):
        request.app.state.file_service = FileService()
    
    return request.app.state.file_service


def get_file_type(filename: str) -> str:
    """
    Determine file type based on extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        str: File type category
    """
    ext = Path(filename).suffix.lower()
    
    if ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        return 'audio'
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return 'video'
    elif ext in ['.txt', '.json', '.csv']:
        return 'text'
    elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
        return 'image'
    else:
        return 'other'


def validate_file_type(filename: str, allowed_types: List[str] = None) -> bool:
    """
    Validate file type against allowed types.
    
    Args:
        filename: Name of the file
        allowed_types: List of allowed file types
        
    Returns:
        bool: True if file type is allowed
    """
    if allowed_types is None:
        # Default allowed types
        allowed_types = ['audio', 'video', 'text']
    
    file_type = get_file_type(filename)
    return file_type in allowed_types


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form("general"),
    custom_name: str = Form(None),
    request: Request = None
):
    """
    Upload a file to the server.
    
    Args:
        file: File to upload
        category: Category for the file (general, audio, video, etc.)
        custom_name: Optional custom name for the file
        request: HTTP request object
        
    Returns:
        FileUploadResponse: Upload result
    """
    try:
        file_service = get_file_service(request)
        
        # Read file content
        content = await file.read()
        
        # Upload file using service
        result = file_service.upload_file_from_bytes(
            file_bytes=content,
            filename=file.filename,
            category=category,
            custom_name=custom_name
        )
        
        return FileUploadResponse(
            filename=result["filename"],
            file_path=result["file_path"],
            size_bytes=result["size_bytes"],
            content_type=result["content_type"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileUploadError(f"Failed to upload file: {str(e)}")


@router.get("/download/{category}/{filename}")
async def download_file(category: str, filename: str, request: Request):
    """
    Download a file from the server.
    
    Args:
        category: Category of the file
        filename: Name of the file
        request: HTTP request object
        
    Returns:
        FileResponse: File download
    """
    try:
        file_service = get_file_service(request)
        
        # Get file info from service
        result = file_service.download_file(category, filename)
        
        return FileResponse(
            path=result["file_path"],
            filename=filename,
            media_type=result["media_type"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileNotFoundError(f"Failed to download file: {str(e)}")


@router.get("/list/{category}", response_model=FileListResponse)
async def list_files(category: str, request: Request):
    """
    List files in a specific category.
    
    Args:
        category: Category to list files from
        request: HTTP request object
        
    Returns:
        FileListResponse: List of files
    """
    try:
        file_service = get_file_service(request)
        
        # List files using service
        result = file_service.list_files(category)
        
        return FileListResponse(
            files=result["files"],
            total_count=result["total_count"],
            message=result["message"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileUploadError(f"Failed to list files: {str(e)}")


@router.delete("/{category}/{filename}", response_model=BaseResponse)
async def delete_file(category: str, filename: str, request: Request):
    """
    Delete a file from the server.
    
    Args:
        category: Category of the file
        filename: Name of the file to delete
        request: HTTP request object
        
    Returns:
        BaseResponse: Deletion result
    """
    try:
        file_service = get_file_service(request)
        
        # Delete file using service
        result = file_service.delete_file(category, filename)
        
        return BaseResponse(message=result["message"])
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileUploadError(f"Failed to delete file: {str(e)}")


@router.get("/categories")
async def list_categories(request: Request):
    """
    List all available file categories.
    
    Args:
        request: HTTP request object
        
    Returns:
        BaseResponse: List of categories
    """
    try:
        file_service = get_file_service(request)
        
        # List categories using service
        result = file_service.list_categories()
        
        return BaseResponse(
            message=result["message"],
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileUploadError(f"Failed to list categories: {str(e)}")


@router.post("/cleanup/{category}", response_model=BaseResponse)
async def cleanup_category(
    category: str,
    older_than_days: int = 30,
    dry_run: bool = True,
    request: Request = None
):
    """
    Clean up old files in a category.
    
    Args:
        category: Category to clean up
        older_than_days: Delete files older than this many days
        dry_run: If True, only show what would be deleted
        request: HTTP request object
        
    Returns:
        BaseResponse: Cleanup result
    """
    try:
        file_service = get_file_service(request)
        
        # Cleanup category using service
        result = file_service.cleanup_category(
            category=category,
            older_than_days=older_than_days,
            dry_run=dry_run
        )
        
        return BaseResponse(
            message=result["message"],
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileUploadError(f"Failed to cleanup category: {str(e)}")


@router.get("/storage-info")
async def get_storage_info(request: Request):
    """
    Get storage information and statistics.
    
    Args:
        request: HTTP request object
        
    Returns:
        BaseResponse: Storage information
    """
    try:
        file_service = get_file_service(request)
        
        # Get storage info using service
        result = file_service.get_storage_info()
        
        return BaseResponse(
            message=result["message"],
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileUploadError(f"Failed to get storage info: {str(e)}")


@router.post("/move-file")
async def move_file(
    source_category: str,
    source_filename: str,
    target_category: str,
    new_name: Optional[str] = None,
    request: Request = None
):
    """
    Move a file from one category to another.
    
    Args:
        source_category: Source category
        source_filename: Source filename
        target_category: Target category
        new_name: Optional new filename
        request: HTTP request object
        
    Returns:
        BaseResponse: Move result
    """
    try:
        file_service = get_file_service(request)
        
        # Move file using service
        result = file_service.move_file(
            source_category=source_category,
            source_filename=source_filename,
            target_category=target_category,
            new_name=new_name
        )
        
        return BaseResponse(
            message=result["message"],
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileUploadError(f"Failed to move file: {str(e)}")


@router.post("/copy-file")
async def copy_file(
    source_category: str,
    source_filename: str,
    target_category: str,
    new_name: Optional[str] = None,
    request: Request = None
):
    """
    Copy a file from one category to another.
    
    Args:
        source_category: Source category
        source_filename: Source filename
        target_category: Target category
        new_name: Optional new filename
        request: HTTP request object
        
    Returns:
        BaseResponse: Copy result
    """
    try:
        file_service = get_file_service(request)
        
        # Copy file using service
        result = file_service.copy_file(
            source_category=source_category,
            source_filename=source_filename,
            target_category=target_category,
            new_name=new_name
        )
        
        return BaseResponse(
            message=result["message"],
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise FileUploadError(f"Failed to copy file: {str(e)}")