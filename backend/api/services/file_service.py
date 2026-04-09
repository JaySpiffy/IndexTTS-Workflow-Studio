"""
File Service for IndexTTS2 API.
Handles file uploads, downloads, and management operations.
"""

import os
import uuid
import mimetypes
import time
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..core.app_paths import OUTPUT_DIR, SOURCE_CLIPS_DIR, SPEAKERS_DIR, TEMP_CONVERSATION_SEGMENTS_DIR, UPLOAD_DIR
from ..exceptions import FileUploadError, FileNotFoundError, ValidationError
from ..config import settings


class FileService:
    """Service for handling file management operations."""
    
    def __init__(self):
        """Initialize file service."""
        self.upload_dir = UPLOAD_DIR
        self._ensure_directories()

    def _resolve_category_dir(self, category: str) -> Path:
        """Map logical file categories to their canonical on-disk directories."""
        category_map = {
            "source_clips": SOURCE_CLIPS_DIR,
            "speakers": SPEAKERS_DIR,
            "outputs": OUTPUT_DIR,
            "temp_conversation_segments": TEMP_CONVERSATION_SEGMENTS_DIR,
        }
        return category_map.get(category, self.upload_dir / category)
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create common categories
        categories = ["general", "audio", "video", "text", "temp"]
        for category in categories:
            self._resolve_category_dir(category).mkdir(parents=True, exist_ok=True)
    
    def get_file_type(self, filename: str) -> str:
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
    
    def validate_file_type(self, filename: str, allowed_types: List[str] = None) -> bool:
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
        
        file_type = self.get_file_type(filename)
        return file_type in allowed_types
    
    def upload_file(
        self,
        source_path: str,
        category: str = "general",
        custom_name: Optional[str] = None,
        allowed_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a file to the server.
        
        Args:
            source_path: Path to source file
            category: Category for the file
            custom_name: Optional custom name for the file
            allowed_types: List of allowed file types
        
        Returns:
            Dict: Upload result
        """
        source_file = Path(source_path)
        
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Validate file size
        if source_file.stat().st_size > settings.max_file_size:
            raise ValidationError(f"File too large (max {settings.max_file_size // (1024*1024)}MB)")
        
        # Validate file type
        if not self.validate_file_type(source_file.name, allowed_types):
            raise ValidationError(f"File type not allowed: {source_file.name}")
        
        # Determine upload directory
        upload_dir = self._resolve_category_dir(category)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        if custom_name:
            filename = custom_name.strip()
            # Preserve original extension if not provided
            if not Path(filename).suffix and Path(source_file.name).suffix:
                filename += Path(source_file.name).suffix
        else:
            filename = source_file.name
        
        # Ensure unique filename
        file_path = upload_dir / filename
        counter = 1
        while file_path.exists():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            file_path = upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        # Copy file
        try:
            shutil.copy2(source_file, file_path)
        except Exception as e:
            raise FileUploadError(f"Failed to copy file: {str(e)}")
        
        # Get content type
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"
        
        return {
            "success": True,
            "filename": file_path.name,
            "file_path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "size_kb": round(file_path.stat().st_size / 1024, 1),
            "content_type": content_type,
            "category": category,
            "file_type": self.get_file_type(file_path.name)
        }
    
    def upload_file_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        category: str = "general",
        custom_name: Optional[str] = None,
        allowed_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a file from bytes data.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            category: Category for the file
            custom_name: Optional custom name for the file
            allowed_types: List of allowed file types
        
        Returns:
            Dict: Upload result
        """
        # Validate file size
        if len(file_bytes) > settings.max_file_size:
            raise ValidationError(f"File too large (max {settings.max_file_size // (1024*1024)}MB)")
        
        # Validate file type
        if not self.validate_file_type(filename, allowed_types):
            raise ValidationError(f"File type not allowed: {filename}")
        
        # Determine upload directory
        upload_dir = self._resolve_category_dir(category)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        if custom_name:
            final_filename = custom_name.strip()
            # Preserve original extension if not provided
            if not Path(final_filename).suffix and Path(filename).suffix:
                final_filename += Path(filename).suffix
        else:
            final_filename = filename
        
        # Ensure unique filename
        file_path = upload_dir / final_filename
        counter = 1
        while file_path.exists():
            stem = Path(final_filename).stem
            suffix = Path(final_filename).suffix
            file_path = upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        # Save file
        try:
            with open(file_path, "wb") as f:
                f.write(file_bytes)
        except Exception as e:
            raise FileUploadError(f"Failed to save file: {str(e)}")
        
        # Get content type
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"
        
        return {
            "success": True,
            "filename": file_path.name,
            "file_path": str(file_path),
            "size_bytes": len(file_bytes),
            "size_kb": round(len(file_bytes) / 1024, 1),
            "content_type": content_type,
            "category": category,
            "file_type": self.get_file_type(file_path.name)
        }
    
    def download_file(self, category: str, filename: str) -> Dict[str, Any]:
        """
        Get file information for download.
        
        Args:
            category: Category of the file
            filename: Name of the file
        
        Returns:
            Dict: File information
        """
        file_path = self._resolve_category_dir(category) / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        # Determine media type
        media_type, _ = mimetypes.guess_type(str(file_path))
        if not media_type:
            media_type = "application/octet-stream"
        
        return {
            "success": True,
            "file_path": str(file_path),
            "filename": filename,
            "media_type": media_type,
            "size_bytes": file_path.stat().st_size
        }
    
    def list_files(self, category: str, file_type: Optional[str] = None) -> Dict[str, Any]:
        """
        List files in a specific category.
        
        Args:
            category: Category to list files from
            file_type: Optional file type filter
        
        Returns:
            Dict: List of files
        """
        category_dir = self._resolve_category_dir(category)
        
        if not category_dir.exists():
            category_dir.mkdir(parents=True, exist_ok=True)
        
        files = []
        for file_path in category_dir.iterdir():
            if file_path.is_file():
                # Apply file type filter if specified
                current_file_type = self.get_file_type(file_path.name)
                if file_type and current_file_type != file_type:
                    continue
                
                stat = file_path.stat()
                
                # Get file info
                media_type, _ = mimetypes.guess_type(str(file_path))
                if not media_type:
                    media_type = "application/octet-stream"
                
                files.append({
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "size_bytes": stat.st_size,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "content_type": media_type,
                    "file_type": current_file_type,
                    "created_time": stat.st_ctime,
                    "modified_time": stat.st_mtime
                })
        
        # Sort by filename
        files.sort(key=lambda x: x["filename"])
        
        return {
            "success": True,
            "files": files,
            "total_count": len(files),
            "category": category,
            "file_type_filter": file_type
        }
    
    def delete_file(self, category: str, filename: str) -> Dict[str, Any]:
        """
        Delete a file from the server.
        
        Args:
            category: Category of the file
            filename: Name of the file to delete
        
        Returns:
            Dict: Deletion result
        """
        file_path = self._resolve_category_dir(category) / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        # Get info before deletion
        file_info = {
            "filename": filename,
            "size_bytes": file_path.stat().st_size,
            "category": category
        }
        
        try:
            file_path.unlink()
        except Exception as e:
            raise FileUploadError(f"Failed to delete file: {str(e)}")
        
        return {
            "success": True,
            "message": f"File deleted successfully: {filename}",
            "deleted_file": file_info
        }
    
    def list_categories(self) -> Dict[str, Any]:
        """
        List all available file categories.
        
        Returns:
            Dict: List of categories
        """
        categories = []
        for category_path in self.upload_dir.iterdir():
            if category_path.is_dir():
                # Count files in category
                file_count = len(list(category_path.iterdir()))
                
                categories.append({
                    "name": category_path.name,
                    "path": str(category_path),
                    "file_count": file_count
                })
        
        # Sort by category name
        categories.sort(key=lambda x: x["name"])
        
        return {
            "success": True,
            "categories": categories,
            "upload_directory": str(self.upload_dir)
        }
    
    def cleanup_category(
        self,
        category: str,
        older_than_days: int = 30,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Clean up old files in a category.
        
        Args:
            category: Category to clean up
            older_than_days: Delete files older than this many days
            dry_run: If True, only show what would be deleted
        
        Returns:
            Dict: Cleanup result
        """
        category_dir = self._resolve_category_dir(category)
        
        if not category_dir.exists():
            raise ValidationError(f"Category not found: {category}")
        
        current_time = time.time()
        cutoff_time = current_time - (older_than_days * 24 * 60 * 60)
        
        files_to_delete = []
        total_size = 0
        
        for file_path in category_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                if stat.st_mtime < cutoff_time:
                    files_to_delete.append({
                        "filename": file_path.name,
                        "size_bytes": stat.st_size,
                        "size_kb": round(stat.st_size / 1024, 1),
                        "modified_time": stat.st_mtime,
                        "days_old": (current_time - stat.st_mtime) / (24 * 60 * 60)
                    })
                    total_size += stat.st_size
        
        # Delete files if not dry run
        deleted_files = []
        if not dry_run:
            for file_info in files_to_delete:
                file_path = category_dir / file_info["filename"]
                try:
                    file_path.unlink()
                    deleted_files.append(file_info["filename"])
                except Exception as e:
                    print(f"Failed to delete {file_info['filename']}: {e}")
        
        return {
            "success": True,
            "category": category,
            "older_than_days": older_than_days,
            "dry_run": dry_run,
            "files_found": len(files_to_delete),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files_to_delete": files_to_delete,
            "deleted_files": deleted_files if not dry_run else []
        }
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage information and statistics.
        
        Returns:
            Dict: Storage information
        """
        # Get disk usage
        total, used, free = shutil.disk_usage(self.upload_dir)
        
        # Count files by category
        category_stats = {}
        total_files = 0
        total_size = 0
        
        for category_path in self.upload_dir.iterdir():
            if category_path.is_dir():
                files = list(category_path.iterdir())
                file_count = len(files)
                size = sum(f.stat().st_size for f in files if f.is_file())
                
                category_stats[category_path.name] = {
                    "file_count": file_count,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2)
                }
                
                total_files += file_count
                total_size += size
        
        return {
            "success": True,
            "upload_directory": str(self.upload_dir),
            "disk_usage": {
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
                "total_gb": round(total / (1024 ** 3), 2),
                "used_gb": round(used / (1024 ** 3), 2),
                "free_gb": round(free / (1024 ** 3), 2),
                "usage_percent": round((used / total) * 100, 1)
            },
            "file_statistics": {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "categories": category_stats
            }
        }
    
    def move_file(
        self,
        source_category: str,
        source_filename: str,
        target_category: str,
        new_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Move a file from one category to another.
        
        Args:
            source_category: Source category
            source_filename: Source filename
            target_category: Target category
            new_name: Optional new filename
        
        Returns:
            Dict: Move result
        """
        source_path = self.upload_dir / source_category / source_filename
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_filename}")
        
        # Ensure target directory exists
        target_dir = self.upload_dir / target_category
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine target filename
        target_filename = new_name or source_filename
        target_path = target_dir / target_filename
        
        # Handle existing file
        if target_path.exists() and target_path != source_path:
            counter = 1
            stem = Path(target_filename).stem
            suffix = Path(target_filename).suffix
            while target_path.exists():
                target_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        
        try:
            shutil.move(str(source_path), str(target_path))
        except Exception as e:
            raise FileUploadError(f"Failed to move file: {str(e)}")
        
        return {
            "success": True,
            "message": f"File moved from {source_category}/{source_filename} to {target_category}/{target_path.name}",
            "source_path": str(source_path),
            "target_path": str(target_path),
            "target_filename": target_path.name
        }
    
    def copy_file(
        self,
        source_category: str,
        source_filename: str,
        target_category: str,
        new_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Copy a file from one category to another.
        
        Args:
            source_category: Source category
            source_filename: Source filename
            target_category: Target category
            new_name: Optional new filename
        
        Returns:
            Dict: Copy result
        """
        source_path = self.upload_dir / source_category / source_filename
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_filename}")
        
        # Ensure target directory exists
        target_dir = self.upload_dir / target_category
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine target filename
        target_filename = new_name or source_filename
        target_path = target_dir / target_filename
        
        # Handle existing file
        if target_path.exists():
            counter = 1
            stem = Path(target_filename).stem
            suffix = Path(target_filename).suffix
            while target_path.exists():
                target_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        
        try:
            shutil.copy2(source_path, target_path)
        except Exception as e:
            raise FileUploadError(f"Failed to copy file: {str(e)}")
        
        return {
            "success": True,
            "message": f"File copied from {source_category}/{source_filename} to {target_category}/{target_path.name}",
            "source_path": str(source_path),
            "target_path": str(target_path),
            "target_filename": target_path.name
        }
