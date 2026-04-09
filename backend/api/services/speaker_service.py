"""
Speaker Service for IndexTTS2 API.
Handles speaker file management and operations.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ..core.app_paths import SOURCE_CLIPS_DIR, SPEAKERS_DIR
from ..exceptions import SpeakerError, FileNotFoundError, ValidationError
from ..config import settings


class SpeakerService:
    """Service for handling speaker management operations."""
    
    def __init__(self):
        """Initialize speaker service."""
        self.speakers_dir = SPEAKERS_DIR
        self.source_clips_dir = SOURCE_CLIPS_DIR
        print(f"DEBUG: speakers_dir: {self.speakers_dir}")
        print(f"DEBUG: source_clips_dir: {self.source_clips_dir}")
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.speakers_dir.mkdir(parents=True, exist_ok=True)
        self.source_clips_dir.mkdir(parents=True, exist_ok=True)
    
    def list_speakers(self) -> List[Dict[str, Any]]:
        """
        List all available speaker files.
        
        Returns:
            List: Speaker information
        """
        speakers = []
        
        for speaker_path in self.speakers_dir.glob("*.wav"):
            if speaker_path.is_file():
                stat = speaker_path.stat()
                speakers.append({
                    "filename": speaker_path.name,
                    "name": speaker_path.stem,
                    "size_bytes": stat.st_size,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "path": str(speaker_path)
                })
        
        # Sort by filename
        speakers.sort(key=lambda x: x["filename"])
        return speakers
    
    def get_speaker_info(self, speaker_filename: str) -> Dict[str, Any]:
        """
        Get information about a specific speaker.
        
        Args:
            speaker_filename: Name of the speaker file
        
        Returns:
            Dict: Speaker information
        """
        if not speaker_filename.endswith('.wav'):
            speaker_filename += '.wav'
        
        speaker_path = self.speakers_dir / speaker_filename
        
        if not speaker_path.exists():
            raise FileNotFoundError(f"Speaker file not found: {speaker_filename}")
        
        stat = speaker_path.stat()
        
        return {
            "filename": speaker_path.name,
            "name": speaker_path.stem,
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 1),
            "path": str(speaker_path),
            "created_time": stat.st_ctime,
            "modified_time": stat.st_mtime
        }
    
    def upload_speaker(
        self, 
        source_path: str, 
        custom_name: Optional[str] = None,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Upload a new speaker audio file.
        
        Args:
            source_path: Path to source audio file
            custom_name: Optional custom name for the speaker
            overwrite: Whether to overwrite existing file
        
        Returns:
            Dict: Upload result
        """
        source_file = Path(source_path)
        
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Validate file type
        if source_file.suffix.lower() != '.wav':
            raise ValidationError("Only WAV files are supported for speakers")
        
        # Validate file size
        if source_file.stat().st_size > settings.max_file_size:
            raise ValidationError(f"File too large (max {settings.max_file_size // (1024*1024)}MB)")
        
        # Determine filename
        if custom_name:
            filename = custom_name.strip()
            if not filename.endswith('.wav'):
                filename += '.wav'
        else:
            filename = source_file.name
        
        speaker_path = self.speakers_dir / filename
        
        # Handle existing file
        if speaker_path.exists() and not overwrite:
            # Create unique filename
            counter = 1
            stem = Path(filename).stem
            while speaker_path.exists():
                speaker_path = self.speakers_dir / f"{stem}_{counter}.wav"
                counter += 1
        
        # Copy file
        try:
            shutil.copy2(source_file, speaker_path)
        except Exception as e:
            raise SpeakerError(f"Failed to copy speaker file: {str(e)}")
        
        return self.get_speaker_info(speaker_path.name)
    
    def delete_speaker(self, speaker_filename: str) -> Dict[str, Any]:
        """
        Delete a speaker file.
        
        Args:
            speaker_filename: Name of the speaker file to delete
        
        Returns:
            Dict: Deletion result
        """
        if not speaker_filename.endswith('.wav'):
            speaker_filename += '.wav'
        
        speaker_path = self.speakers_dir / speaker_filename
        
        if not speaker_path.exists():
            raise FileNotFoundError(f"Speaker file not found: {speaker_filename}")
        
        # Get info before deletion
        speaker_info = self.get_speaker_info(speaker_filename)
        
        try:
            speaker_path.unlink()
        except Exception as e:
            raise SpeakerError(f"Failed to delete speaker file: {str(e)}")
        
        return {
            "success": True,
            "message": f"Speaker deleted successfully: {speaker_filename}",
            "deleted_speaker": speaker_info
        }
    
    def validate_speaker(self, speaker_filename: str) -> Dict[str, Any]:
        """
        Validate a speaker file for compatibility.
        
        Args:
            speaker_filename: Name of the speaker file to validate
        
        Returns:
            Dict: Validation result
        """
        if not speaker_filename.endswith('.wav'):
            speaker_filename += '.wav'
        
        speaker_path = self.speakers_dir / speaker_filename
        
        if not speaker_path.exists():
            raise FileNotFoundError(f"Speaker file not found: {speaker_filename}")
        
        # Basic validation
        stat = speaker_path.stat()
        validation_results = {
            "filename": speaker_filename,
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check file size
        if stat.st_size == 0:
            validation_results["valid"] = False
            validation_results["errors"].append("Speaker file is empty")
        elif stat.st_size > 100 * 1024 * 1024:  # 100MB
            validation_results["warnings"].append("Speaker file is very large (>100MB)")
        
        validation_results["info"]["size_bytes"] = stat.st_size
        validation_results["info"]["size_kb"] = round(stat.st_size / 1024, 1)
        
        # Try to get audio info
        try:
            from backend.api.core.audio_processing import check_audio_quality
            quality_info = check_audio_quality(str(speaker_path))
            validation_results["info"]["audio_quality"] = quality_info
        except Exception as e:
            validation_results["warnings"].append(f"Could not analyze audio quality: {str(e)}")
        
        return validation_results
    
    def refresh_speaker_database(self) -> Dict[str, Any]:
        """
        Refresh the speaker database and return updated lists.
        
        Returns:
            Dict: Refresh result with speaker lists
        """
        try:
            # Import here to avoid circular imports
            from backend.api.core.audio_processing import refresh_speaker_lists
            
            source_list, speaker_list = refresh_speaker_lists()
            
            return {
                "success": True,
                "message": "Speaker database refreshed successfully",
                "source_clips": source_list,
                "speakers": speaker_list,
                "total_speakers": len(self.list_speakers())
            }
            
        except Exception as e:
            raise SpeakerError(f"Failed to refresh speaker database: {str(e)}")
    
    def get_speaker_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the speaker collection.
        
        Returns:
            Dict: Speaker statistics
        """
        speakers = self.list_speakers()
        
        if not speakers:
            return {
                "total_speakers": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0.0,
                "average_size_kb": 0.0,
                "largest_speaker": None,
                "smallest_speaker": None
            }
        
        total_size = sum(s["size_bytes"] for s in speakers)
        average_size = total_size / len(speakers)
        
        largest = max(speakers, key=lambda x: x["size_bytes"])
        smallest = min(speakers, key=lambda x: x["size_bytes"])
        
        return {
            "total_speakers": len(speakers),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "average_size_kb": round(average_size / 1024, 1),
            "largest_speaker": {
                "filename": largest["filename"],
                "size_kb": largest["size_kb"]
            },
            "smallest_speaker": {
                "filename": smallest["filename"],
                "size_kb": smallest["size_kb"]
            }
        }
    
    def search_speakers(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for speakers by name.
        
        Args:
            query: Search query
        
        Returns:
            List: Matching speakers
        """
        speakers = self.list_speakers()
        query_lower = query.lower()
        
        matching_speakers = []
        for speaker in speakers:
            if (query_lower in speaker["name"].lower() or 
                query_lower in speaker["filename"].lower()):
                matching_speakers.append(speaker)
        
        return matching_speakers
    
    def batch_validate_speakers(self, speaker_filenames: List[str]) -> Dict[str, Any]:
        """
        Validate multiple speaker files.
        
        Args:
            speaker_filenames: List of speaker filenames to validate
        
        Returns:
            Dict: Batch validation results
        """
        results = {
            "total_files": len(speaker_filenames),
            "valid_files": 0,
            "invalid_files": 0,
            "validation_results": []
        }
        
        for filename in speaker_filenames:
            try:
                validation = self.validate_speaker(filename)
                results["validation_results"].append(validation)
                
                if validation["valid"]:
                    results["valid_files"] += 1
                else:
                    results["invalid_files"] += 1
                    
            except Exception as e:
                results["validation_results"].append({
                    "filename": filename,
                    "valid": False,
                    "errors": [str(e)],
                    "warnings": [],
                    "info": {}
                })
                results["invalid_files"] += 1
        
        return results
    
    def copy_speaker_to_source_clips(self, speaker_filename: str) -> Dict[str, Any]:
        """
        Copy a speaker file to source clips directory.
        
        Args:
            speaker_filename: Name of the speaker file to copy
        
        Returns:
            Dict: Copy result
        """
        if not speaker_filename.endswith('.wav'):
            speaker_filename += '.wav'
        
        speaker_path = self.speakers_dir / speaker_filename
        
        if not speaker_path.exists():
            raise FileNotFoundError(f"Speaker file not found: {speaker_filename}")
        
        target_path = self.source_clips_dir / speaker_filename
        
        # Handle existing file
        if target_path.exists():
            counter = 1
            stem = Path(speaker_filename).stem
            while target_path.exists():
                target_path = self.source_clips_dir / f"{stem}_{counter}.wav"
                counter += 1
        
        try:
            shutil.copy2(speaker_path, target_path)
        except Exception as e:
            raise SpeakerError(f"Failed to copy speaker to source clips: {str(e)}")
        
        return {
            "success": True,
            "message": f"Speaker copied to source clips: {target_path.name}",
            "source_path": str(speaker_path),
            "target_path": str(target_path)
        }
