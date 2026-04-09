"""
Timeline Service for IndexTTS2 API.
Handles timeline-based TTS generation with precise timing and multi-track support.
"""

import uuid
import time
import json
import os
import wave
from contextlib import closing
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydub import AudioSegment

from ..exceptions import TimelineError, ValidationError, ModelNotLoadedError
from ..config import settings
from ..core.app_paths import OUTPUT_DIR, SPEAKERS_DIR, TEMP_CONVERSATION_SEGMENTS_DIR, TIMELINE_PROJECTS_DIR
from ..models import (
    TimelineProject, TimelineTrack, TimelineSegment,
    TimelineProjectRequest, TimelineTrackRequest, TimelineSegmentRequest,
    EmotionKeyframe, EmotionInterpolationType
)


class TimelineService:
    """Service for handling timeline-based TTS operations."""
    
    def __init__(self, tts_service=None, conversation_service=None, emotion_service=None):
        """
        Initialize timeline service.
        
        Args:
            tts_service: TTSService instance for audio generation
            conversation_service: ConversationService instance for conversation integration
            emotion_service: EmotionService instance for emotion manipulation
        """
        self.tts_service = tts_service
        self.conversation_service = conversation_service
        self.emotion_service = emotion_service
        self.active_projects: Dict[str, Dict[str, Any]] = {}
        
        # Create timeline directory for storing project data
        self.timeline_dir = TIMELINE_PROJECTS_DIR
        self.timeline_dir.mkdir(parents=True, exist_ok=True)
    
    def create_timeline_project(
        self,
        project_name: str,
        description: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new timeline project.
        
        Args:
            project_name: Name of the timeline project
            description: Optional project description
            conversation_id: Optional conversation ID to link with
        
        Returns:
            Dict: Created project information
        """
        if not project_name or not project_name.strip():
            raise ValidationError("Project name cannot be empty")
        
        # Generate unique project ID
        project_id = str(uuid.uuid4())
        
        # Initialize project
        project = TimelineProject(
            project_id=project_id,
            project_name=project_name.strip(),
            description=description,
            conversation_id=conversation_id,
            tracks=[],
            total_duration=0.0,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Store project in memory
        self.active_projects[project_id] = {
            "project": project.dict(),
            "status": "created",
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # If conversation ID is provided, import conversation data
        if conversation_id and self.conversation_service:
            self._import_conversation_to_timeline(project_id, conversation_id)
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Created timeline project {project_id}: {project_name}")
        
        return {
            "project_id": project_id,
            "project_name": project_name,
            "status": "created",
            "message": "Timeline project created successfully"
        }
    
    def get_timeline_project(self, project_id: str) -> Dict[str, Any]:
        """
        Get timeline project information.
        
        Args:
            project_id: ID of the timeline project
        
        Returns:
            Dict: Project information
        """
        if project_id not in self.active_projects:
            # Try to load from file
            if self._load_project_from_file(project_id):
                return self.get_timeline_project(project_id)
            else:
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        
        # Update total duration
        project = project_data["project"]
        total_duration = self._calculate_total_duration(project["tracks"])
        project["total_duration"] = total_duration
        
        return {
            "project": project,
            "status": project_data["status"],
            "created_at": project_data["created_at"],
            "updated_at": project_data["updated_at"]
        }
    
    def add_track_to_project(
        self,
        project_id: str,
        track_name: str,
        speaker_filename: str
    ) -> Dict[str, Any]:
        """
        Add a new track to a timeline project.
        
        Args:
            project_id: ID of the timeline project
            track_name: Name of the track
            speaker_filename: Speaker audio filename for this track
        
        Returns:
            Dict: Created track information
        """
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        if not track_name or not track_name.strip():
            raise ValidationError("Track name cannot be empty")
        
        if not speaker_filename:
            raise ValidationError("Speaker filename is required")
        
        # Validate speaker file exists
        speaker_path = SPEAKERS_DIR / speaker_filename
        print(f"DEBUG: TimelineService.add_track_to_project - Checking speaker path: {speaker_path}")
        if not speaker_path.exists():
            raise ValidationError(f"Speaker file not found: {speaker_filename}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Generate unique track ID
        track_id = str(uuid.uuid4())
        
        # Create new track
        track = TimelineTrack(
            track_id=track_id,
            track_name=track_name.strip(),
            speaker_filename=speaker_filename,
            segments=[],
            volume=1.0,
            muted=False,
            solo=False
        )
        
        # Add track to project
        project["tracks"].append(track.dict())
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Added track {track_id} to project {project_id}")
        
        return {
            "track_id": track_id,
            "track_name": track_name,
            "project_id": project_id,
            "message": "Track added successfully"
        }
    
    def add_segment_to_track(
        self,
        project_id: str,
        track_id: str,
        text: str,
        start_time: float,
        duration: float,
        emotion_control_method: str = "from_speaker",
        emotion_reference_filename: Optional[str] = None,
        emotion_weight: float = 1.0,
        emotion_vectors: List[float] = None,
        emotion_text: Optional[str] = None,
        use_random_sampling: bool = False,
        **generation_params
    ) -> Dict[str, Any]:
        """
        Add a segment to a track in a timeline project.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            text: Text to synthesize
            start_time: Start time in seconds
            duration: Duration in seconds
            emotion_control_method: Emotion control method
            emotion_reference_filename: Emotion reference audio filename
            emotion_weight: Emotion weight
            emotion_vectors: Emotion vector components
            emotion_text: Emotion description text
            use_random_sampling: Whether to use random sampling
            **generation_params: Additional TTS generation parameters
        
        Returns:
            Dict: Created segment information
        """
        print(f"DEBUG: TimelineService.add_segment_to_track called with project_id={project_id}, track_id={track_id}")
        print(f"DEBUG: Segment data: text='{text}', start_time={start_time}, duration={duration}")
        
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty")
        
        if start_time < 0:
            raise ValidationError("Start time must be non-negative")
        
        if duration <= 0:
            raise ValidationError("Duration must be positive")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        print(f"DEBUG: Project has {len(project['tracks'])} tracks")
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        print(f"DEBUG: Found track: {track['track_name']} with {len(track.get('segments', []))} existing segments")
        
        # Check for overlapping segments
        for segment in track["segments"]:
            if (start_time < segment["start_time"] + segment["duration"] and
                start_time + duration > segment["start_time"]):
                raise ValidationError(f"Segment overlaps with existing segment at {segment['start_time']}")
        
        # Generate unique segment ID
        segment_id = str(uuid.uuid4())
        
        # Create new segment
        segment = TimelineSegment(
            segment_id=segment_id,
            text=text.strip(),
            speaker_filename=track["speaker_filename"],
            start_time=start_time,
            duration=duration,
            emotion_control_method=emotion_control_method,
            emotion_reference_filename=emotion_reference_filename,
            emotion_weight=emotion_weight,
            emotion_vectors=emotion_vectors or [],
            emotion_text=emotion_text,
            use_random_sampling=use_random_sampling,
            **generation_params
        )
        
        # Add segment to track
        track["segments"].append(segment.dict())
        
        print(f"DEBUG: Added segment to track, new segment count: {len(track['segments'])}")
        
        # Sort segments by start time
        track["segments"].sort(key=lambda x: x["start_time"])
        
        # Update project duration
        project["total_duration"] = self._calculate_total_duration(project["tracks"])
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Added segment {segment_id} to track {track_id}")
        
        return {
            "segment_id": segment_id,
            "track_id": track_id,
            "project_id": project_id,
            "start_time": start_time,
            "duration": duration,
            "message": "Segment added successfully"
        }
    
    def update_segment_timing(
        self,
        project_id: str,
        track_id: str,
        segment_id: str,
        start_time: float,
        duration: float
    ) -> Dict[str, Any]:
        """
        Update the timing of a segment in a track.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            segment_id: ID of the segment
            start_time: New start time in seconds
            duration: New duration in seconds
        
        Returns:
            Dict: Updated segment information
        """
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        if start_time < 0:
            raise ValidationError("Start time must be non-negative")
        
        if duration <= 0:
            raise ValidationError("Duration must be positive")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Find the segment
        segment = None
        for s in track["segments"]:
            if s["segment_id"] == segment_id:
                segment = s
                break
        
        if not segment:
            raise ValidationError(f"Segment not found: {segment_id}")
        
        # Check for overlapping segments (excluding current segment)
        for s in track["segments"]:
            if s["segment_id"] != segment_id:
                if (start_time < s["start_time"] + s["duration"] and
                    start_time + duration > s["start_time"]):
                    raise ValidationError(f"Segment overlaps with existing segment at {s['start_time']}")
        
        # Update segment timing
        segment["start_time"] = start_time
        segment["duration"] = duration
        
        # Sort segments by start time
        track["segments"].sort(key=lambda x: x["start_time"])
        
        # Update project duration
        project["total_duration"] = self._calculate_total_duration(project["tracks"])
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Updated timing for segment {segment_id}")
        
        return {
            "segment_id": segment_id,
            "track_id": track_id,
            "project_id": project_id,
            "start_time": start_time,
            "duration": duration,
            "message": "Segment timing updated successfully"
        }
    
    def generate_segment_audio(
        self,
        project_id: str,
        track_id: str,
        segment_id: str
    ) -> Dict[str, Any]:
        """
        Generate audio for a specific segment.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            segment_id: ID of the segment
        
        Returns:
            Dict: Generation result
        """
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        if not self.tts_service:
            raise ModelNotLoadedError("TTS service not available")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Find the segment
        segment = None
        for s in track["segments"]:
            if s["segment_id"] == segment_id:
                segment = s
                break
        
        if not segment:
            raise ValidationError(f"Segment not found: {segment_id}")
        
        try:
            # Check if emotion timeline is enabled for this segment
            if segment.get("emotion_timeline_enabled", False) and self.emotion_service:
                # Generate emotion timeline for this segment
                segment_obj = TimelineSegment(**segment)
                emotion_timeline = self.emotion_service.generate_emotion_timeline(
                    segment=segment_obj,
                    sample_rate=10  # 10 samples per second
                )
                
                # Generate TTS with emotion timeline
                tts_result = self._generate_with_emotion_timeline(
                    segment=segment,
                    emotion_timeline=emotion_timeline
                )
            else:
                # Generate standard TTS for the segment
                tts_result = self.tts_service.generate_single_tts(
                    speaker_filename=segment["speaker_filename"],
                    text=segment["text"],
                    emotion_control_method=segment["emotion_control_method"],
                    emotion_reference_filename=segment.get("emotion_reference_filename"),
                    emotion_weight=segment.get("emotion_weight", 1.0),
                    emotion_vectors=segment.get("emotion_vectors", []),
                    emotion_text=segment.get("emotion_text"),
                    use_random_sampling=segment.get("use_random_sampling", False),
                    max_text_tokens_per_segment=segment.get("max_text_tokens_per_segment", 120),
                    do_sample=segment.get("do_sample", True),
                    top_p=segment.get("top_p", 0.8),
                    top_k=segment.get("top_k", 30),
                    temperature=segment.get("temperature", 0.8),
                    length_penalty=segment.get("length_penalty", 0.0),
                    num_beams=segment.get("num_beams", 3),
                    repetition_penalty=segment.get("repetition_penalty", 10.0),
                    max_mel_tokens=segment.get("max_mel_tokens", 1500)
                )
            
            if tts_result["success"]:
                # Update segment with audio information
                segment["audio_filename"] = tts_result["audio_filename"]
                project_data["updated_at"] = time.time()
                project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Save project to file
                self._save_project_to_file(project_id)
                
                print(f"DEBUG: TimelineService - Generated audio for segment {segment_id}")
                
                return {
                    "segment_id": segment_id,
                    "track_id": track_id,
                    "project_id": project_id,
                    "audio_filename": tts_result["audio_filename"],
                    "audio_path": tts_result["audio_path"],
                    "message": "Audio generated successfully"
                }
            else:
                raise TimelineError(f"TTS generation failed: {tts_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            raise TimelineError(f"Failed to generate audio for segment: {str(e)}")
    
    def _generate_with_emotion_timeline(
        self,
        segment: Dict[str, Any],
        emotion_timeline: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate TTS audio with emotion timeline support.
        
        Args:
            segment: Segment data
            emotion_timeline: Emotion timeline data
        
        Returns:
            Dict: Generation result
        """
        if not self.tts_service:
            raise TimelineError("TTS service not available")
        
        try:
            # Get speaker audio path
            speaker_path = SPEAKERS_DIR / segment["speaker_filename"]
            print(f"DEBUG: TimelineService._generate_with_emotion_timeline - Checking speaker path: {speaker_path}")
            if not speaker_path.exists():
                raise ValidationError(f"Speaker file not found: {segment['speaker_filename']}")
            
            # Get TTS core from service
            tts_core = getattr(self.tts_service, 'tts_core', None)
            if not tts_core:
                raise TimelineError("TTS core not available")
            
            # Use TTS core to generate with emotion timeline
            result = tts_core.generate_with_emotion_timeline(
                prompt=str(speaker_path),
                text=segment["text"],
                emotion_timeline=emotion_timeline,
                segment_duration=segment["duration"],
                emotion_control_method=segment["emotion_control_method"],
                emo_ref_path=segment.get("emotion_reference_filename"),
                emo_weight=segment.get("emotion_weight", 1.0),
                use_random_sampling=segment.get("use_random_sampling", False),
                max_text_tokens_per_segment=segment.get("max_text_tokens_per_segment", 120),
                do_sample=segment.get("do_sample", True),
                top_p=segment.get("top_p", 0.8),
                top_k=segment.get("top_k", 30),
                temperature=segment.get("temperature", 0.8),
                length_penalty=segment.get("length_penalty", 0.0),
                num_beams=segment.get("num_beams", 3),
                repetition_penalty=segment.get("repetition_penalty", 10.0),
                max_mel_tokens=segment.get("max_mel_tokens", 1500)
            )
            
            if result and result.get("audio_path"):
                return {
                    "success": True,
                    "audio_filename": Path(result["audio_path"]).name,
                    "audio_path": result["audio_path"]
                }
            else:
                raise TimelineError("TTS generation with emotion timeline failed")
                
        except Exception as e:
            raise TimelineError(f"Failed to generate audio with emotion timeline: {str(e)}")
    
    def _import_conversation_to_timeline(self, project_id: str, conversation_id: str) -> None:
        """
        Import conversation data into a timeline project.
        
        Args:
            project_id: ID of the timeline project
            conversation_id: ID of the conversation to import
        """
        if not self.conversation_service:
            return
        
        try:
            # Get conversation data
            conversation_status = self.conversation_service.get_conversation_status(conversation_id)
            
            if conversation_status["status"] != "completed":
                print(f"DEBUG: TimelineService - Conversation {conversation_id} not completed, skipping import")
                return
            
            project_data = self.active_projects[project_id]
            project = project_data["project"]
            
            speaker_tracks = {}
            current_start_time = 0.0
            
            for line in conversation_status.get("lines", []):
                speaker_filename = line.get("speaker_filename")
                if speaker_filename not in speaker_tracks:
                    # Create track for this speaker
                    track_id = str(uuid.uuid4())
                    track = TimelineTrack(
                        track_id=track_id,
                        track_name=f"Speaker: {speaker_filename}",
                        speaker_filename=speaker_filename,
                        segments=[],
                        volume=1.0,
                        muted=False,
                        solo=False
                    )
                    track_data = track.dict()
                    speaker_tracks[speaker_filename] = track_data
                    project["tracks"].append(track_data)

                track = speaker_tracks[speaker_filename]
                selected_version = self._select_line_version(line)
                resolved_audio_path = self._resolve_segment_audio_path(
                    selected_version.get("audio_path") if selected_version else None
                )
                audio_filename = resolved_audio_path.name if resolved_audio_path else None
                duration = self._get_segment_duration_seconds(selected_version, resolved_audio_path)
                start_time = current_start_time

                # Create segment
                segment = TimelineSegment(
                    segment_id=str(uuid.uuid4()),
                    text=line.get("edited_text") or line.get("text", ""),
                    speaker_filename=speaker_filename,
                    start_time=start_time,
                    duration=duration,
                    audio_filename=audio_filename
                )
                
                track["segments"].append(segment.dict())
                track["segments"].sort(key=lambda item: item["start_time"])
                current_start_time += duration
            
            # Update project duration
            project["total_duration"] = self._calculate_total_duration(project["tracks"])
            project_data["updated_at"] = time.time()
            project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"DEBUG: TimelineService - Imported conversation {conversation_id} to timeline {project_id}")
            
        except Exception as e:
            print(f"DEBUG: TimelineService - Failed to import conversation {conversation_id}: {str(e)}")

    def _select_line_version(self, line: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        versions = line.get("versions", []) or []
        if not versions:
            return None

        selected_version = next((version for version in versions if version.get("is_selected")), None)
        if selected_version:
            return selected_version

        best_version_index = int(line.get("best_version_index", 0) or 0)
        if 0 <= best_version_index < len(versions):
            return versions[best_version_index]

        return versions[0]

    def _resolve_segment_audio_path(self, audio_path: Optional[str]) -> Optional[Path]:
        if not audio_path:
            return None

        raw_path = Path(audio_path)
        candidates = [raw_path]
        if raw_path.name:
            candidates.extend(
                [
                    TEMP_CONVERSATION_SEGMENTS_DIR / raw_path.name,
                    OUTPUT_DIR / raw_path.name,
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def resolve_project_audio_path(self, project_id: str, audio_filename: Optional[str]) -> Optional[Path]:
        if not audio_filename:
            return None

        return self._resolve_segment_audio_path(
            str(OUTPUT_DIR / "timeline_assets" / project_id / audio_filename)
        ) or self._resolve_segment_audio_path(audio_filename)

    def _find_track(self, project: Dict[str, Any], track_id: str) -> Optional[Dict[str, Any]]:
        for track in project.get("tracks", []):
            if track["track_id"] == track_id:
                return track
        return None

    def _find_segment(self, track: Dict[str, Any], segment_id: str) -> Optional[Dict[str, Any]]:
        for segment in track.get("segments", []):
            if segment["segment_id"] == segment_id:
                return segment
        return None

    def _split_text_for_segment(
        self,
        text: str,
        first_text: Optional[str] = None,
        second_text: Optional[str] = None,
    ) -> tuple[str, str]:
        if first_text and second_text:
            return first_text.strip(), second_text.strip()

        source_text = (text or "").strip()
        if "|" in source_text:
            before, after = source_text.split("|", 1)
            before = (first_text or before).strip()
            after = (second_text or after).strip()
            if before and after:
                return before, after

        words = source_text.split()
        if len(words) < 2:
            raise ValidationError("Segment text is too short to split automatically. Add a '|' divider in the text first.")

        midpoint = max(1, len(words) // 2)
        return " ".join(words[:midpoint]).strip(), " ".join(words[midpoint:]).strip()

    def split_segment(
        self,
        project_id: str,
        track_id: str,
        segment_id: str,
        split_offset: float,
        first_text: Optional[str] = None,
        second_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        if project_id not in self.active_projects:
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")

        project_data = self.active_projects[project_id]
        project = project_data["project"]
        track = self._find_track(project, track_id)
        if not track:
            raise ValidationError(f"Track not found: {track_id}")

        segment = self._find_segment(track, segment_id)
        if not segment:
            raise ValidationError(f"Segment not found: {segment_id}")

        duration = float(segment.get("duration", 0.0) or 0.0)
        split_offset = float(split_offset or 0.0)
        min_piece_duration = 0.2

        if split_offset <= min_piece_duration or split_offset >= duration - min_piece_duration:
            raise ValidationError(
                f"Split point must leave at least {min_piece_duration:.1f}s on both sides of the segment"
            )

        split_first_text, split_second_text = self._split_text_for_segment(
            segment.get("text", ""),
            first_text=first_text,
            second_text=second_text,
        )
        if not split_first_text or not split_second_text:
            raise ValidationError("Both split text sections must contain content")

        original_start = float(segment.get("start_time", 0.0) or 0.0)
        first_duration = round(split_offset, 3)
        second_duration = round(duration - split_offset, 3)
        second_start = round(original_start + split_offset, 3)

        segment["text"] = split_first_text
        segment["duration"] = first_duration
        segment["audio_filename"] = None

        new_segment = dict(segment)
        new_segment["segment_id"] = str(uuid.uuid4())
        new_segment["text"] = split_second_text
        new_segment["start_time"] = second_start
        new_segment["duration"] = second_duration
        new_segment["audio_filename"] = None

        track["segments"].append(new_segment)
        track["segments"].sort(key=lambda item: item["start_time"])

        project["total_duration"] = self._calculate_total_duration(project["tracks"])
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_project_to_file(project_id)

        return {
            "project_id": project_id,
            "track_id": track_id,
            "segment_id": segment_id,
            "new_segment_id": new_segment["segment_id"],
            "updated_segment": segment,
            "new_segment": new_segment,
            "message": "Segment split successfully. Regenerate audio for both halves to match the new timing.",
        }

    def get_segment_waveform(
        self,
        project_id: str,
        track_id: str,
        segment_id: str,
        bars: int = 64,
    ) -> Dict[str, Any]:
        if project_id not in self.active_projects:
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")

        project = self.active_projects[project_id]["project"]
        track = self._find_track(project, track_id)
        if not track:
            raise ValidationError(f"Track not found: {track_id}")

        segment = self._find_segment(track, segment_id)
        if not segment:
            raise ValidationError(f"Segment not found: {segment_id}")

        audio_filename = segment.get("audio_filename")
        audio_path = self.resolve_project_audio_path(project_id, audio_filename)
        if not audio_path or not audio_path.exists():
            raise ValidationError("Generate audio for this segment before requesting a waveform preview")

        bar_count = max(16, min(int(bars or 64), 256))
        with open(audio_path, "rb") as audio_handle:
            audio = AudioSegment.from_file(audio_handle).set_channels(1)
        samples = audio.get_array_of_samples()
        if not samples:
            peaks = [0.0] * bar_count
        else:
            max_possible = float(1 << (8 * audio.sample_width - 1))
            chunk_size = max(1, len(samples) // bar_count)
            peaks = []
            for index in range(bar_count):
                start = index * chunk_size
                end = len(samples) if index == bar_count - 1 else min(len(samples), start + chunk_size)
                chunk = samples[start:end]
                if not chunk:
                    peaks.append(0.0)
                    continue
                chunk_peak = max(abs(sample) for sample in chunk)
                peaks.append(round(min(1.0, chunk_peak / max_possible), 4))

        return {
            "project_id": project_id,
            "track_id": track_id,
            "segment_id": segment_id,
            "audio_filename": audio_filename,
            "duration_ms": len(audio),
            "bar_count": bar_count,
            "peaks": peaks,
        }

    def _get_segment_duration_seconds(
        self,
        version: Optional[Dict[str, Any]],
        resolved_audio_path: Optional[Path],
        fallback: float = 2.0,
    ) -> float:
        if version:
            for key in ("duration", "duration_seconds"):
                raw_duration = version.get(key)
                try:
                    if raw_duration is not None and float(raw_duration) > 0:
                        return float(raw_duration)
                except (TypeError, ValueError):
                    pass

        if resolved_audio_path and resolved_audio_path.exists():
            try:
                with closing(wave.open(str(resolved_audio_path), "rb")) as wav_file:
                    frame_rate = wav_file.getframerate()
                    if frame_rate > 0:
                        return wav_file.getnframes() / float(frame_rate)
            except (wave.Error, OSError):
                pass

        return fallback
    
    def _calculate_total_duration(self, tracks: List[Dict[str, Any]]) -> float:
        """
        Calculate the total duration of a timeline project.
        
        Args:
            tracks: List of tracks
        
        Returns:
            float: Total duration in seconds
        """
        max_end_time = 0.0
        
        for track in tracks:
            for segment in track.get("segments", []):
                end_time = segment.get("start_time", 0) + segment.get("duration", 0)
                if end_time > max_end_time:
                    max_end_time = end_time
        
        return max_end_time
    
    def _save_project_to_file(self, project_id: str) -> None:
        """
        Save project data to file.
        
        Args:
            project_id: ID of the project to save
        """
        if project_id not in self.active_projects:
            return
        
        project_file = self.timeline_dir / f"{project_id}.json"
        
        try:
            with open(project_file, 'w') as f:
                json.dump(self.active_projects[project_id], f, indent=2)
        except Exception as e:
            print(f"DEBUG: TimelineService - Failed to save project {project_id}: {str(e)}")
    
    def _load_project_from_file(self, project_id: str) -> bool:
        """
        Load project data from file.
        
        Args:
            project_id: ID of the project to load
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        project_file = self.timeline_dir / f"{project_id}.json"
        
        if not project_file.exists():
            return False
        
        try:
            with open(project_file, 'r') as f:
                project_data = json.load(f)
            
            self.active_projects[project_id] = project_data
            return True
        except Exception as e:
            print(f"DEBUG: TimelineService - Failed to load project {project_id}: {str(e)}")
            return False
    
    def delete_timeline_project(self, project_id: str) -> Dict[str, Any]:
        """
        Delete a timeline project.
        
        Args:
            project_id: ID of the timeline project to delete
        
        Returns:
            Dict: Deletion result
        """
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        # Remove from memory
        if project_id in self.active_projects:
            del self.active_projects[project_id]
        
        # Remove from file system
        project_file = self.timeline_dir / f"{project_id}.json"
        if project_file.exists():
            try:
                project_file.unlink()
                print(f"DEBUG: TimelineService - Deleted project file {project_file}")
            except Exception as e:
                print(f"DEBUG: TimelineService - Failed to delete project file {project_file}: {str(e)}")
        
        print(f"DEBUG: TimelineService - Deleted timeline project {project_id}")
        
        return {
            "project_id": project_id,
            "message": "Timeline project deleted successfully"
        }
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all timeline projects.
        
        Returns:
            List: Timeline projects
        """
        # Load projects from files
        for project_file in self.timeline_dir.glob("*.json"):
            project_id = project_file.stem
            if project_id not in self.active_projects:
                self._load_project_from_file(project_id)
        
        projects = []
        for project_id, project_data in self.active_projects.items():
            project = project_data["project"]
            projects.append({
                "project_id": project_id,
                "project_name": project.get("project_name", "Unknown"),
                "description": project.get("description"),
                "conversation_id": project.get("conversation_id"),
                "total_duration": project.get("total_duration", 0.0),
                "track_count": len(project.get("tracks", [])),
                "segment_count": sum(len(track.get("segments", [])) for track in project.get("tracks", [])),
                "created_at": project_data.get("created_at"),
                "updated_at": project_data.get("updated_at")
            })
        
        # Sort by updated time (newest first)
        projects.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return projects
    
    def update_segment_properties(
        self,
        project_id: str,
        track_id: str,
        segment_id: str,
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update properties of a segment in a track.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            segment_id: ID of the segment
            properties: Properties to update
        
        Returns:
            Dict: Updated segment information
        """
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Find the segment
        segment = None
        for s in track["segments"]:
            if s["segment_id"] == segment_id:
                segment = s
                break
        
        if not segment:
            raise ValidationError(f"Segment not found: {segment_id}")
        
        # Update segment properties
        segment.update(properties)
        
        # Update project metadata
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Updated properties for segment {segment_id}")
        
        return {
            "segment_id": segment_id,
            "track_id": track_id,
            "project_id": project_id,
            "properties": properties,
            "message": "Segment properties updated successfully"
        }
    
    def delete_segment_from_track(
        self,
        project_id: str,
        track_id: str,
        segment_id: str
    ) -> Dict[str, Any]:
        """
        Delete a segment from a track.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            segment_id: ID of the segment to delete
        
        Returns:
            Dict: Deletion result
        """
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Find and remove the segment
        segment_index = None
        for i, s in enumerate(track["segments"]):
            if s["segment_id"] == segment_id:
                segment_index = i
                break
        
        if segment_index is None:
            raise ValidationError(f"Segment not found: {segment_id}")
        
        # Remove the segment
        removed_segment = track["segments"].pop(segment_index)
        
        # Update project duration
        project["total_duration"] = self._calculate_total_duration(project["tracks"])
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Deleted segment {segment_id} from track {track_id}")
        
        return {
            "segment_id": segment_id,
            "track_id": track_id,
            "project_id": project_id,
            "removed_segment": removed_segment,
            "message": "Segment deleted successfully"
        }
    
    def move_segment_to_track(
        self,
        project_id: str,
        source_track_id: str,
        target_track_id: str,
        segment_id: str,
        new_start_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Move a segment from one track to another.
        
        Args:
            project_id: ID of the timeline project
            source_track_id: ID of the source track
            target_track_id: ID of the target track
            segment_id: ID of the segment to move
            new_start_time: Optional new start time for the segment
        
        Returns:
            Dict: Move result
        """
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the source track
        source_track = None
        for t in project["tracks"]:
            if t["track_id"] == source_track_id:
                source_track = t
                break
        
        if not source_track:
            raise ValidationError(f"Source track not found: {source_track_id}")
        
        # Find the target track
        target_track = None
        for t in project["tracks"]:
            if t["track_id"] == target_track_id:
                target_track = t
                break
        
        if not target_track:
            raise ValidationError(f"Target track not found: {target_track_id}")
        
        # Find and remove the segment from source track
        segment_index = None
        segment = None
        for i, s in enumerate(source_track["segments"]):
            if s["segment_id"] == segment_id:
                segment_index = i
                segment = s
                break
        
        if segment is None:
            raise ValidationError(f"Segment not found: {segment_id}")
        
        # Remove segment from source track
        removed_segment = source_track["segments"].pop(segment_index)
        
        # Update speaker filename if tracks have different speakers
        if source_track["speaker_filename"] != target_track["speaker_filename"]:
            removed_segment["speaker_filename"] = target_track["speaker_filename"]
            # Clear audio filename since it was generated for a different speaker
            removed_segment["audio_filename"] = None
        
        # Update start time if provided
        if new_start_time is not None:
            removed_segment["start_time"] = new_start_time
        
        # Check for overlapping segments in target track
        for s in target_track["segments"]:
            if (removed_segment["start_time"] < s["start_time"] + s["duration"] and
                removed_segment["start_time"] + removed_segment["duration"] > s["start_time"]):
                raise ValidationError(f"Segment overlaps with existing segment at {s['start_time']}")
        
        # Add segment to target track
        target_track["segments"].append(removed_segment)
        
        # Sort segments by start time
        target_track["segments"].sort(key=lambda x: x["start_time"])
        
        # Update project metadata
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Moved segment {segment_id} from track {source_track_id} to track {target_track_id}")
        
        return {
            "segment_id": segment_id,
            "source_track_id": source_track_id,
            "target_track_id": target_track_id,
            "project_id": project_id,
            "moved_segment": removed_segment,
            "message": "Segment moved successfully"
        }
    
    def toggle_track_mute(
        self,
        project_id: str,
        track_id: str
    ) -> Dict[str, Any]:
        """
        Toggle the mute state of a track.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
        
        Returns:
            Dict: Toggle result
        """
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Toggle mute state
        track["muted"] = not track.get("muted", False)
        
        # Update project metadata
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Toggled mute for track {track_id} to {track['muted']}")
        
        return {
            "track_id": track_id,
            "project_id": project_id,
            "muted": track["muted"],
            "message": f"Track {'muted' if track['muted'] else 'unmuted'} successfully"
        }
    
    def toggle_track_solo(
        self,
        project_id: str,
        track_id: str
    ) -> Dict[str, Any]:
        """
        Toggle the solo state of a track.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
        
        Returns:
            Dict: Toggle result
        """
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Toggle solo state
        track["solo"] = not track.get("solo", False)
        
        # Update project metadata
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Toggled solo for track {track_id} to {track['solo']}")
        
        return {
            "track_id": track_id,
            "project_id": project_id,
            "solo": track["solo"],
            "message": f"Track {'soloed' if track['solo'] else 'unsoloed'} successfully"
        }

    def update_track_volume(
        self,
        project_id: str,
        track_id: str,
        volume: float
    ) -> Dict[str, Any]:
        """
        Update the playback volume for a track.

        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            volume: Linear playback multiplier for the track

        Returns:
            Dict: Updated track volume information
        """
        if project_id not in self.active_projects:
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")

        if volume < 0:
            raise ValidationError("Track volume must be non-negative")

        project_data = self.active_projects[project_id]
        project = project_data["project"]

        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break

        if not track:
            raise ValidationError(f"Track not found: {track_id}")

        track["volume"] = float(volume)
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        self._save_project_to_file(project_id)

        print(f"DEBUG: TimelineService - Updated volume for track {track_id} to {track['volume']}")

        return {
            "track_id": track_id,
            "project_id": project_id,
            "volume": track["volume"],
            "message": "Track volume updated successfully"
        }
    
    def add_emotion_keyframe_to_segment(
        self,
        project_id: str,
        track_id: str,
        segment_id: str,
        timestamp: float,
        emotion_vectors: List[float],
        interpolation_type: str = "linear",
        transition_duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Add an emotion keyframe to a segment.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            segment_id: ID of the segment
            timestamp: Timestamp in seconds from segment start
            emotion_vectors: 8-dimensional emotion vector
            interpolation_type: Type of interpolation to use
            transition_duration: Duration of transition in seconds
        
        Returns:
            Dict: Result with keyframe information
        """
        if not self.emotion_service:
            raise TimelineError("Emotion service not available")
        
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Find the segment
        segment = None
        for s in track["segments"]:
            if s["segment_id"] == segment_id:
                segment = s
                break
        
        if not segment:
            raise ValidationError(f"Segment not found: {segment_id}")
        
        # Convert segment dict to TimelineSegment object
        segment_obj = TimelineSegment(**segment)
        
        # Convert interpolation type string to enum
        try:
            interpolation_enum = EmotionInterpolationType(interpolation_type)
        except ValueError:
            raise ValidationError(f"Invalid interpolation type: {interpolation_type}")
        
        # Add keyframe using emotion service
        result = self.emotion_service.add_keyframe_to_segment(
            segment=segment_obj,
            timestamp=timestamp,
            emotion_vectors=emotion_vectors,
            interpolation_type=interpolation_enum,
            transition_duration=transition_duration
        )
        
        # Update segment in track
        for i, s in enumerate(track["segments"]):
            if s["segment_id"] == segment_id:
                track["segments"][i] = segment_obj.dict()
                break
        
        # Update project metadata
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Added emotion keyframe {result['keyframe_id']} to segment {segment_id}")
        
        return {
            "project_id": project_id,
            "track_id": track_id,
            "segment_id": segment_id,
            **result
        }
    
    def update_emotion_keyframe_in_segment(
        self,
        project_id: str,
        track_id: str,
        segment_id: str,
        keyframe_id: str,
        emotion_vectors: Optional[List[float]] = None,
        interpolation_type: Optional[str] = None,
        transition_duration: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update an emotion keyframe in a segment.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            segment_id: ID of the segment
            keyframe_id: ID of the keyframe to update
            emotion_vectors: New emotion vectors (optional)
            interpolation_type: New interpolation type (optional)
            transition_duration: New transition duration (optional)
            timestamp: New timestamp (optional)
        
        Returns:
            Dict: Result with updated keyframe information
        """
        if not self.emotion_service:
            raise TimelineError("Emotion service not available")
        
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Find the segment
        segment = None
        for s in track["segments"]:
            if s["segment_id"] == segment_id:
                segment = s
                break
        
        if not segment:
            raise ValidationError(f"Segment not found: {segment_id}")
        
        # Convert segment dict to TimelineSegment object
        segment_obj = TimelineSegment(**segment)
        
        # Convert interpolation type string to enum if provided
        interpolation_enum = None
        if interpolation_type is not None:
            try:
                interpolation_enum = EmotionInterpolationType(interpolation_type)
            except ValueError:
                raise ValidationError(f"Invalid interpolation type: {interpolation_type}")
        
        # Update keyframe using emotion service
        result = self.emotion_service.update_keyframe_in_segment(
            segment=segment_obj,
            keyframe_id=keyframe_id,
            emotion_vectors=emotion_vectors,
            interpolation_type=interpolation_enum,
            transition_duration=transition_duration,
            timestamp=timestamp
        )
        
        # Update segment in track
        for i, s in enumerate(track["segments"]):
            if s["segment_id"] == segment_id:
                track["segments"][i] = segment_obj.dict()
                break
        
        # Update project metadata
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Updated emotion keyframe {keyframe_id} in segment {segment_id}")
        
        return {
            "project_id": project_id,
            "track_id": track_id,
            "segment_id": segment_id,
            **result
        }
    
    def remove_emotion_keyframe_from_segment(
        self,
        project_id: str,
        track_id: str,
        segment_id: str,
        keyframe_id: str
    ) -> Dict[str, Any]:
        """
        Remove an emotion keyframe from a segment.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            segment_id: ID of the segment
            keyframe_id: ID of the keyframe to remove
        
        Returns:
            Dict: Result with removed keyframe information
        """
        if not self.emotion_service:
            raise TimelineError("Emotion service not available")
        
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Find the segment
        segment = None
        for s in track["segments"]:
            if s["segment_id"] == segment_id:
                segment = s
                break
        
        if not segment:
            raise ValidationError(f"Segment not found: {segment_id}")
        
        # Convert segment dict to TimelineSegment object
        segment_obj = TimelineSegment(**segment)
        
        # Remove keyframe using emotion service
        result = self.emotion_service.remove_keyframe_from_segment(
            segment=segment_obj,
            keyframe_id=keyframe_id
        )
        
        # Update segment in track
        for i, s in enumerate(track["segments"]):
            if s["segment_id"] == segment_id:
                track["segments"][i] = segment_obj.dict()
                break
        
        # Update project metadata
        project_data["updated_at"] = time.time()
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save project to file
        self._save_project_to_file(project_id)
        
        print(f"DEBUG: TimelineService - Removed emotion keyframe {keyframe_id} from segment {segment_id}")
        
        return {
            "project_id": project_id,
            "track_id": track_id,
            "segment_id": segment_id,
            **result
        }
    
    def get_emotion_timeline_for_segment(
        self,
        project_id: str,
        track_id: str,
        segment_id: str,
        sample_rate: int = 10
    ) -> Dict[str, Any]:
        """
        Get emotion timeline data for a segment.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            segment_id: ID of the segment
            sample_rate: Number of samples per second
        
        Returns:
            Dict: Emotion timeline data
        """
        if not self.emotion_service:
            raise TimelineError("Emotion service not available")
        
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Find the segment
        segment = None
        for s in track["segments"]:
            if s["segment_id"] == segment_id:
                segment = s
                break
        
        if not segment:
            raise ValidationError(f"Segment not found: {segment_id}")
        
        # Convert segment dict to TimelineSegment object
        segment_obj = TimelineSegment(**segment)
        
        # Generate emotion timeline using emotion service
        timeline = self.emotion_service.generate_emotion_timeline(
            segment=segment_obj,
            sample_rate=sample_rate
        )
        
        return {
            "project_id": project_id,
            "track_id": track_id,
            "segment_id": segment_id,
            "segment_duration": segment_obj.duration,
            "emotion_timeline_enabled": segment_obj.emotion_timeline_enabled,
            "keyframe_count": len(segment_obj.emotion_keyframes),
            "sample_rate": sample_rate,
            "timeline": timeline
        }
    
    def calculate_emotion_at_timestamp(
        self,
        project_id: str,
        track_id: str,
        segment_id: str,
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Calculate emotion vector at a specific timestamp within a segment.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            segment_id: ID of the segment
            timestamp: Timestamp in seconds from segment start
        
        Returns:
            Dict: Emotion vector at the specified timestamp
        """
        if not self.emotion_service:
            raise TimelineError("Emotion service not available")
        
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Find the segment
        segment = None
        for s in track["segments"]:
            if s["segment_id"] == segment_id:
                segment = s
                break
        
        if not segment:
            raise ValidationError(f"Segment not found: {segment_id}")
        
        # Convert segment dict to TimelineSegment object
        segment_obj = TimelineSegment(**segment)
        
        # Calculate emotion at timestamp using emotion service
        emotion_vectors = self.emotion_service.calculate_emotion_at_timestamp(
            segment=segment_obj,
            timestamp=timestamp
        )
        
        return {
            "project_id": project_id,
            "track_id": track_id,
            "segment_id": segment_id,
            "timestamp": timestamp,
            "emotion_vectors": emotion_vectors
        }
    
    def preview_emotion_keyframe_change(
        self,
        project_id: str,
        track_id: str,
        segment_id: str,
        keyframe_id: str,
        preview_duration: float = 2.0
    ) -> Dict[str, Any]:
        """
        Generate a preview of emotion change around a keyframe.
        
        Args:
            project_id: ID of the timeline project
            track_id: ID of the track
            segment_id: ID of the segment
            keyframe_id: ID of the keyframe to preview
            preview_duration: Duration of preview in seconds (before and after)
        
        Returns:
            Dict: Preview data with emotion timeline
        """
        if not self.emotion_service:
            raise TimelineError("Emotion service not available")
        
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the track
        track = None
        for t in project["tracks"]:
            if t["track_id"] == track_id:
                track = t
                break
        
        if not track:
            raise ValidationError(f"Track not found: {track_id}")
        
        # Find the segment
        segment = None
        for s in track["segments"]:
            if s["segment_id"] == segment_id:
                segment = s
                break
        
        if not segment:
            raise ValidationError(f"Segment not found: {segment_id}")
        
        # Convert segment dict to TimelineSegment object
        segment_obj = TimelineSegment(**segment)
        
        # Generate preview using emotion service
        preview = self.emotion_service.preview_emotion_change(
            segment=segment_obj,
            keyframe_id=keyframe_id,
            preview_duration=preview_duration
        )
        
        return {
            "project_id": project_id,
            "track_id": track_id,
            "segment_id": segment_id,
            **preview
        }
    
    def calculate_segment_transition(
        self,
        project_id: str,
        from_track_id: str,
        from_segment_id: str,
        to_track_id: str,
        to_segment_id: str,
        transition_duration: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate emotion transition between two segments.
        
        Args:
            project_id: ID of the timeline project
            from_track_id: ID of the source track
            from_segment_id: ID of the source segment
            to_track_id: ID of the target track
            to_segment_id: ID of the target segment
            transition_duration: Duration of transition in seconds
        
        Returns:
            Dict: Transition timeline data
        """
        if not self.emotion_service:
            raise TimelineError("Emotion service not available")
        
        if project_id not in self.active_projects:
            # Try to load from file
            if not self._load_project_from_file(project_id):
                raise ValidationError(f"Timeline project not found: {project_id}")
        
        project_data = self.active_projects[project_id]
        project = project_data["project"]
        
        # Find the source track
        from_track = None
        for t in project["tracks"]:
            if t["track_id"] == from_track_id:
                from_track = t
                break
        
        if not from_track:
            raise ValidationError(f"Source track not found: {from_track_id}")
        
        # Find the source segment
        from_segment = None
        for s in from_track["segments"]:
            if s["segment_id"] == from_segment_id:
                from_segment = s
                break
        
        if not from_segment:
            raise ValidationError(f"Source segment not found: {from_segment_id}")
        
        # Find the target track
        to_track = None
        for t in project["tracks"]:
            if t["track_id"] == to_track_id:
                to_track = t
                break
        
        if not to_track:
            raise ValidationError(f"Target track not found: {to_track_id}")
        
        # Find the target segment
        to_segment = None
        for s in to_track["segments"]:
            if s["segment_id"] == to_segment_id:
                to_segment = s
                break
        
        if not to_segment:
            raise ValidationError(f"Target segment not found: {to_segment_id}")
        
        # Convert segments to TimelineSegment objects
        from_segment_obj = TimelineSegment(**from_segment)
        to_segment_obj = TimelineSegment(**to_segment)
        
        # Calculate transition using emotion service
        transition_timeline = self.emotion_service.calculate_segment_transition(
            from_segment=from_segment_obj,
            to_segment=to_segment_obj,
            transition_duration=transition_duration
        )
        
        return {
            "project_id": project_id,
            "from_track_id": from_track_id,
            "from_segment_id": from_segment_id,
            "to_track_id": to_track_id,
            "to_segment_id": to_segment_id,
            "transition_duration": transition_duration,
            "transition_timeline": transition_timeline
        }
