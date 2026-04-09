"""
Timeline management endpoints for IndexTTS2 API.
Handles timeline-based TTS generation with precise timing and multi-track support.
"""

import uuid
import time
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import FileResponse

from ..models import (
    TimelineProjectRequest, TimelineProjectResponse,
    TimelineTrackRequest, TimelineTrackResponse, TimelineTrackVolumeRequest,
    TimelineSegmentRequest, TimelineSegmentResponse,
    TimelineSegmentUpdateRequest, TimelineSegmentSplitRequest, TimelineSegmentSplitResponse,
    TimelineExportRequest, TimelineExportResponse, TimelineWaveformResponse,
    BaseResponse
)
from ..exceptions import (
    IndexTTSException, ModelNotLoadedError, ValidationError,
    TimelineError
)
from ..config import settings
from ..core.audio_mixing import mix_audio_files_at_positions
from ..core.app_paths import OUTPUT_DIR, TEMP_CONVERSATION_SEGMENTS_DIR
from ..services import TTSService, ConversationService, TimelineService

router = APIRouter()


def _guess_audio_media_type(audio_path: str) -> str:
    guessed_type, _ = mimetypes.guess_type(audio_path)
    return guessed_type or "application/octet-stream"


def get_tts_service(request: Request) -> TTSService:
    """Get TTS service from app state."""
    app = request.app
    if not getattr(app.state, 'model_loaded', False):
        raise ModelNotLoadedError()
    
    if not hasattr(app.state, 'tts_service'):
        tts_core = getattr(app.state, 'tts_core', None) or getattr(app.state, 'tts', None)
        app.state.tts_service = TTSService(
            tts_core=tts_core,
            cmd_args=app.state.cmd_args if hasattr(app.state, 'cmd_args') else None
        )
    
    return app.state.tts_service


def get_conversation_service(request: Request) -> ConversationService:
    """Get conversation service from app state."""
    app = request.app
    if not getattr(app.state, 'model_loaded', False):
        raise ModelNotLoadedError()
    
    if not hasattr(app.state, 'conversation_service'):
        app.state.conversation_service = ConversationService(
            conversation_manager=app.state.conversation_manager
        )
    
    return app.state.conversation_service


def get_timeline_service(request: Request) -> TimelineService:
    """Get timeline service from app state."""
    app = request.app
    if not getattr(app.state, 'model_loaded', False):
        raise ModelNotLoadedError()
    
    if not hasattr(app.state, 'timeline_service'):
        tts_service = get_tts_service(request)
        conversation_service = get_conversation_service(request)
        app.state.timeline_service = TimelineService(
            tts_service=tts_service,
            conversation_service=conversation_service
        )
    
    return app.state.timeline_service


@router.post("/create/{conversation_id}", response_model=TimelineProjectResponse)
async def create_timeline_from_conversation(
    conversation_id: str,
    request: TimelineProjectRequest,
    http_request: Request
):
    """
    Create a timeline project from an existing conversation.
    
    Args:
        conversation_id: ID of the conversation to create timeline from
        request: Timeline project creation request
        http_request: HTTP request object
        
    Returns:
        TimelineProjectResponse: Created timeline project
    """
    try:
        timeline_service = get_timeline_service(http_request)
        
        # Create timeline project linked to conversation
        result = timeline_service.create_timeline_project(
            project_name=request.project_name,
            description=request.description,
            conversation_id=conversation_id
        )
        
        # Get the created project
        project_data = timeline_service.get_timeline_project(result["project_id"])
        
        return TimelineProjectResponse(
            project=project_data["project"],
            message=result["message"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to create timeline from conversation: {str(e)}")


@router.post("/create", response_model=TimelineProjectResponse)
async def create_timeline_project(
    request: TimelineProjectRequest,
    http_request: Request
):
    """
    Create a new timeline project.
    
    Args:
        request: Timeline project creation request
        http_request: HTTP request object
        
    Returns:
        TimelineProjectResponse: Created timeline project
    """
    try:
        timeline_service = get_timeline_service(http_request)
        
        # Create timeline project
        result = timeline_service.create_timeline_project(
            project_name=request.project_name,
            description=request.description,
            conversation_id=request.conversation_id
        )
        
        # Get the created project
        project_data = timeline_service.get_timeline_project(result["project_id"])
        
        return TimelineProjectResponse(
            project=project_data["project"],
            message=result["message"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to create timeline project: {str(e)}")


@router.get("/list", response_model=BaseResponse)
async def list_timeline_projects(request: Request):
    """
    List all timeline projects.
    
    Args:
        request: HTTP request object
        
    Returns:
        BaseResponse: List of timeline projects
    """
    try:
        timeline_service = get_timeline_service(request)
        
        # Get projects from service
        projects = timeline_service.list_projects()
        
        return BaseResponse(
            message=f"Found {len(projects)} timeline projects",
            details={"projects": projects}
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to list timeline projects: {str(e)}")


@router.get("/{project_id}", response_model=TimelineProjectResponse)
async def get_timeline_project(project_id: str, http_request: Request):
    """
    Get timeline project information.
    
    Args:
        project_id: ID of the timeline project
        http_request: HTTP request object
        
    Returns:
        TimelineProjectResponse: Timeline project data
    """
    try:
        timeline_service = get_timeline_service(http_request)
        
        # Get project data
        project_data = timeline_service.get_timeline_project(project_id)
        
        return TimelineProjectResponse(
            project=project_data["project"],
            message="Timeline project retrieved successfully"
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to get timeline project: {str(e)}")


@router.delete("/{project_id}", response_model=BaseResponse)
async def delete_timeline_project(project_id: str, http_request: Request):
    """
    Delete a timeline project.
    
    Args:
        project_id: ID of the timeline project
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Deletion result
    """
    try:
        timeline_service = get_timeline_service(http_request)
        
        # Delete the project
        result = timeline_service.delete_timeline_project(project_id)
        
        return BaseResponse(
            message=result["message"],
            details={"project_id": project_id}
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to delete timeline project: {str(e)}")


@router.post("/{project_id}/tracks", response_model=TimelineTrackResponse)
async def add_track_to_project(
    project_id: str,
    request: TimelineTrackRequest,
    http_request: Request
):
    """
    Add a new track to a timeline project.
    
    Args:
        project_id: ID of the timeline project
        request: Track creation request
        http_request: HTTP request object
        
    Returns:
        TimelineTrackResponse: Created track
    """
    try:
        timeline_service = get_timeline_service(http_request)
        
        # Add track to project
        result = timeline_service.add_track_to_project(
            project_id=project_id,
            track_name=request.track_name,
            speaker_filename=request.speaker_filename
        )
        
        # Get updated project data
        project_data = timeline_service.get_timeline_project(project_id)
        
        # Find the created track
        track = None
        for t in project_data["project"]["tracks"]:
            if t["track_id"] == result["track_id"]:
                track = t
                break
        
        return TimelineTrackResponse(
            track=track,
            message=result["message"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to add track to project: {str(e)}")


@router.post("/{project_id}/tracks/{track_id}/segments", response_model=TimelineSegmentResponse)
async def add_segment_to_track(
    project_id: str,
    track_id: str,
    request: TimelineSegmentRequest,
    http_request: Request
):
    """
    Add a segment to a track in a timeline project.
    
    Args:
        project_id: ID of the timeline project
        track_id: ID of the track
        request: Segment creation request
        http_request: HTTP request object
        
    Returns:
        TimelineSegmentResponse: Created segment
    """
    try:
        print(f"DEBUG: add_segment_to_track called with project_id={project_id}, track_id={track_id}")
        print(f"DEBUG: Request data: text='{request.text}', start_time={request.start_time}, duration={request.duration}")
        
        timeline_service = get_timeline_service(http_request)
        
        # Add segment to track
        result = timeline_service.add_segment_to_track(
            project_id=project_id,
            track_id=track_id,
            text=request.text,
            start_time=request.start_time,
            duration=request.duration,
            emotion_control_method=request.emotion_control_method.value,
            emotion_reference_filename=request.emotion_reference_filename,
            emotion_weight=request.emotion_weight,
            emotion_vectors=request.emotion_vectors,
            emotion_text=request.emotion_text,
            use_random_sampling=request.use_random_sampling,
            max_text_tokens_per_segment=request.max_text_tokens_per_segment,
            do_sample=request.do_sample,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            length_penalty=request.length_penalty,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            max_mel_tokens=request.max_mel_tokens
        )
        
        print(f"DEBUG: Segment created with segment_id={result['segment_id']}")
        
        # Get updated project data
        project_data = timeline_service.get_timeline_project(project_id)
        
        # Find the created segment
        segment = None
        for track in project_data["project"]["tracks"]:
            if track["track_id"] == track_id:
                for s in track["segments"]:
                    if s["segment_id"] == result["segment_id"]:
                        segment = s
                        break
                break
        
        print(f"DEBUG: Found segment in project data: {segment is not None}")
        
        return TimelineSegmentResponse(
            segment=segment,
            message=result["message"]
        )
        
    except Exception as e:
        print(f"DEBUG: Error in add_segment_to_track: {str(e)}")
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to add segment to track: {str(e)}")


@router.put("/{project_id}/tracks/{track_id}/segments/{segment_id}", response_model=TimelineSegmentResponse)
async def update_segment_timing(
    project_id: str,
    track_id: str,
    segment_id: str,
    request: TimelineSegmentUpdateRequest,
    http_request: Request
):
    """
    Update the timing of a segment in a track.
    
    Args:
        project_id: ID of the timeline project
        track_id: ID of the track
        segment_id: ID of the segment
        request: Segment timing update request
        http_request: HTTP request object
        
    Returns:
        TimelineSegmentResponse: Updated segment
    """
    try:
        timeline_service = get_timeline_service(http_request)
        
        # Update segment timing
        result = timeline_service.update_segment_timing(
            project_id=project_id,
            track_id=track_id,
            segment_id=segment_id,
            start_time=request.start_time,
            duration=request.duration
        )
        
        # Get updated project data
        project_data = timeline_service.get_timeline_project(project_id)
        
        # Find the updated segment
        segment = None
        for track in project_data["project"]["tracks"]:
            if track["track_id"] == track_id:
                for s in track["segments"]:
                    if s["segment_id"] == segment_id:
                        segment = s
                        break
                break
        
        return TimelineSegmentResponse(
            segment=segment,
            message=result["message"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to update segment timing: {str(e)}")


@router.put("/{project_id}/tracks/{track_id}/segments/{segment_id}/properties", response_model=TimelineSegmentResponse)
async def update_segment_properties(
    project_id: str,
    track_id: str,
    segment_id: str,
    request: Dict[str, Any],
    http_request: Request
):
    """
    Update the properties of a segment in a track.
    
    Args:
        project_id: ID of the timeline project
        track_id: ID of the track
        segment_id: ID of the segment
        request: Segment properties update request
        http_request: HTTP request object
        
    Returns:
        TimelineSegmentResponse: Updated segment
    """
    try:
        print(f"DEBUG: update_segment_properties called with project_id={project_id}, track_id={track_id}, segment_id={segment_id}")
        print(f"DEBUG: Request properties: {request}")
        
        timeline_service = get_timeline_service(http_request)
        
        # Update segment properties
        result = timeline_service.update_segment_properties(
            project_id=project_id,
            track_id=track_id,
            segment_id=segment_id,
            properties=request
        )
        
        # Get updated project data
        project_data = timeline_service.get_timeline_project(project_id)
        
        # Find the updated segment
        segment = None
        for track in project_data["project"]["tracks"]:
            if track["track_id"] == track_id:
                for s in track["segments"]:
                    if s["segment_id"] == segment_id:
                        segment = s
                        break
                break
        
        return TimelineSegmentResponse(
            segment=segment,
            message=result["message"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to update segment properties: {str(e)}")


@router.post("/{project_id}/tracks/{track_id}/segments/{segment_id}/split", response_model=TimelineSegmentSplitResponse)
async def split_timeline_segment(
    project_id: str,
    track_id: str,
    segment_id: str,
    request: TimelineSegmentSplitRequest,
    http_request: Request
):
    """
    Split a segment into two timeline pieces.
    """
    try:
        timeline_service = get_timeline_service(http_request)
        result = timeline_service.split_segment(
            project_id=project_id,
            track_id=track_id,
            segment_id=segment_id,
            split_offset=request.split_offset,
            first_text=request.first_text,
            second_text=request.second_text,
        )

        return TimelineSegmentSplitResponse(
            updated_segment=result["updated_segment"],
            new_segment=result["new_segment"],
            message=result["message"],
        )
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to split segment: {str(e)}")


@router.delete("/{project_id}/tracks/{track_id}/segments/{segment_id}", response_model=BaseResponse)
async def delete_segment_from_track(
    project_id: str,
    track_id: str,
    segment_id: str,
    http_request: Request
):
    """
    Delete a segment from a track in a timeline project.
    
    Args:
        project_id: ID of the timeline project
        track_id: ID of the track
        segment_id: ID of the segment
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Deletion result
    """
    try:
        timeline_service = get_timeline_service(http_request)
        
        # Delete the segment
        result = timeline_service.delete_segment_from_track(
            project_id=project_id,
            track_id=track_id,
            segment_id=segment_id
        )
        
        return BaseResponse(
            message=result["message"],
            details={"project_id": project_id, "track_id": track_id, "segment_id": segment_id}
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to delete segment from track: {str(e)}")


@router.put("/{project_id}/tracks/{source_track_id}/segments/{segment_id}/move", response_model=TimelineSegmentResponse)
async def move_segment_to_track(
    project_id: str,
    source_track_id: str,
    segment_id: str,
    request: Dict[str, Any],
    http_request: Request
):
    """
    Move a segment from one track to another.
    
    Args:
        project_id: ID of the timeline project
        source_track_id: ID of the source track
        segment_id: ID of the segment
        request: Move request containing target_track_id and optional timing data
        http_request: HTTP request object
        
    Returns:
        TimelineSegmentResponse: Moved segment
    """
    try:
        timeline_service = get_timeline_service(http_request)
        
        target_track_id = request.get('target_track_id')
        new_start_time = request.get('new_start_time')
        
        if not target_track_id:
            raise ValidationError("target_track_id is required")
        
        # Move the segment
        result = timeline_service.move_segment_to_track(
            project_id=project_id,
            source_track_id=source_track_id,
            target_track_id=target_track_id,
            segment_id=segment_id,
            new_start_time=new_start_time
        )
        
        # Get updated project data
        project_data = timeline_service.get_timeline_project(project_id)
        
        # Find the moved segment
        segment = None
        for track in project_data["project"]["tracks"]:
            if track["track_id"] == target_track_id:
                for s in track["segments"]:
                    if s["segment_id"] == segment_id:
                        segment = s
                        break
                break
        
        return TimelineSegmentResponse(
            segment=segment,
            message=result["message"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to move segment to track: {str(e)}")


@router.put("/{project_id}/tracks/{track_id}/mute", response_model=BaseResponse)
async def toggle_track_mute(
    project_id: str,
    track_id: str,
    http_request: Request
):
    """
    Toggle mute state for a track.
    
    Args:
        project_id: ID of the timeline project
        track_id: ID of the track
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Toggle result
    """
    try:
        timeline_service = get_timeline_service(http_request)
        
        # Toggle track mute
        result = timeline_service.toggle_track_mute(
            project_id=project_id,
            track_id=track_id
        )
        
        return BaseResponse(
            message=result["message"],
            details={"project_id": project_id, "track_id": track_id, "muted": result.get("muted", False)}
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to toggle track mute: {str(e)}")


@router.put("/{project_id}/tracks/{track_id}/solo", response_model=BaseResponse)
async def toggle_track_solo(
    project_id: str,
    track_id: str,
    http_request: Request
):
    """
    Toggle solo state for a track.
    
    Args:
        project_id: ID of the timeline project
        track_id: ID of the track
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Toggle result
    """
    try:
        timeline_service = get_timeline_service(http_request)
        
        # Toggle track solo
        result = timeline_service.toggle_track_solo(
            project_id=project_id,
            track_id=track_id
        )
        
        return BaseResponse(
            message=result["message"],
            details={"project_id": project_id, "track_id": track_id, "solo": result.get("solo", False)}
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to toggle track solo: {str(e)}")


@router.put("/{project_id}/tracks/{track_id}/volume", response_model=BaseResponse)
async def update_track_volume(
    project_id: str,
    track_id: str,
    request: TimelineTrackVolumeRequest,
    http_request: Request
):
    """
    Update the playback level for a timeline track.

    Args:
        project_id: ID of the timeline project
        track_id: ID of the track
        request: Track volume payload
        http_request: HTTP request object

    Returns:
        BaseResponse: Updated track level details
    """
    try:
        timeline_service = get_timeline_service(http_request)

        result = timeline_service.update_track_volume(
            project_id=project_id,
            track_id=track_id,
            volume=request.volume
        )

        return BaseResponse(
            message=result["message"],
            details={
                "project_id": project_id,
                "track_id": track_id,
                "volume": result.get("volume", request.volume),
            }
        )

    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to update track volume: {str(e)}")


@router.post("/{project_id}/tracks/{track_id}/segments/{segment_id}/generate", response_model=TimelineSegmentResponse)
async def generate_segment_audio(
    project_id: str,
    track_id: str,
    segment_id: str,
    http_request: Request
):
    """
    Generate audio for a specific segment.
    
    Args:
        project_id: ID of the timeline project
        track_id: ID of the track
        segment_id: ID of the segment
        http_request: HTTP request object
        
    Returns:
        TimelineSegmentResponse: Updated segment with audio
    """
    try:
        timeline_service = get_timeline_service(http_request)
        
        # Generate audio for segment
        result = timeline_service.generate_segment_audio(
            project_id=project_id,
            track_id=track_id,
            segment_id=segment_id
        )
        
        # Get updated project data
        project_data = timeline_service.get_timeline_project(project_id)
        
        # Find the updated segment
        segment = None
        for track in project_data["project"]["tracks"]:
            if track["track_id"] == track_id:
                for s in track["segments"]:
                    if s["segment_id"] == segment_id:
                        segment = s
                        break
                break
        
        return TimelineSegmentResponse(
            segment=segment,
            message=result["message"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to generate segment audio: {str(e)}")


@router.get("/{project_id}/assets/audio/{filename}")
async def get_timeline_audio_file(project_id: str, filename: str, http_request: Request):
    """
    Serve audio files for timeline segments.
    
    Args:
        project_id: ID of the timeline project
        filename: Name of the audio file
        http_request: HTTP request object
        
    Returns:
        FileResponse: The audio file
    """
    try:
        timeline_service = get_timeline_service(http_request)
        resolved_path = timeline_service.resolve_project_audio_path(project_id, filename)
        if resolved_path and resolved_path.exists():
            return FileResponse(str(resolved_path), media_type="audio/wav")

        raise ValidationError(f"Audio file not found: {filename}")
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to get audio file: {str(e)}")


@router.get("/{project_id}/tracks/{track_id}/segments/{segment_id}/waveform", response_model=TimelineWaveformResponse)
async def get_timeline_segment_waveform(
    project_id: str,
    track_id: str,
    segment_id: str,
    http_request: Request,
    bars: int = 64,
):
    """
    Return a lightweight waveform preview for a generated timeline segment.
    """
    try:
        timeline_service = get_timeline_service(http_request)
        result = timeline_service.get_segment_waveform(
            project_id=project_id,
            track_id=track_id,
            segment_id=segment_id,
            bars=bars,
        )
        return TimelineWaveformResponse(**result, message="Waveform loaded successfully")
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to get segment waveform: {str(e)}")


@router.post("/{project_id}/export", response_model=TimelineExportResponse)
async def export_timeline(
    project_id: str,
    request: TimelineExportRequest,
    http_request: Request
):
    """
    Export timeline as concatenated audio.
    
    Args:
        project_id: ID of the timeline project
        request: Export request
        http_request: HTTP request object
        
    Returns:
        TimelineExportResponse: Export result
    """
    try:
        timeline_service = get_timeline_service(http_request)
        export_started_at = time.perf_counter()

        # Get project data
        project_data = timeline_service.get_timeline_project(project_id)
        project = project_data["project"]

        solo_tracks = [track for track in project["tracks"] if track.get("solo") and not track.get("muted")]
        active_tracks = solo_tracks if solo_tracks else [track for track in project["tracks"] if not track.get("muted")]

        placements = []
        for track in active_tracks:
            for segment in track["segments"]:
                audio_filename = segment.get("audio_filename")
                if not audio_filename:
                    continue

                audio_path = None
                for possible_path in [
                    str(OUTPUT_DIR / audio_filename),
                    str(OUTPUT_DIR / "timeline_assets" / project_id / audio_filename),
                    str(TEMP_CONVERSATION_SEGMENTS_DIR / audio_filename),
                ]:
                    if Path(possible_path).exists():
                        audio_path = possible_path
                        break

                if not audio_path:
                    continue

                placements.append(
                    {
                        "track_id": track["track_id"],
                        "track_name": track["track_name"],
                        "segment_id": segment["segment_id"],
                        "audio_path": audio_path,
                        "start_ms": int(round(float(segment.get("start_time", 0.0) or 0.0) * 1000)),
                        "volume": float(track.get("volume", 1.0) or 1.0),
                    }
                )

        if not placements:
            raise ValidationError("No audio files found in timeline project")

        output_filename = request.output_filename.strip()
        if not output_filename.lower().endswith(f".{request.format}"):
            output_filename = f"{output_filename}.{request.format}"

        # Generate output path
        output_path = str(OUTPUT_DIR / output_filename)

        mix_result = mix_audio_files_at_positions(
            placements,
            output_path,
            output_format=request.format,
            output_bitrate_kbps=request.output_bitrate_kbps,
            total_duration_ms=int(round(float(project.get("total_duration", 0.0) or 0.0) * 1000)),
            duck_overlaps=request.duck_overlaps,
            duck_amount_db=request.duck_amount_db,
            duck_fade_ms=request.duck_fade_ms,
            normalize_segments=request.normalize_segments,
            target_level_dbfs=request.target_level_dbfs,
            peak_limit_dbfs=request.peak_limit_dbfs,
            normalize_final_mix=request.normalize_final_mix,
            trim_leading_silence=request.trim_leading_silence,
            trim_trailing_silence=request.trim_trailing_silence,
            trim_silence_threshold_dbfs=request.trim_silence_threshold_dbfs,
            trim_min_silence_len_ms=request.trim_min_silence_len_ms,
            fade_in_ms=request.fade_in_ms,
            fade_out_ms=request.fade_out_ms,
        )

        # Get file info
        output_file = Path(output_path)
        file_size = output_file.stat().st_size if output_file.exists() else 0
        duration_seconds = float(mix_result.get("duration_ms", 0) or 0) / 1000.0
        
        return TimelineExportResponse(
            output_filename=output_filename,
            output_path=output_path,
            file_size=file_size,
            duration=duration_seconds,
            export_time_seconds=round(time.perf_counter() - export_started_at, 3),
            message="Timeline exported successfully",
            details={
                "output_format": mix_result.get("output_format", request.format),
                "output_bitrate_kbps": mix_result.get("finishing", {}).get("output_bitrate_kbps"),
                "normalization": mix_result.get("normalization", {}),
                "finishing": mix_result.get("finishing", {}),
                "ducking": {
                    "duck_overlaps": request.duck_overlaps,
                    "duck_amount_db": request.duck_amount_db,
                    "duck_fade_ms": request.duck_fade_ms,
                },
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to export timeline: {str(e)}")


@router.get("/{project_id}/export/download")
async def download_exported_timeline(project_id: str, http_request: Request):
    """
    Download exported timeline audio.
    
    Args:
        project_id: ID of the timeline project
        http_request: HTTP request object
        
    Returns:
        FileResponse: Exported audio file
    """
    try:
        # Try to find the exported file
        candidates = sorted(OUTPUT_DIR.glob(f"timeline_{project_id}_exported.*"))
        if not candidates:
            candidates = sorted(OUTPUT_DIR.glob(f"timeline_{project_id}.*"))

        for output_file in candidates:
            if output_file.is_file():
                return FileResponse(
                    path=str(output_file),
                    media_type=_guess_audio_media_type(str(output_file)),
                    filename=output_file.name
                )
        
        raise ValidationError(f"Exported timeline file not found for project: {project_id}")
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to download exported timeline: {str(e)}")
