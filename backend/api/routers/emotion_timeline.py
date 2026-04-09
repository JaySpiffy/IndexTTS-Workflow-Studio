"""
Emotion Timeline endpoints for IndexTTS2 API.
Handles emotion keyframe management and timeline-based emotion control for TTS segments.
"""

import uuid
import time
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import FileResponse

from ..models import (
    EmotionKeyframe, EmotionInterpolationType, EmotionTransitionSettings,
    BaseResponse, ErrorResponse
)
from ..exceptions import (
    IndexTTSException, ModelNotLoadedError, ValidationError,
    TimelineError
)
from ..config import settings
from ..services import EmotionService, TimelineService

router = APIRouter()


def get_emotion_service(request: Request) -> EmotionService:
    """Get emotion service from app state."""
    app = request.app
    if not getattr(app.state, 'model_loaded', False):
        raise ModelNotLoadedError()
    
    if not hasattr(app.state, 'emotion_service'):
        app.state.emotion_service = EmotionService()
    
    return app.state.emotion_service


def get_timeline_service(request: Request) -> TimelineService:
    """Get timeline service from app state."""
    app = request.app
    if not getattr(app.state, 'model_loaded', False):
        raise ModelNotLoadedError()
    
    if not hasattr(app.state, 'timeline_service'):
        app.state.timeline_service = TimelineService()
    
    return app.state.timeline_service


@router.post("/segments/{segment_id}/keyframes", response_model=BaseResponse)
async def add_emotion_keyframe(
    segment_id: str,
    request: Request,
    timestamp: float,
    emotion_vectors: List[float],
    interpolation_type: str = "linear",
    transition_duration: Optional[float] = None,
    project_id: Optional[str] = None,
    track_id: Optional[str] = None
):
    """
    Add an emotion keyframe to a segment.
    
    Args:
        segment_id: ID of the segment
        request: HTTP request object
        timestamp: Timestamp in seconds from segment start
        emotion_vectors: 8-dimensional emotion vector
        interpolation_type: Type of interpolation to use
        transition_duration: Duration of transition in seconds
        project_id: ID of the timeline project (required for timeline segments)
        track_id: ID of the track (required for timeline segments)
        
    Returns:
        BaseResponse: Result with keyframe information
    """
    try:
        emotion_service = get_emotion_service(request)
        
        # If project_id and track_id are provided, use timeline service
        if project_id and track_id:
            timeline_service = get_timeline_service(request)
            
            # Convert interpolation type string to enum
            try:
                interpolation_enum = EmotionInterpolationType(interpolation_type)
            except ValueError:
                raise ValidationError(f"Invalid interpolation type: {interpolation_type}")
            
            # Add keyframe using timeline service
            result = timeline_service.add_emotion_keyframe_to_segment(
                project_id=project_id,
                track_id=track_id,
                segment_id=segment_id,
                timestamp=timestamp,
                emotion_vectors=emotion_vectors,
                interpolation_type=interpolation_type,
                transition_duration=transition_duration
            )
            
            return BaseResponse(
                message="Emotion keyframe added successfully to timeline segment",
                details=result
            )
        else:
            # For standalone segments (not part of timeline)
            # This would require additional implementation for standalone segment management
            raise ValidationError("Project ID and Track ID are required for emotion keyframe management")
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to add emotion keyframe: {str(e)}")


@router.put("/keyframes/{keyframe_id}", response_model=BaseResponse)
async def update_emotion_keyframe(
    keyframe_id: str,
    request: Request,
    emotion_vectors: Optional[List[float]] = None,
    interpolation_type: Optional[str] = None,
    transition_duration: Optional[float] = None,
    timestamp: Optional[float] = None,
    project_id: Optional[str] = None,
    track_id: Optional[str] = None,
    segment_id: Optional[str] = None
):
    """
    Update an emotion keyframe.
    
    Args:
        keyframe_id: ID of the keyframe to update
        request: HTTP request object
        emotion_vectors: New emotion vectors (optional)
        interpolation_type: New interpolation type (optional)
        transition_duration: New transition duration (optional)
        timestamp: New timestamp (optional)
        project_id: ID of the timeline project (required for timeline segments)
        track_id: ID of the track (required for timeline segments)
        segment_id: ID of the segment (required for timeline segments)
        
    Returns:
        BaseResponse: Result with updated keyframe information
    """
    try:
        if not project_id or not track_id or not segment_id:
            raise ValidationError("Project ID, Track ID, and Segment ID are required for emotion keyframe updates")
        
        timeline_service = get_timeline_service(request)
        
        # Convert interpolation type string to enum if provided
        interpolation_enum = None
        if interpolation_type is not None:
            try:
                interpolation_enum = EmotionInterpolationType(interpolation_type)
            except ValueError:
                raise ValidationError(f"Invalid interpolation type: {interpolation_type}")
        
        # Update keyframe using timeline service
        result = timeline_service.update_emotion_keyframe_in_segment(
            project_id=project_id,
            track_id=track_id,
            segment_id=segment_id,
            keyframe_id=keyframe_id,
            emotion_vectors=emotion_vectors,
            interpolation_type=interpolation_type,
            transition_duration=transition_duration,
            timestamp=timestamp
        )
        
        return BaseResponse(
            message="Emotion keyframe updated successfully",
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to update emotion keyframe: {str(e)}")


@router.delete("/keyframes/{keyframe_id}", response_model=BaseResponse)
async def delete_emotion_keyframe(
    keyframe_id: str,
    request: Request,
    project_id: Optional[str] = None,
    track_id: Optional[str] = None,
    segment_id: Optional[str] = None
):
    """
    Delete an emotion keyframe.
    
    Args:
        keyframe_id: ID of the keyframe to delete
        request: HTTP request object
        project_id: ID of the timeline project (required for timeline segments)
        track_id: ID of the track (required for timeline segments)
        segment_id: ID of the segment (required for timeline segments)
        
    Returns:
        BaseResponse: Result with deleted keyframe information
    """
    try:
        if not project_id or not track_id or not segment_id:
            raise ValidationError("Project ID, Track ID, and Segment ID are required for emotion keyframe deletion")
        
        timeline_service = get_timeline_service(request)
        
        # Remove keyframe using timeline service
        result = timeline_service.remove_emotion_keyframe_from_segment(
            project_id=project_id,
            track_id=track_id,
            segment_id=segment_id,
            keyframe_id=keyframe_id
        )
        
        return BaseResponse(
            message="Emotion keyframe deleted successfully",
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to delete emotion keyframe: {str(e)}")


@router.get("/segments/{segment_id}/keyframes", response_model=BaseResponse)
async def get_segment_keyframes(
    segment_id: str,
    request: Request,
    project_id: Optional[str] = None,
    track_id: Optional[str] = None
):
    """
    Get all emotion keyframes for a segment.
    
    Args:
        segment_id: ID of the segment
        request: HTTP request object
        project_id: ID of the timeline project (required for timeline segments)
        track_id: ID of the track (required for timeline segments)
        
    Returns:
        BaseResponse: List of emotion keyframes
    """
    try:
        if not project_id or not track_id:
            raise ValidationError("Project ID and Track ID are required to retrieve segment keyframes")
        
        timeline_service = get_timeline_service(request)
        
        # Get timeline project
        project_data = timeline_service.get_timeline_project(project_id)
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
        
        # Return keyframes
        keyframes = segment.get("emotion_keyframes", [])
        
        return BaseResponse(
            message=f"Found {len(keyframes)} emotion keyframes",
            details={
                "segment_id": segment_id,
                "keyframes": keyframes,
                "emotion_timeline_enabled": segment.get("emotion_timeline_enabled", False)
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to get segment keyframes: {str(e)}")


@router.post("/segments/{segment_id}/preview", response_model=BaseResponse)
async def generate_emotion_preview(
    segment_id: str,
    request: Request,
    preview_duration: float = 2.0,
    project_id: Optional[str] = None,
    track_id: Optional[str] = None,
    keyframe_id: Optional[str] = None
):
    """
    Generate emotion preview for a segment or around a specific keyframe.
    
    Args:
        segment_id: ID of the segment
        request: HTTP request object
        preview_duration: Duration of preview in seconds (before and after keyframe)
        project_id: ID of the timeline project (required for timeline segments)
        track_id: ID of the track (required for timeline segments)
        keyframe_id: ID of the keyframe to preview (optional)
        
    Returns:
        BaseResponse: Emotion preview data
    """
    try:
        if not project_id or not track_id:
            raise ValidationError("Project ID and Track ID are required for emotion preview generation")
        
        timeline_service = get_timeline_service(request)
        
        if keyframe_id:
            # Generate preview around specific keyframe
            result = timeline_service.preview_emotion_keyframe_change(
                project_id=project_id,
                track_id=track_id,
                segment_id=segment_id,
                keyframe_id=keyframe_id,
                preview_duration=preview_duration
            )
        else:
            # Generate preview for entire segment
            result = timeline_service.get_emotion_timeline_for_segment(
                project_id=project_id,
                track_id=track_id,
                segment_id=segment_id,
                sample_rate=10  # 10 samples per second
            )
        
        return BaseResponse(
            message="Emotion preview generated successfully",
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to generate emotion preview: {str(e)}")


@router.get("/segments/{segment_id}/timeline", response_model=BaseResponse)
async def get_emotion_timeline(
    segment_id: str,
    request: Request,
    sample_rate: int = 10,
    project_id: Optional[str] = None,
    track_id: Optional[str] = None
):
    """
    Get emotion timeline data for a segment.
    
    Args:
        segment_id: ID of the segment
        request: HTTP request object
        sample_rate: Number of samples per second
        project_id: ID of the timeline project (required for timeline segments)
        track_id: ID of the track (required for timeline segments)
        
    Returns:
        BaseResponse: Emotion timeline data
    """
    try:
        if not project_id or not track_id:
            raise ValidationError("Project ID and Track ID are required to retrieve emotion timeline")
        
        timeline_service = get_timeline_service(request)
        
        # Get emotion timeline using timeline service
        result = timeline_service.get_emotion_timeline_for_segment(
            project_id=project_id,
            track_id=track_id,
            segment_id=segment_id,
            sample_rate=sample_rate
        )
        
        return BaseResponse(
            message="Emotion timeline retrieved successfully",
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to get emotion timeline: {str(e)}")


@router.put("/segments/{segment_id}/settings", response_model=BaseResponse)
async def update_segment_emotion_settings(
    segment_id: str,
    request: Request,
    settings_data: Dict[str, Any],
    project_id: Optional[str] = None,
    track_id: Optional[str] = None
):
    """
    Update emotion settings for a segment.
    
    Args:
        segment_id: ID of the segment
        request: HTTP request object
        settings_data: Settings to update
        project_id: ID of the timeline project (required for timeline segments)
        track_id: ID of the track (required for timeline segments)
        
    Returns:
        BaseResponse: Updated segment settings
    """
    try:
        if not project_id or not track_id:
            raise ValidationError("Project ID and Track ID are required to update segment emotion settings")
        
        timeline_service = get_timeline_service(request)
        
        # Validate settings
        valid_settings = [
            "emotion_timeline_enabled",
            "emotion_interpolation_type",
            "emotion_transition_duration",
            "emotion_control_method",
            "emotion_reference_filename",
            "emotion_weight",
            "emotion_vectors",
            "emotion_text",
            "use_random_sampling"
        ]
        
        # Filter valid settings
        filtered_settings = {}
        for key, value in settings_data.items():
            if key in valid_settings:
                filtered_settings[key] = value
        
        if not filtered_settings:
            raise ValidationError("No valid emotion settings provided")
        
        # Update segment properties using timeline service
        result = timeline_service.update_segment_properties(
            project_id=project_id,
            track_id=track_id,
            segment_id=segment_id,
            properties=filtered_settings
        )
        
        return BaseResponse(
            message="Segment emotion settings updated successfully",
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to update segment emotion settings: {str(e)}")


@router.get("/segments/{segment_id}/emotion-at-time", response_model=BaseResponse)
async def get_emotion_at_timestamp(
    segment_id: str,
    request: Request,
    timestamp: float,
    project_id: Optional[str] = None,
    track_id: Optional[str] = None
):
    """
    Calculate emotion vector at a specific timestamp within a segment.
    
    Args:
        segment_id: ID of the segment
        request: HTTP request object
        timestamp: Timestamp in seconds from segment start
        project_id: ID of the timeline project (required for timeline segments)
        track_id: ID of the track (required for timeline segments)
        
    Returns:
        BaseResponse: Emotion vector at the specified timestamp
    """
    try:
        if not project_id or not track_id:
            raise ValidationError("Project ID and Track ID are required to calculate emotion at timestamp")
        
        timeline_service = get_timeline_service(request)
        
        # Calculate emotion at timestamp using timeline service
        result = timeline_service.calculate_emotion_at_timestamp(
            project_id=project_id,
            track_id=track_id,
            segment_id=segment_id,
            timestamp=timestamp
        )
        
        return BaseResponse(
            message="Emotion at timestamp calculated successfully",
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to calculate emotion at timestamp: {str(e)}")


@router.post("/segments/calculate-transition", response_model=BaseResponse)
async def calculate_segment_transition(
    request: Request,
    from_project_id: str,
    from_track_id: str,
    from_segment_id: str,
    to_project_id: str,
    to_track_id: str,
    to_segment_id: str,
    transition_duration: float = 1.0
):
    """
    Calculate emotion transition between two segments.
    
    Args:
        request: HTTP request object
        from_project_id: ID of the source timeline project
        from_track_id: ID of the source track
        from_segment_id: ID of the source segment
        to_project_id: ID of the target timeline project
        to_track_id: ID of the target track
        to_segment_id: ID of the target segment
        transition_duration: Duration of transition in seconds
        
    Returns:
        BaseResponse: Transition timeline data
    """
    try:
        timeline_service = get_timeline_service(request)
        
        # Calculate transition using timeline service
        result = timeline_service.calculate_segment_transition(
            project_id=from_project_id,
            from_track_id=from_track_id,
            from_segment_id=from_segment_id,
            to_track_id=to_track_id,
            to_segment_id=to_segment_id,
            transition_duration=transition_duration
        )
        
        return BaseResponse(
            message="Segment transition calculated successfully",
            details=result
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TimelineError(f"Failed to calculate segment transition: {str(e)}")