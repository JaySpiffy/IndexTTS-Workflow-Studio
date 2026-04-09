"""
Emotion Service for IndexTTS2 API.
Handles emotion vector interpolation, timeline manipulation, and emotion state calculations.
"""

import math
import uuid
from typing import Dict, Any, List, Optional, Tuple

from ..models import (
    EmotionKeyframe, EmotionInterpolationType, EmotionTransitionSettings,
    TimelineSegment
)
from ..exceptions import ValidationError


class EmotionService:
    """Service for handling emotion vector operations and timeline manipulation."""
    
    def __init__(self):
        """Initialize emotion service."""
        self.default_settings = EmotionTransitionSettings()
    
    def create_emotion_keyframe(
        self,
        timestamp: float,
        emotion_vectors: List[float],
        interpolation_type: EmotionInterpolationType = EmotionInterpolationType.LINEAR,
        transition_duration: float = 0.5
    ) -> EmotionKeyframe:
        """
        Create a new emotion keyframe.
        
        Args:
            timestamp: Timestamp in seconds from segment start
            emotion_vectors: 8-dimensional emotion vector
            interpolation_type: Type of interpolation to use
            transition_duration: Duration of transition in seconds
        
        Returns:
            EmotionKeyframe: Created keyframe
        """
        if timestamp < 0:
            raise ValidationError("Timestamp must be non-negative")
        
        if len(emotion_vectors) > 8:
            raise ValidationError("Maximum 8 emotion vector components allowed")
        
        if sum(abs(x) for x in emotion_vectors) > 1.5:
            raise ValidationError("Sum of emotion vector components cannot exceed 1.5")
        
        if transition_duration < 0:
            raise ValidationError("Transition duration must be non-negative")
        
        return EmotionKeyframe(
            keyframe_id=str(uuid.uuid4()),
            timestamp=timestamp,
            emotion_vectors=emotion_vectors,
            interpolation_type=interpolation_type,
            transition_duration=transition_duration
        )
    
    def add_keyframe_to_segment(
        self,
        segment: TimelineSegment,
        timestamp: float,
        emotion_vectors: List[float],
        interpolation_type: EmotionInterpolationType = EmotionInterpolationType.LINEAR,
        transition_duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Add an emotion keyframe to a timeline segment.
        
        Args:
            segment: Timeline segment to add keyframe to
            timestamp: Timestamp in seconds from segment start
            emotion_vectors: 8-dimensional emotion vector
            interpolation_type: Type of interpolation to use
            transition_duration: Duration of transition in seconds
        
        Returns:
            Dict: Result with keyframe information
        """
        if timestamp > segment.duration:
            raise ValidationError(f"Timestamp {timestamp} exceeds segment duration {segment.duration}")
        
        # Use segment's default transition duration if not provided
        if transition_duration is None:
            transition_duration = segment.emotion_transition_duration
        
        # Create new keyframe
        keyframe = self.create_emotion_keyframe(
            timestamp=timestamp,
            emotion_vectors=emotion_vectors,
            interpolation_type=interpolation_type,
            transition_duration=transition_duration
        )
        
        # Add to segment
        segment.emotion_keyframes.append(keyframe)
        
        # Sort keyframes by timestamp
        segment.emotion_keyframes.sort(key=lambda x: x.timestamp)
        
        # Enable emotion timeline if not already enabled
        segment.emotion_timeline_enabled = True
        
        return {
            "keyframe_id": keyframe.keyframe_id,
            "segment_id": segment.segment_id,
            "timestamp": timestamp,
            "emotion_vectors": emotion_vectors,
            "message": "Emotion keyframe added successfully"
        }
    
    def update_keyframe_in_segment(
        self,
        segment: TimelineSegment,
        keyframe_id: str,
        emotion_vectors: Optional[List[float]] = None,
        interpolation_type: Optional[EmotionInterpolationType] = None,
        transition_duration: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update an existing emotion keyframe in a timeline segment.
        
        Args:
            segment: Timeline segment containing the keyframe
            keyframe_id: ID of the keyframe to update
            emotion_vectors: New emotion vectors (optional)
            interpolation_type: New interpolation type (optional)
            transition_duration: New transition duration (optional)
            timestamp: New timestamp (optional)
        
        Returns:
            Dict: Result with updated keyframe information
        """
        # Find the keyframe
        keyframe = None
        for kf in segment.emotion_keyframes:
            if kf.keyframe_id == keyframe_id:
                keyframe = kf
                break
        
        if not keyframe:
            raise ValidationError(f"Keyframe not found: {keyframe_id}")
        
        # Update keyframe properties
        updated_fields = {}
        
        if emotion_vectors is not None:
            if len(emotion_vectors) > 8:
                raise ValidationError("Maximum 8 emotion vector components allowed")
            if sum(abs(x) for x in emotion_vectors) > 1.5:
                raise ValidationError("Sum of emotion vector components cannot exceed 1.5")
            keyframe.emotion_vectors = emotion_vectors
            updated_fields["emotion_vectors"] = emotion_vectors
        
        if interpolation_type is not None:
            keyframe.interpolation_type = interpolation_type
            updated_fields["interpolation_type"] = interpolation_type
        
        if transition_duration is not None:
            if transition_duration < 0:
                raise ValidationError("Transition duration must be non-negative")
            keyframe.transition_duration = transition_duration
            updated_fields["transition_duration"] = transition_duration
        
        if timestamp is not None:
            if timestamp < 0:
                raise ValidationError("Timestamp must be non-negative")
            if timestamp > segment.duration:
                raise ValidationError(f"Timestamp {timestamp} exceeds segment duration {segment.duration}")
            keyframe.timestamp = timestamp
            updated_fields["timestamp"] = timestamp
        
        # Re-sort keyframes if timestamp was updated
        if timestamp is not None:
            segment.emotion_keyframes.sort(key=lambda x: x.timestamp)
        
        return {
            "keyframe_id": keyframe_id,
            "segment_id": segment.segment_id,
            "updated_fields": updated_fields,
            "message": "Emotion keyframe updated successfully"
        }
    
    def remove_keyframe_from_segment(
        self,
        segment: TimelineSegment,
        keyframe_id: str
    ) -> Dict[str, Any]:
        """
        Remove an emotion keyframe from a timeline segment.
        
        Args:
            segment: Timeline segment containing the keyframe
            keyframe_id: ID of the keyframe to remove
        
        Returns:
            Dict: Result with removed keyframe information
        """
        # Find and remove the keyframe
        keyframe_index = None
        for i, kf in enumerate(segment.emotion_keyframes):
            if kf.keyframe_id == keyframe_id:
                keyframe_index = i
                break
        
        if keyframe_index is None:
            raise ValidationError(f"Keyframe not found: {keyframe_id}")
        
        removed_keyframe = segment.emotion_keyframes.pop(keyframe_index)
        
        # Disable emotion timeline if no keyframes left
        if not segment.emotion_keyframes:
            segment.emotion_timeline_enabled = False
        
        return {
            "keyframe_id": keyframe_id,
            "segment_id": segment.segment_id,
            "removed_keyframe": removed_keyframe.dict(),
            "message": "Emotion keyframe removed successfully"
        }
    
    def calculate_emotion_at_timestamp(
        self,
        segment: TimelineSegment,
        timestamp: float
    ) -> List[float]:
        """
        Calculate the emotion vector at a specific timestamp within a segment.
        
        Args:
            segment: Timeline segment with emotion keyframes
            timestamp: Timestamp in seconds from segment start
        
        Returns:
            List[float]: Interpolated 8-dimensional emotion vector
        """
        if timestamp < 0 or timestamp > segment.duration:
            raise ValidationError(f"Timestamp {timestamp} is outside segment range [0, {segment.duration}]")
        
        # If emotion timeline is disabled or no keyframes, return base emotion vectors
        if not segment.emotion_timeline_enabled or not segment.emotion_keyframes:
            return segment.emotion_vectors or [0.0] * 8
        
        # Sort keyframes by timestamp
        keyframes = sorted(segment.emotion_keyframes, key=lambda x: x.timestamp)
        
        # If timestamp is before the first keyframe, use the first keyframe's emotion
        if timestamp <= keyframes[0].timestamp:
            return keyframes[0].emotion_vectors
        
        # If timestamp is after the last keyframe, use the last keyframe's emotion
        if timestamp >= keyframes[-1].timestamp:
            return keyframes[-1].emotion_vectors
        
        # Find the keyframes that bracket the timestamp
        prev_keyframe = None
        next_keyframe = None
        
        for i in range(len(keyframes) - 1):
            if keyframes[i].timestamp <= timestamp <= keyframes[i + 1].timestamp:
                prev_keyframe = keyframes[i]
                next_keyframe = keyframes[i + 1]
                break
        
        if not prev_keyframe or not next_keyframe:
            # Fallback to nearest keyframe
            return min(keyframes, key=lambda x: abs(x.timestamp - timestamp)).emotion_vectors
        
        # Calculate interpolation based on the previous keyframe's interpolation type
        return self._interpolate_emotion_vectors(
            prev_keyframe.emotion_vectors,
            next_keyframe.emotion_vectors,
            timestamp,
            prev_keyframe.timestamp,
            next_keyframe.timestamp,
            prev_keyframe.interpolation_type,
            prev_keyframe.transition_duration
        )
    
    def _interpolate_emotion_vectors(
        self,
        start_vectors: List[float],
        end_vectors: List[float],
        current_time: float,
        start_time: float,
        end_time: float,
        interpolation_type: EmotionInterpolationType,
        transition_duration: float
    ) -> List[float]:
        """
        Interpolate between two emotion vectors.
        
        Args:
            start_vectors: Starting emotion vectors
            end_vectors: Ending emotion vectors
            current_time: Current timestamp
            start_time: Start timestamp
            end_time: End timestamp
            interpolation_type: Type of interpolation
            transition_duration: Duration of transition
        
        Returns:
            List[float]: Interpolated emotion vectors
        """
        # Calculate the effective transition period
        transition_start = start_time
        transition_end = min(start_time + transition_duration, end_time)
        
        # If we're outside the transition period, return the appropriate vector
        if current_time <= transition_start:
            return start_vectors
        elif current_time >= transition_end:
            return end_vectors
        
        # Calculate interpolation progress (0.0 to 1.0)
        if transition_end > transition_start:
            progress = (current_time - transition_start) / (transition_end - transition_start)
        else:
            progress = 1.0
        
        # Apply interpolation curve
        adjusted_progress = self._apply_interpolation_curve(progress, interpolation_type)
        
        # Interpolate each dimension
        result = []
        for i in range(max(len(start_vectors), len(end_vectors))):
            start_val = start_vectors[i] if i < len(start_vectors) else 0.0
            end_val = end_vectors[i] if i < len(end_vectors) else 0.0
            
            # Linear interpolation
            interpolated_val = start_val + (end_val - start_val) * adjusted_progress
            result.append(interpolated_val)
        
        # Ensure we have exactly 8 dimensions
        while len(result) < 8:
            result.append(0.0)
        
        return result[:8]
    
    def _apply_interpolation_curve(
        self,
        progress: float,
        interpolation_type: EmotionInterpolationType
    ) -> float:
        """
        Apply interpolation curve to progress value.
        
        Args:
            progress: Linear progress value (0.0 to 1.0)
            interpolation_type: Type of interpolation curve
        
        Returns:
            float: Adjusted progress value
        """
        if interpolation_type == EmotionInterpolationType.LINEAR:
            return progress
        elif interpolation_type == EmotionInterpolationType.EASE_IN:
            return progress * progress
        elif interpolation_type == EmotionInterpolationType.EASE_OUT:
            return 1.0 - (1.0 - progress) * (1.0 - progress)
        elif interpolation_type == EmotionInterpolationType.EASE_IN_OUT:
            if progress < 0.5:
                return 2.0 * progress * progress
            else:
                return 1.0 - 2.0 * (1.0 - progress) * (1.0 - progress)
        else:
            return progress
    
    def generate_emotion_timeline(
        self,
        segment: TimelineSegment,
        sample_rate: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate emotion timeline data for a segment.
        
        Args:
            segment: Timeline segment with emotion keyframes
            sample_rate: Number of samples per second
        
        Returns:
            List[Dict]: Timeline data with timestamps and emotion vectors
        """
        if not segment.emotion_timeline_enabled or not segment.emotion_keyframes:
            # Return base emotion for the entire segment
            return [{
                "timestamp": 0.0,
                "emotion_vectors": segment.emotion_vectors or [0.0] * 8
            }]
        
        timeline = []
        num_samples = max(1, int(segment.duration * sample_rate))
        
        for i in range(num_samples):
            timestamp = (i / (num_samples - 1)) * segment.duration if num_samples > 1 else 0.0
            emotion_vectors = self.calculate_emotion_at_timestamp(segment, timestamp)
            
            timeline.append({
                "timestamp": timestamp,
                "emotion_vectors": emotion_vectors
            })
        
        return timeline
    
    def calculate_segment_transition(
        self,
        from_segment: TimelineSegment,
        to_segment: TimelineSegment,
        transition_duration: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Calculate emotion transition between two segments.
        
        Args:
            from_segment: Source segment
            to_segment: Target segment
            transition_duration: Duration of transition in seconds
        
        Returns:
            List[Dict]: Transition timeline data
        """
        # Get emotion vectors at the end of the from_segment and start of to_segment
        from_emotion = self.calculate_emotion_at_timestamp(from_segment, from_segment.duration - 0.01)
        to_emotion = self.calculate_emotion_at_timestamp(to_segment, 0.01)
        
        # Generate transition timeline
        transition_timeline = []
        num_samples = max(2, int(transition_duration * 10))  # 10 samples per second
        
        for i in range(num_samples):
            progress = i / (num_samples - 1) if num_samples > 1 else 0.0
            timestamp = progress * transition_duration
            
            # Apply ease-in-out curve for segment transitions
            adjusted_progress = self._apply_interpolation_curve(progress, EmotionInterpolationType.EASE_IN_OUT)
            
            # Interpolate emotion vectors
            emotion_vectors = []
            for j in range(8):
                from_val = from_emotion[j] if j < len(from_emotion) else 0.0
                to_val = to_emotion[j] if j < len(to_emotion) else 0.0
                interpolated_val = from_val + (to_val - from_val) * adjusted_progress
                emotion_vectors.append(interpolated_val)
            
            transition_timeline.append({
                "timestamp": timestamp,
                "emotion_vectors": emotion_vectors
            })
        
        return transition_timeline
    
    def preview_emotion_change(
        self,
        segment: TimelineSegment,
        keyframe_id: str,
        preview_duration: float = 2.0
    ) -> Dict[str, Any]:
        """
        Generate a preview of emotion change around a keyframe.
        
        Args:
            segment: Timeline segment containing the keyframe
            keyframe_id: ID of the keyframe to preview
            preview_duration: Duration of preview in seconds (before and after)
        
        Returns:
            Dict: Preview data with emotion timeline
        """
        # Find the keyframe
        keyframe = None
        for kf in segment.emotion_keyframes:
            if kf.keyframe_id == keyframe_id:
                keyframe = kf
                break
        
        if not keyframe:
            raise ValidationError(f"Keyframe not found: {keyframe_id}")
        
        # Calculate preview range
        start_time = max(0, keyframe.timestamp - preview_duration)
        end_time = min(segment.duration, keyframe.timestamp + preview_duration)
        
        # Generate preview timeline
        preview_timeline = []
        num_samples = max(2, int((end_time - start_time) * 10))  # 10 samples per second
        
        for i in range(num_samples):
            timestamp = start_time + (i / (num_samples - 1)) * (end_time - start_time) if num_samples > 1 else start_time
            emotion_vectors = self.calculate_emotion_at_timestamp(segment, timestamp)
            
            preview_timeline.append({
                "timestamp": timestamp,
                "emotion_vectors": emotion_vectors
            })
        
        return {
            "keyframe_id": keyframe_id,
            "segment_id": segment.segment_id,
            "keyframe_timestamp": keyframe.timestamp,
            "preview_start_time": start_time,
            "preview_end_time": end_time,
            "preview_timeline": preview_timeline,
            "keyframe_emotion": keyframe.emotion_vectors
        }