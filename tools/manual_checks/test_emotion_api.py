#!/usr/bin/env python3
"""
Test script for emotion timeline API integration.
Tests the backend API endpoints to ensure they're working correctly.
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running and healthy."""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API is running and healthy")
            return True
        else:
            print(f"❌ API health check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check failed with error: {str(e)}")
        return False

def test_create_timeline_project():
    """Test creating a timeline project."""
    try:
        project_data = {
            "project_name": "Emotion Timeline Test",
            "description": "Test project for emotion timeline API integration"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/timeline/create",
            json=project_data
        )
        
        if response.status_code == 200:
            project = response.json()
            project_id = project["project"]["project_id"]
            print(f"✅ Timeline project created successfully: {project_id}")
            return project_id
        else:
            print(f"❌ Failed to create timeline project: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Failed to create timeline project with error: {str(e)}")
        return None

def test_add_track_to_project(project_id):
    """Test adding a track to the timeline project."""
    try:
        track_data = {
            "track_name": "Test Track",
            "speaker_filename": "narrator.wav"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/timeline/{project_id}/tracks",
            json=track_data
        )
        
        if response.status_code == 200:
            track = response.json()
            track_id = track["track"]["track_id"]
            print(f"✅ Track added successfully: {track_id}")
            return track_id
        else:
            print(f"❌ Failed to add track: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Failed to add track with error: {str(e)}")
        return None

def test_add_segment_to_track(project_id, track_id):
    """Test adding a segment to the track."""
    try:
        segment_data = {
            "track_id": track_id,
            "text": "This is a test segment for emotion timeline integration.",
            "start_time": 0.0,
            "duration": 5.0,
            "emotion_control_method": "from_speaker"
        }
        
        response = requests.post(
            f"{BASE_URL}/api/timeline/{project_id}/tracks/{track_id}/segments",
            json=segment_data
        )
        
        if response.status_code == 200:
            segment = response.json()
            segment_id = segment["segment"]["segment_id"]
            print(f"✅ Segment added successfully: {segment_id}")
            return segment_id
        else:
            print(f"❌ Failed to add segment: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Failed to add segment with error: {str(e)}")
        return None

def test_add_emotion_keyframe(project_id, track_id, segment_id):
    """Test adding an emotion keyframe to the segment."""
    try:
        params = {
            "timestamp": 2.0,
            "emotion_vectors": [0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 0.8, 0.1],
            "interpolation_type": "linear",
            "transition_duration": 1.0,
            "project_id": project_id,
            "track_id": track_id
        }
        
        # Add empty list to satisfy FastAPI
        response = requests.post(
            f"{BASE_URL}/api/emotion-timeline/segments/{segment_id}/keyframes",
            json=[],
            params=params
        )
        
        if response.status_code == 200:
            result = response.json()
            keyframe_id = result["details"]["keyframe_id"]
            print(f"✅ Emotion keyframe added successfully: {keyframe_id}")
            return keyframe_id
        else:
            print(f"❌ Failed to add emotion keyframe: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Failed to add emotion keyframe with error: {str(e)}")
        return None

def test_get_emotion_timeline(project_id, track_id, segment_id):
    """Test getting emotion timeline data for a segment."""
    try:
        params = {
            "project_id": project_id,
            "track_id": track_id
        }
        
        response = requests.get(
            f"{BASE_URL}/api/emotion-timeline/segments/{segment_id}/timeline",
            params=params
        )
        
        if response.status_code == 200:
            result = response.json()
            timeline = result["details"]
            print(f"✅ Emotion timeline retrieved successfully")
            print(f"   - Timeline length: {len(timeline.get('timeline', []))}")
            print(f"   - Sample rate: {timeline.get('sample_rate', 0)} samples/second")
            return timeline
        else:
            print(f"❌ Failed to get emotion timeline: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Failed to get emotion timeline with error: {str(e)}")
        return None

def test_update_emotion_keyframe(project_id, track_id, segment_id, keyframe_id):
    """Test updating an emotion keyframe."""
    try:
        params = {
            "emotion_vectors": [0.6, 0.4, 0.8, 0.3, 0.7, 0.5, 0.9, 0.2],
            "interpolation_type": "linear",  # Changed from "smooth" to "linear"
            "timestamp": 2.5,
            "project_id": project_id,
            "track_id": track_id,
            "segment_id": segment_id
        }
        
        # Add empty list to satisfy FastAPI
        response = requests.put(
            f"{BASE_URL}/api/emotion-timeline/keyframes/{keyframe_id}",
            json=[],
            params=params
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Emotion keyframe updated successfully")
            return result
        else:
            print(f"❌ Failed to update emotion keyframe: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Failed to update emotion keyframe with error: {str(e)}")
        return None

def test_calculate_emotion_at_timestamp(project_id, track_id, segment_id):
    """Test calculating emotion at a specific timestamp."""
    try:
        params = {
            "timestamp": 2.5,
            "project_id": project_id,
            "track_id": track_id
        }
        
        response = requests.get(
            f"{BASE_URL}/api/emotion-timeline/segments/{segment_id}/emotion-at-time",
            params=params
        )
        
        if response.status_code == 200:
            result = response.json()
            emotion = result["details"]
            print(f"✅ Emotion at timestamp calculated successfully")
            print(f"   - Timestamp: {emotion.get('timestamp', 0)}s")
            print(f"   - Emotion vectors: {emotion.get('emotion_vectors', [])}")
            return emotion
        else:
            print(f"❌ Failed to calculate emotion at timestamp: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Failed to calculate emotion at timestamp with error: {str(e)}")
        return None

def test_delete_emotion_keyframe(project_id, track_id, segment_id, keyframe_id):
    """Test deleting an emotion keyframe."""
    try:
        params = {
            "project_id": project_id,
            "track_id": track_id,
            "segment_id": segment_id
        }
        
        # Add empty list to satisfy FastAPI
        response = requests.delete(
            f"{BASE_URL}/api/emotion-timeline/keyframes/{keyframe_id}",
            json=[],
            params=params
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Emotion keyframe deleted successfully")
            return result
        else:
            print(f"❌ Failed to delete emotion keyframe: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Failed to delete emotion keyframe with error: {str(e)}")
        return None

def main():
    """Run all API tests."""
    print("🚀 Testing Emotion Timeline API Integration\n")
    
    # Test API health
    if not test_api_health():
        return
    
    # Test creating a timeline project
    project_id = test_create_timeline_project()
    if not project_id:
        return
    
    # Test adding a track to the project
    track_id = test_add_track_to_project(project_id)
    if not track_id:
        return
    
    # Test adding a segment to the track
    segment_id = test_add_segment_to_track(project_id, track_id)
    if not segment_id:
        return
    
    # Test adding an emotion keyframe
    keyframe_id = test_add_emotion_keyframe(project_id, track_id, segment_id)
    if not keyframe_id:
        return
    
    # Test getting emotion timeline
    timeline = test_get_emotion_timeline(project_id, track_id, segment_id)
    
    # Test updating emotion keyframe
    test_update_emotion_keyframe(project_id, track_id, segment_id, keyframe_id)
    
    # Test calculating emotion at timestamp
    test_calculate_emotion_at_timestamp(project_id, track_id, segment_id)
    
    # Test deleting emotion keyframe
    test_delete_emotion_keyframe(project_id, track_id, segment_id, keyframe_id)
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()