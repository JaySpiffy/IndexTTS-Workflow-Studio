#!/usr/bin/env python3
"""
Debug script to test SpeakerService directly
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

try:
    from backend.api.services.speaker_service import SpeakerService
    
    print("=== SpeakerService Debug ===")
    
    # Create service instance
    service = SpeakerService()
    
    # Check directories
    print(f"Speakers directory: {service.speakers_dir}")
    print(f"Speakers directory exists: {service.speakers_dir.exists()}")
    print(f"Speakers directory absolute path: {service.speakers_dir.absolute()}")
    
    # List files in directory
    if service.speakers_dir.exists():
        wav_files = list(service.speakers_dir.glob("*.wav"))
        print(f"Found {len(wav_files)} .wav files in directory:")
        for wav_file in wav_files[:5]:  # Show first 5
            print(f"  - {wav_file.name}")
        if len(wav_files) > 5:
            print(f"  ... and {len(wav_files) - 5} more")
    
    # Test list_speakers method
    print("\n=== Testing list_speakers() ===")
    speakers = service.list_speakers()
    print(f"list_speakers() returned {len(speakers)} speakers")
    
    if speakers:
        print("First few speakers:")
        for speaker in speakers[:3]:
            print(f"  - {speaker}")
    else:
        print("No speakers returned!")
    
    # Test get_speaker_info for narrator
    print("\n=== Testing get_speaker_info('narrator') ===")
    try:
        narrator_info = service.get_speaker_info("narrator")
        print(f"Narrator info: {narrator_info}")
    except Exception as e:
        print(f"Error getting narrator info: {e}")
    
    # Test get_speaker_info for narrator.wav
    print("\n=== Testing get_speaker_info('narrator.wav') ===")
    try:
        narrator_info = service.get_speaker_info("narrator.wav")
        print(f"Narrator info: {narrator_info}")
    except Exception as e:
        print(f"Error getting narrator info: {e}")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()