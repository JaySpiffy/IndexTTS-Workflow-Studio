#!/usr/bin/env python3
"""
Debug script to identify the exact root causes of conversation workflow issues.
"""

import os
import sys
import json
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def debug_speaker_service():
    """Debug the speaker service to understand why it's not finding speakers."""
    print("=== DEBUGGING SPEAKER SERVICE ===")
    
    try:
        from backend.api.services.speaker_service import SpeakerService
        
        service = SpeakerService()
        print(f"Speaker service speakers_dir: {service.speakers_dir}")
        print(f"Speaker service speakers_dir exists: {service.speakers_dir.exists()}")
        print(f"Speaker service speakers_dir absolute: {service.speakers_dir.absolute()}")
        
        if service.speakers_dir.exists():
            wav_files = list(service.speakers_dir.glob("*.wav"))
            print(f"Found {len(wav_files)} .wav files in speakers_dir:")
            for f in wav_files[:5]:  # Show first 5
                print(f"  - {f.name} (exists: {f.exists()})")
            if len(wav_files) > 5:
                print(f"  ... and {len(wav_files) - 5} more")
        
        # Try to list speakers
        speakers = service.list_speakers()
        print(f"list_speakers() returned {len(speakers)} speakers")
        
        return speakers
        
    except Exception as e:
        print(f"Error debugging speaker service: {e}")
        import traceback
        traceback.print_exc()
        return []

def debug_file_utils_parsing():
    """Debug the file_utils parsing logic."""
    print("\n=== DEBUGGING FILE_UTILS PARSING ===")
    
    try:
        from backend.api.core.file_utils import parse_conversation_script
        
        # Test script with different speaker names
        test_script = """narrator: Welcome to the show.
JoeRogan: Thanks for having me.
ElonMusk: It's great to be here."""
        
        # Test with different available_speakers lists
        test_cases = [
            # Case 1: Exact filenames with .wav
            ["narrator.wav", "JoeRogan.wav", "ElonMusk.wav"],
            # Case 2: Names without .wav
            ["narrator", "JoeRogan", "ElonMusk"],
            # Case 3: Mixed case
            ["Narrator.wav", "joerogan.wav", "elonmusk.wav"],
            # Case 4: Empty list
            []
        ]
        
        for i, available_speakers in enumerate(test_cases, 1):
            print(f"\nTest case {i}: available_speakers = {available_speakers}")
            try:
                result = parse_conversation_script(test_script, available_speakers)
                print(f"  Result: {len(result)} lines parsed")
                for line in result:
                    print(f"    - {line['speaker']}: {line['text'][:30]}...")
            except Exception as e:
                print(f"  Error: {e}")
        
    except Exception as e:
        print(f"Error debugging file_utils: {e}")
        import traceback
        traceback.print_exc()

def debug_conversation_manager():
    """Debug the conversation manager emotion control."""
    print("\n=== DEBUGGING CONVERSATION MANAGER ===")
    
    try:
        from backend.api.core.conversation_manager import ConversationManager
        
        manager = ConversationManager()
        
        # Test emotion control with different vector configurations
        test_cases = [
            # Case 1: 4 vectors (what frontend sends)
            {"emotion_control": "from_vectors", "vec1": 0.5, "vec2": 0.3, "vec3": 0.2, "vec4": 0.1},
            # Case 2: 8 vectors (what backend expects)
            {"emotion_control": "from_vectors", "vec1": 0.5, "vec2": 0.3, "vec3": 0.2, "vec4": 0.1, 
             "vec5": 0.1, "vec6": 0.1, "vec7": 0.1, "vec8": 0.1},
            # Case 3: Missing vectors
            {"emotion_control": "from_vectors", "vec1": 0.5, "vec2": 0.3}
        ]
        
        for i, emotion_params in enumerate(test_cases, 1):
            print(f"\nTest case {i}: emotion_params = {emotion_params}")
            try:
                result = manager._process_emotion_control(emotion_params)
                print(f"  Result: {result}")
            except Exception as e:
                print(f"  Error: {e}")
        
    except Exception as e:
        print(f"Error debugging conversation manager: {e}")
        import traceback
        traceback.print_exc()

def debug_speaker_file_locations():
    """Debug different speaker file locations."""
    print("\n=== DEBUGGING SPEAKER FILE LOCATIONS ===")
    
    locations = [
        "shared/audio/speakers",
        "speakers", 
        "shared/audio/source_clips",
        "source_clips"
    ]
    
    for location in locations:
        path = Path(location)
        print(f"\nLocation: {location}")
        print(f"  Path exists: {path.exists()}")
        print(f"  Absolute path: {path.absolute()}")
        
        if path.exists():
            wav_files = list(path.glob("*.wav"))
            print(f"  Found {len(wav_files)} .wav files:")
            for f in wav_files[:3]:  # Show first 3
                print(f"    - {f.name}")
            if len(wav_files) > 3:
                print(f"    ... and {len(wav_files) - 3} more")

def main():
    """Main debug function."""
    print("CONVERSATION WORKFLOW DEBUG SCRIPT")
    print("=" * 50)
    
    # Debug speaker file locations
    debug_speaker_file_locations()
    
    # Debug speaker service
    speakers = debug_speaker_service()
    
    # Debug file_utils parsing with actual speakers
    print("\n=== DEBUGGING WITH ACTUAL SPEAKERS ===")
    if speakers:
        speaker_names = [s["name"] for s in speakers[:5]]  # Use first 5 speakers
        print(f"Testing with actual speakers: {speaker_names}")
        
        try:
            from backend.api.core.file_utils import parse_conversation_script
            test_script = f"{speaker_names[0]}: Hello.\n{speaker_names[1]}: Hi there."
            result = parse_conversation_script(test_script, speaker_names)
            print(f"Parsing result: {len(result)} lines")
        except Exception as e:
            print(f"Error with actual speakers: {e}")
    
    # Debug file_utils parsing
    debug_file_utils_parsing()
    
    # Debug conversation manager
    debug_conversation_manager()
    
    print("\n=== SUMMARY ===")
    print("Root causes identified:")
    print("1. Speaker service path mismatch - check where it's looking vs where files are")
    print("2. Speaker name matching issues - exact vs fuzzy matching")
    print("3. Emotion vector parameter mismatch - frontend sends 4, backend expects 8")
    print("4. Advanced settings may not be properly passed through")

if __name__ == "__main__":
    main()