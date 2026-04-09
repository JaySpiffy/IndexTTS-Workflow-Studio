#!/usr/bin/env python3
"""
Test script to validate conversation parsing and emotion vector issues
"""

import sys
import os
sys.path.append('backend')

from backend.api.core.file_utils import parse_conversation_script

def test_speaker_parsing():
    """Test speaker parsing with different speaker name formats"""
    print("=== Testing Speaker Parsing ===")
    
    # Test script with different speaker name formats
    test_script = """narrator: Hello, this is a test.
JoeRogan: This is Joe speaking.
ElonMusk.wav: And this is Elon with .wav extension.
UnknownSpeaker: This should fail."""
    
    print(f"Test script:")
    print(test_script)
    print("\n" + "="*50)
    
    parsed_script, errors = parse_conversation_script(test_script)
    
    print(f"\nParsed script ({len(parsed_script)} lines):")
    for line in parsed_script:
        print(f"  - {line}")
    
    print(f"\nErrors ({len(errors)}):")
    for error in errors:
        print(f"  - {error}")
    
    return parsed_script, errors

def test_emotion_vectors():
    """Test emotion vector processing"""
    print("\n=== Testing Emotion Vector Processing ===")
    
    # Simulate frontend sending 4 vectors + 4 zeros
    frontend_vectors = [0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    
    print(f"Frontend sends vectors: {frontend_vectors}")
    print(f"Sum of vectors: {sum(frontend_vectors)}")
    print(f"Number of vectors: {len(frontend_vectors)}")
    
    # Test vector sum validation
    if sum(frontend_vectors) > 1.5:
        print("ERROR: Vector sum exceeds 1.5!")
    else:
        print("Vector sum is within acceptable range")
    
    return frontend_vectors

if __name__ == "__main__":
    print("Testing conversation workflow issues...")
    
    # Test speaker parsing
    parsed_script, errors = test_speaker_parsing()
    
    # Test emotion vectors
    vectors = test_emotion_vectors()
    
    print("\n=== Summary ===")
    print(f"Speaker parsing: {'SUCCESS' if len(parsed_script) > 0 else 'FAILED'}")
    print(f"Emotion vectors: {'SUCCESS' if len(vectors) == 8 else 'FAILED'}")
    print(f"Errors found: {len(errors)}")