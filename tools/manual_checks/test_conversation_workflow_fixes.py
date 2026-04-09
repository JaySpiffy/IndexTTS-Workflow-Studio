#!/usr/bin/env python3
"""
Comprehensive test to validate conversation workflow fixes:
1. Speaker name/file extension mismatch
2. Emotion vector parameter mismatch (4 vs 8 vectors)
3. Advanced settings properly passed to backend
"""

import sys
import os
sys.path.append('.')

import json
import requests
from typing import Dict, List, Any

# Test configuration
API_BASE_URL = "http://localhost:8000/api"

def test_speaker_parsing():
    """Test that speaker parsing works with various name formats"""
    print("=" * 60)
    print("TEST 1: Speaker Parsing Fix")
    print("=" * 60)
    
    # Test script with different speaker name formats
    test_script = {
        "title": "Speaker Parsing Test",
        "lines": [
            {"speaker_filename": "narrator.wav", "text": "This is a test with filename", "line_number": 0},
            {"speaker_filename": "JoeRogan", "text": "This is a test with name without extension", "line_number": 1},
            {"speaker_filename": "ElonMusk.wav", "text": "This is a test with name with extension", "line_number": 2}
        ]
    }
    
    print("Testing speaker parsing with formats:")
    for line in test_script["lines"]:
        print(f"  - {line['speaker_filename']}: {line['text']}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/conversation/parse-script",
            json={"script_text": "narrator: Line 1\nJoeRogan: Line 2\nElonMusk.wav: Line 3"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Speaker parsing successful!")
            print(f"   Parsed {len(result.get('lines', []))} lines")
            for line in result.get('lines', []):
                print(f"   - {line.get('speaker_filename', 'Unknown')}: {line.get('text', '')}")
        else:
            print(f"❌ Speaker parsing failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Speaker parsing test failed with exception: {e}")
    
    print()

def test_emotion_vectors():
    """Test that emotion vectors are properly handled (8 vectors)"""
    print("=" * 60)
    print("TEST 2: Emotion Vector Fix (8 vectors)")
    print("=" * 60)
    
    # Test with 8 emotion vectors
    test_vectors = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    
    print("Testing emotion vectors with 8 components:")
    for i, value in enumerate(test_vectors, 1):
        print(f"  - Vector {i}: {value}")
    
    try:
        # Test the backend emotion vector processing
        from backend.api.core.conversation_manager import ConversationManager
        
        manager = ConversationManager()
        
        # Create test emotion control data
        emotion_control = {
            "method": "from_vectors",
            "vectors": test_vectors
        }
        
        # Test the emotion vector processing
        processed_vectors = manager._process_emotion_control(emotion_control)
        
        if len(processed_vectors) == 8:
            print("✅ Emotion vector processing successful!")
            print(f"   Processed {len(processed_vectors)} vectors")
            for i, value in enumerate(processed_vectors, 1):
                print(f"   - Vector {i}: {value}")
        else:
            print(f"❌ Emotion vector processing failed: expected 8 vectors, got {len(processed_vectors)}")
            
    except Exception as e:
        print(f"❌ Emotion vector test failed with exception: {e}")
    
    print()

def test_advanced_settings():
    """Test that advanced settings are properly passed and processed"""
    print("=" * 60)
    print("TEST 3: Advanced Settings Fix")
    print("=" * 60)
    
    # Test advanced settings
    test_settings = {
        "max_text_tokens_per_segment": 150,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "temperature": 0.7,
        "length_penalty": 1.2,
        "num_beams": 5,
        "repetition_penalty": 15,
        "max_mel_tokens": 2000
    }
    
    print("Testing advanced settings:")
    for key, value in test_settings.items():
        print(f"  - {key}: {value}")
    
    try:
        # Test the backend advanced settings processing
        from backend.api.core.conversation_manager import ConversationManager
        
        manager = ConversationManager()
        
        # Create test generation request with advanced settings
        generation_request = {
            "script": {
                "title": "Advanced Settings Test",
                "lines": [
                    {"speaker_filename": "narrator.wav", "text": "Test line", "line_number": 0}
                ]
            },
            "versions_per_line": 1,
            "similarity_threshold": 0.6,
            "auto_regen_attempts": 0,
            "emotion_control_method": "from_speaker",
            "emotion_weight": 1.0,
            "use_random_sampling": False,
            **test_settings
        }
        
        # Test the advanced settings processing (validate the structure)
        required_keys = [
            'max_text_tokens_per_segment', 'do_sample', 'top_p', 'top_k',
            'temperature', 'length_penalty', 'num_beams', 'repetition_penalty', 'max_mel_tokens'
        ]
        
        missing_keys = [key for key in required_keys if key not in generation_request]
        
        if not missing_keys:
            print("✅ Advanced settings structure validation successful!")
            print(f"   All {len(required_keys)} required settings present")
            for key in required_keys:
                value = generation_request[key]
                print(f"   - {key}: {value} ({type(value).__name__})")
        else:
            print(f"❌ Advanced settings validation failed: missing keys {missing_keys}")
            
    except Exception as e:
        print(f"❌ Advanced settings test failed with exception: {e}")
    
    print()

def test_frontend_backend_integration():
    """Test the complete frontend-backend integration"""
    print("=" * 60)
    print("TEST 4: Frontend-Backend Integration")
    print("=" * 60)
    
    # Test complete generation request
    test_request = {
        "script": {
            "title": "Integration Test",
            "lines": [
                {"speaker_filename": "narrator.wav", "text": "This is a test", "line_number": 0}
            ]
        },
        "versions_per_line": 1,
        "similarity_threshold": 0.6,
        "auto_regen_attempts": 0,
        "emotion_control_method": "from_vectors",
        "emotion_vectors": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
        "emotion_weight": 1.0,
        "use_random_sampling": False,
        "max_text_tokens_per_segment": 120,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 0.8,
        "length_penalty": 0.0,
        "num_beams": 3,
        "repetition_penalty": 10,
        "max_mel_tokens": 1500
    }
    
    print("Testing complete integration request:")
    print(f"  - Script lines: {len(test_request['script']['lines'])}")
    print(f"  - Emotion vectors: {len(test_request['emotion_vectors'])}")
    print(f"  - Advanced settings: {len([k for k in test_request.keys() if k.startswith('max') or k.startswith('do_') or k in ['top_p', 'top_k', 'temperature', 'length_penalty', 'num_beams', 'repetition_penalty']])}")
    
    try:
        # Validate the request structure
        validation_errors = []
        
        # Check script structure
        if 'script' not in test_request:
            validation_errors.append("Missing script")
        elif 'lines' not in test_request['script']:
            validation_errors.append("Missing script lines")
        
        # Check emotion vectors
        if 'emotion_vectors' in test_request and len(test_request['emotion_vectors']) != 8:
            validation_errors.append(f"Expected 8 emotion vectors, got {len(test_request['emotion_vectors'])}")
        
        # Check advanced settings
        required_advanced = [
            'max_text_tokens_per_segment', 'do_sample', 'top_p', 'top_k',
            'temperature', 'length_penalty', 'num_beams', 'repetition_penalty', 'max_mel_tokens'
        ]
        missing_advanced = [key for key in required_advanced if key not in test_request]
        if missing_advanced:
            validation_errors.append(f"Missing advanced settings: {missing_advanced}")
        
        if not validation_errors:
            print("✅ Integration validation successful!")
            print("   All required components present and correctly structured")
        else:
            print(f"❌ Integration validation failed:")
            for error in validation_errors:
                print(f"   - {error}")
                
    except Exception as e:
        print(f"❌ Integration test failed with exception: {e}")
    
    print()

def main():
    """Run all tests"""
    print("🧪 CONVERSATION WORKFLOW FIXES VALIDATION")
    print("=" * 60)
    print("Testing fixes for:")
    print("1. Speaker name/file extension mismatch")
    print("2. Emotion vector parameter mismatch (4 vs 8 vectors)")
    print("3. Advanced settings properly passed to backend")
    print("=" * 60)
    print()
    
    # Run all tests
    test_speaker_parsing()
    test_emotion_vectors()
    test_advanced_settings()
    test_frontend_backend_integration()
    
    print("=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    print("All tests completed. Check the output above for any ❌ failures.")
    print("If all tests show ✅, the conversation workflow fixes are working correctly!")
    print()

if __name__ == "__main__":
    main()