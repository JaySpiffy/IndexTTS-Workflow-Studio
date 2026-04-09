"""
Test script for standalone FastAPI implementation.
Verifies that the FastAPI server works without Gradio dependencies.
"""

import os
import sys
import requests
import time
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_server_health():
    """Test server health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_tts_generation():
    """Test TTS generation endpoint."""
    try:
        # Check if we have any speaker files
        speakers_dir = Path("speakers")
        if not speakers_dir.exists():
            print("❌ Speakers directory not found")
            return False
        
        speaker_files = list(speakers_dir.glob("*.wav"))
        if not speaker_files:
            print("❌ No speaker files found for testing")
            return False
        
        speaker_file = speaker_files[0].name
        
        # Test TTS generation
        tts_request = {
            "speaker_filename": speaker_file,
            "text": "Hello, this is a test of the standalone FastAPI server.",
            "emotion_control_method": "from_speaker",
            "use_random_sampling": False,
            "max_text_tokens_per_segment": 120,
            "do_sample": True,
            "top_p": 0.8,
            "top_k": 30,
            "temperature": 0.8,
            "length_penalty": 0.0,
            "num_beams": 3,
            "repetition_penalty": 10.0,
            "max_mel_tokens": 1500
        }
        
        response = requests.post(
            "http://localhost:8000/api/conversation/generate-single",
            json=tts_request,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ TTS generation passed: {result.get('message', 'No message')}")
            audio_path = result.get("audio_path")
            if audio_path and Path(audio_path).exists():
                print(f"✅ Audio file created: {audio_path}")
            return True
        else:
            print(f"❌ TTS generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ TTS generation error: {str(e)}")
        return False

def test_api_docs():
    """Test API documentation endpoints."""
    try:
        # Test Swagger UI
        response = requests.get("http://localhost:8000/docs", timeout=10)
        if response.status_code == 200:
            print("✅ Swagger UI accessible")
        else:
            print(f"❌ Swagger UI failed: {response.status_code}")
        
        # Test ReDoc
        response = requests.get("http://localhost:8000/redoc", timeout=10)
        if response.status_code == 200:
            print("✅ ReDoc accessible")
        else:
            print(f"❌ ReDoc failed: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ API docs error: {str(e)}")
        return False

def test_speaker_list():
    """Test speaker list endpoint."""
    try:
        response = requests.get("http://localhost:8000/api/speakers/list", timeout=10)
        if response.status_code == 200:
            speakers = response.json()
            print(f"✅ Speaker list passed: {len(speakers.get('speakers', []))} speakers found")
            return True
        else:
            print(f"❌ Speaker list failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Speaker list error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Standalone FastAPI Server")
    print("=" * 50)
    
    # Wait a moment for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    # Run tests
    tests = [
        ("Health Check", test_server_health),
        ("API Documentation", test_api_docs),
        ("Speaker List", test_speaker_list),
        ("TTS Generation", test_tts_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"⚠️ Test '{test_name}' failed")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Standalone FastAPI server is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the server logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())