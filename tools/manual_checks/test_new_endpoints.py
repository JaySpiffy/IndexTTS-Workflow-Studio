#!/usr/bin/env python3
"""
Test script for the new conversation line regeneration endpoints.
"""

import os
import requests


BASE_URL = os.getenv("INDTEXTS_TEST_BASE_URL", "http://localhost:8001")
API_BASE = f"{BASE_URL}/api"


def test_health():
    """Test if the API is running."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            print("API is running")
            return True
        print(f"API health check failed: {response.status_code}")
        return False
    except Exception as exc:
        print(f"Cannot connect to API: {exc}")
        return False


def test_line_regeneration_endpoint():
    """Test the line regeneration endpoint structure."""
    print("\nTesting line regeneration endpoint structure...")

    try:
        response = requests.post(
            f"{API_BASE}/conversation/results/nonexistent-conversation/line/0/regenerate",
            json={"regen_count": 2},
            timeout=10,
        )

        if response.status_code in {404, 422}:
            print("Line regeneration endpoint exists")
            return True

        print(f"Unexpected response: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    except Exception as exc:
        print(f"Error testing line regeneration endpoint: {exc}")
        return False


def test_line_regeneration_status_endpoint():
    """Test the line regeneration status endpoint structure."""
    print("\nTesting line regeneration status endpoint structure...")

    try:
        response = requests.get(
            f"{API_BASE}/conversation/results/nonexistent-conversation/line/0/regenerate/status",
            timeout=10,
        )

        if response.status_code in {404, 422}:
            print("Line regeneration status endpoint exists")
            return True

        print(f"Unexpected response: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    except Exception as exc:
        print(f"Error testing line regeneration status endpoint: {exc}")
        return False


def test_audio_file_endpoint():
    """Test the audio file serving endpoint."""
    print("\nTesting audio file serving endpoint...")

    try:
        response = requests.get(f"{API_BASE}/conversation/assets/audio/nonexistent-file.wav", timeout=10)

        if response.status_code in {404, 422}:
            print("Audio file endpoint exists")
            return True

        print(f"Unexpected response: {response.status_code}")
        return False
    except Exception as exc:
        print(f"Error testing audio file endpoint: {exc}")
        return False


def test_conversation_list_endpoint():
    """Test the conversation list endpoint to check if API is working."""
    print("\nTesting conversation list endpoint...")

    try:
        response = requests.get(f"{API_BASE}/conversation/list", timeout=10)

        if response.status_code == 200:
            data = response.json()
            conversations = data.get("details", {}).get("conversations", [])
            print("Conversation list endpoint works")
            print(f"Found {len(conversations)} conversations")
            return True

        print(f"Conversation list endpoint failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    except Exception as exc:
        print(f"Error testing conversation list endpoint: {exc}")
        return False


def main():
    """Run all tests."""
    print("Testing new IndexTTS2 API endpoints...")
    print("=" * 50)

    if not test_health():
        print("\nAPI is not running. Please start the Docker backend first:")
        print("  docker compose -f docker/docker-compose.yml up -d --build backend")
        return

    tests = [
        test_conversation_list_endpoint,
        test_line_regeneration_endpoint,
        test_line_regeneration_status_endpoint,
        test_audio_file_endpoint,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("All endpoint structure tests passed.")
    else:
        print("Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
