#!/usr/bin/env python3
"""
Test the enhanced local API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8004"
ENDPOINT = f"{BASE_URL}/hackrx/run"

def test_local_enhanced():
    """Test enhanced Gemini locally"""
    
    print("🧪 Testing Enhanced Local API...")
    print(f"📍 URL: {ENDPOINT}")
    
    # Test with a simple question first
    test_request = {
        "documents": "testdata/CHOTGDP23004V012223.pdf",
        "questions": [
            "What is the grace period for premium payment?"
        ]
    }
    
    headers = {
        "Authorization": "Bearer default-token",
        "Content-Type": "application/json"
    }
    
    try:
        print("⏳ Testing health endpoint first...")
        health_response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"🏥 Health Status: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"📊 Health Data: {health_response.json()}")
        
        print("\n⏳ Sending main request...")
        start_time = time.time()
        
        response = requests.post(ENDPOINT, json=test_request, headers=headers, timeout=60)
        duration = time.time() - start_time
        
        print(f"⏱️  Response time: {duration:.2f} seconds")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📋 Answer: {result['answers'][0]}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_local_enhanced()