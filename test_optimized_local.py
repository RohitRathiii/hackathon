#!/usr/bin/env python3
"""
Test the optimized version locally
"""

import requests
import json
import time

BASE_URL = "http://localhost:8004"
ENDPOINT = f"{BASE_URL}/hackrx/run"

def test_optimized_speed():
    """Test optimized version for speed"""
    
    print("🚀 Testing OPTIMIZED Version for Speed...")
    print(f"📍 URL: {ENDPOINT}")
    
    # Test with fewer questions first
    test_request = {
        "documents": "testdata/BAJHLIP23020V012223.pdf",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?"
        ]
    }
    
    headers = {
        "Authorization": "Bearer default-token",
        "Content-Type": "application/json"
    }
    
    try:
        print("⏳ Testing health endpoint...")
        health_response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"🏥 Health Status: {health_response.status_code}")
        
        print("\n⏳ Sending optimized request...")
        start_time = time.time()
        
        response = requests.post(ENDPOINT, json=test_request, headers=headers, timeout=90)
        duration = time.time() - start_time
        
        print(f"⏱️  OPTIMIZED Response time: {duration:.2f} seconds")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📋 Number of answers: {len(result['answers'])}")
            
            for i, (question, answer) in enumerate(zip(test_request['questions'], result['answers']), 1):
                print(f"\n🔍 Q{i}: {question[:50]}...")
                print(f"💬 A{i}: {answer[:100]}...")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_optimized_speed()