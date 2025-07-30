#!/usr/bin/env python3
"""
Test script that matches exact hackathon format
"""

import requests
import json
import time

# Your deployed API URL
BASE_URL = "https://legal-bert-api-95314557912.us-central1.run.app"
ENDPOINT = f"{BASE_URL}/hackrx/run"

def test_hackathon_format():
    """Test with exact hackathon format"""
    
    print("🧪 Testing with exact hackathon format...")
    print(f"📍 URL: {ENDPOINT}")
    
    # Test data matching hackathon format exactly
    test_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",  # Hackathon URL
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "What are the maternity benefits?"
        ]
    }
    
    headers = {
        "Authorization": "Bearer default-token",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        print("⏳ Sending request...")
        start_time = time.time()
        
        response = requests.post(
            ENDPOINT,
            json=test_request,
            headers=headers,
            timeout=120
        )
        
        duration = time.time() - start_time
        print(f"⏱️  Response time: {duration:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📋 Response format: {type(result)}")
            print(f"📋 Keys: {list(result.keys())}")
            print(f"📋 Answers count: {len(result.get('answers', []))}")
            
            # Validate exact format
            if 'answers' in result and isinstance(result['answers'], list):
                print("✅ Response format is correct!")
                for i, answer in enumerate(result['answers'], 1):
                    print(f"\n🔍 Question {i}: {test_request['questions'][i-1]}")
                    print(f"💬 Answer: {answer[:200]}...")
            else:
                print("❌ Response format is incorrect!")
                print(f"Expected: {{'answers': [...]}}")
                print(f"Got: {result}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (>120s)")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Hackathon Format Test")
    print("=" * 50)
    test_hackathon_format()