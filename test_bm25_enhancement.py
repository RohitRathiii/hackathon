#!/usr/bin/env python3
"""
Test BM25 enhancement locally
"""

import requests
import json
import time

# Test locally first
BASE_URL = "http://localhost:8004"  # Local test
ENDPOINT = f"{BASE_URL}/hackrx/run"

def test_bm25_enhancement():
    """Test BM25 enhanced chunk selection"""
    
    print("🧪 Testing BM25 Enhancement...")
    
    test_request = {
        "documents": "testdata/sample_insurance_policy.pdf",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "What are the maternity benefits?"
        ]
    }
    
    headers = {
        "Authorization": "Bearer default-token",
        "Content-Type": "application/json"
    }
    
    try:
        print("⏳ Testing enhanced BM25 + Gemini...")
        start_time = time.time()
        
        response = requests.post(
            ENDPOINT,
            json=test_request,
            headers=headers,
            timeout=60
        )
        
        duration = time.time() - start_time
        print(f"⏱️  Response time: {duration:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            
            for i, answer in enumerate(result['answers'], 1):
                print(f"\n🔍 Question {i}: {test_request['questions'][i-1]}")
                print(f"💬 Answer: {answer}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_bm25_enhancement()