#!/usr/bin/env python3
"""
Test the enhanced Gemini implementation
"""

import requests
import json
import time

BASE_URL = "https://legal-bert-api-95314557912.us-central1.run.app"
ENDPOINT = f"{BASE_URL}/hackrx/run"

def test_enhanced_quality():
    """Test with enhanced Gemini for quality"""
    
    print("🧪 Testing Enhanced Gemini Quality...")
    print(f"📍 URL: {ENDPOINT}")
    
    # Test with hackathon questions
    test_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    headers = {
        "Authorization": "Bearer default-token",
        "Content-Type": "application/json"
    }
    
    try:
        print("⏳ Sending request...")
        start_time = time.time()
        
        response = requests.post(ENDPOINT, json=test_request, headers=headers, timeout=120)
        duration = time.time() - start_time
        
        print(f"⏱️  Response time: {duration:.2f} seconds")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            
            for i, (question, answer) in enumerate(zip(test_request['questions'], result['answers']), 1):
                print(f"\n🔍 Question {i}: {question}")
                print(f"💬 Enhanced Answer: {answer}")
                print("-" * 80)
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_enhanced_quality()