#!/usr/bin/env python3
"""
Test script for Heroku deployed API
"""

import requests
import json
import time

def test_heroku_api():
    """Test the deployed Heroku API"""
    
    # Replace with your actual Heroku app URL
    base_url = "https://legal-bert-hackathon-api.herokuapp.com"  # Update this!
    url = f"{base_url}/hackrx/run"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer default-token"
    }
    
    # Test with hackathon document URL
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    print("🧪 Testing Heroku Deployed API")
    print("=" * 50)
    print(f"📡 URL: {url}")
    print(f"📄 Document: Hackathon Policy PDF")
    print(f"❓ Questions: {len(payload['questions'])}")
    print()
    
    # Test health endpoint first
    try:
        health_response = requests.get(f"{base_url}/health", timeout=10)
        if health_response.status_code == 200:
            print("🟢 Health Check: PASSED")
        else:
            print(f"🔴 Health Check: FAILED ({health_response.status_code})")
    except:
        print("🔴 Health Check: FAILED (Connection error)")
    
    print()
    start_time = time.time()
    
    try:
        print("⏳ Sending request to Heroku...")
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        duration = time.time() - start_time
        print(f"⏱️  Response time: {duration:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if "answers" in result and isinstance(result["answers"], list):
                print(f"\n✅ SUCCESS! Received {len(result['answers'])} answers")
                print("=" * 50)
                
                for i, answer in enumerate(result["answers"], 1):
                    question = payload["questions"][i-1]
                    print(f"\n📝 Question {i}: {question}")
                    print(f"💡 Answer: {answer}")
                    print("-" * 30)
                
                print(f"\n🎯 Heroku deployment test SUCCESSFUL!")
                print(f"🌐 Your API URL: {url}")
                print(f"📋 Ready for hackathon submission!")
                
            else:
                print("❌ Invalid response format")
                print(f"Response: {json.dumps(result, indent=2)}")
                
        else:
            print(f"❌ HTTP Error {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        print("💡 This might indicate the app is sleeping (Heroku free tier)")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed")
        print("💡 Check if the Heroku app is running: heroku ps")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_heroku_api()