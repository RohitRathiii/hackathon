#!/usr/bin/env python3
"""
Test with EXACT hackathon format and URL
"""

import requests
import json
import time

# Your deployed API URL
BASE_URL = "https://legal-bert-api-95314557912.us-central1.run.app"
ENDPOINT = f"{BASE_URL}/hackrx/run"

def test_exact_hackathon():
    """Test with exact hackathon request"""
    
    print("🧪 Testing with EXACT hackathon format...")
    print(f"📍 URL: {ENDPOINT}")
    
    # EXACT hackathon request format
    test_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
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
            
            # Validate exact format
            if 'answers' in result and isinstance(result['answers'], list):
                print("✅ Response format is correct!")
                print(f"📋 Number of answers: {len(result['answers'])}")
                print(f"📋 Number of questions: {len(test_request['questions'])}")
                
                if len(result['answers']) == len(test_request['questions']):
                    print("✅ Answer count matches question count!")
                else:
                    print("❌ Answer count mismatch!")
                
                # Show first few answers
                for i, answer in enumerate(result['answers'][:3], 1):
                    print(f"\n🔍 Question {i}: {test_request['questions'][i-1][:60]}...")
                    print(f"💬 Answer: {answer[:150]}...")
                    
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
    print("🚀 EXACT Hackathon Format Test")
    print("=" * 60)
    test_exact_hackathon()