#!/usr/bin/env python3
"""
Test Render deployment
"""

import requests
import json
import time

# Update this URL after deployment
BASE_URL = "https://legal-bert-api-[your-service].onrender.com"  # Replace with actual Render URL
ENDPOINT = f"{BASE_URL}/hackrx/run"

def test_render_deployment():
    """Test the Render deployed API"""
    
    print("🚀 Testing Render Deployment...")
    print(f"📍 URL: {ENDPOINT}")
    
    # Test data
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
        print("⏳ Testing health endpoint...")
        health_response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"🏥 Health Status: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"📊 Health Data: {health_response.json()}")
        
        print("\n⏳ Sending main request...")
        start_time = time.time()
        
        response = requests.post(
            ENDPOINT,
            json=test_request,
            headers=headers,
            timeout=120
        )
        
        duration = time.time() - start_time
        print(f"⏱️  Render Response time: {duration:.2f} seconds")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📋 Answers received: {len(result['answers'])}")
            
            # Performance analysis
            if duration < 15:
                print("🎉 EXCELLENT: Sub-15 second performance!")
            elif duration < 30:
                print("✅ GOOD: Under 30 seconds")
            else:
                print("⚠️  Needs optimization")
            
            for i, answer in enumerate(result['answers'], 1):
                print(f"\n🔍 Question {i}: {test_request['questions'][i-1][:50]}...")
                print(f"💬 Answer: {answer[:150]}...")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_full_hackathon_questions():
    """Test with all 10 hackathon questions"""
    
    print("\n" + "="*80)
    print("🧪 Testing Render with ALL 10 Hackathon Questions...")
    
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
        "Content-Type": "application/json"
    }
    
    try:
        print("⏳ Sending FULL hackathon request...")
        start_time = time.time()
        
        response = requests.post(ENDPOINT, json=test_request, headers=headers, timeout=120)
        duration = time.time() - start_time
        
        print(f"⏱️  Render FULL TEST Response time: {duration:.2f} seconds")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"📋 All {len(result['answers'])} questions answered")
            
            # Performance analysis
            if duration < 20:
                print("🎉 EXCELLENT: Render performance under 20 seconds!")
            elif duration < 30:
                print("✅ GOOD: Under 30 seconds")
            else:
                print("⚠️  NEEDS MORE OPTIMIZATION")
                
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Render Deployment Test")
    print("=" * 50)
    
    # Update the URL first
    if "[your-service]" in BASE_URL:
        print("❌ Please update BASE_URL with your actual Render URL")
        print("   Get it from your Render dashboard")
        exit(1)
    
    test_render_deployment()
    test_full_hackathon_questions()