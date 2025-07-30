#!/usr/bin/env python3
"""
Test the optimized version on GCP
"""

import requests
import json
import time

BASE_URL = "https://legal-bert-api-95314557912.us-central1.run.app"
ENDPOINT = f"{BASE_URL}/hackrx/run"

def test_optimized_performance():
    """Test optimized version performance"""
    
    print("🚀 Testing OPTIMIZED Version on GCP...")
    print(f"📍 URL: {ENDPOINT}")
    
    # Test with hackathon data
    test_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    headers = {
        "Authorization": "Bearer default-token",
        "Content-Type": "application/json"
    }
    
    try:
        print("⏳ Sending optimized request...")
        start_time = time.time()
        
        response = requests.post(ENDPOINT, json=test_request, headers=headers, timeout=120)
        duration = time.time() - start_time
        
        print(f"⏱️  OPTIMIZED Response time: {duration:.2f} seconds")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📋 Number of answers: {len(result['answers'])}")
            
            # Show performance comparison
            print(f"\n📈 PERFORMANCE COMPARISON:")
            print(f"   Before: 82.55 seconds")
            print(f"   After:  {duration:.2f} seconds")
            print(f"   Improvement: {((82.55 - duration) / 82.55 * 100):.1f}%")
            
            print(f"\n📝 SAMPLE ANSWERS:")
            for i, (question, answer) in enumerate(zip(test_request['questions'], result['answers']), 1):
                print(f"\n🔍 Q{i}: {question[:60]}...")
                print(f"💬 A{i}: {answer[:150]}...")
                print("-" * 60)
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_full_hackathon_questions():
    """Test with all 10 hackathon questions"""
    
    print("\n" + "="*80)
    print("🧪 Testing with ALL 10 Hackathon Questions...")
    
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
        
        print(f"⏱️  FULL TEST Response time: {duration:.2f} seconds")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"📋 All {len(result['answers'])} questions answered")
            
            # Performance analysis
            if duration < 30:
                print("🎉 EXCELLENT: Under 30 seconds!")
            elif duration < 45:
                print("✅ GOOD: Under 45 seconds")
            else:
                print("⚠️  NEEDS MORE OPTIMIZATION: Over 45 seconds")
                
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_optimized_performance()
    test_full_hackathon_questions()