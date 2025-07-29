#!/usr/bin/env python3
"""
Test script for the LEGAL-BERT Document Query System
"""

import requests
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_endpoint():
    """Test the /hackrx/run endpoint with sample data"""
    
    url = "http://localhost:8004/hackrx/run"
    bearer_token = os.getenv("API_BEARER_TOKEN", "default-token")
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }
    
    # Test payload with sample insurance document
    payload = {
        "documents": "testdata/BAJHLIP23020V012223.pdf",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?",
            "Does this policy cover maternity expenses?"
        ]
    }
    
    print("🧪 Testing LEGAL-BERT Document Query System")
    print("=" * 50)
    print(f"📡 Endpoint: {url}")
    print(f"📄 Document: {payload['documents']}")
    print(f"❓ Questions: {len(payload['questions'])}")
    print()
    
    start_time = time.time()
    
    try:
        print("⏳ Sending request...")
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        duration = time.time() - start_time
        print(f"⏱️  Response time: {duration:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Validate response format
            if "answers" in result and isinstance(result["answers"], list):
                print(f"\n✅ Success! Received {len(result['answers'])} answers")
                print("=" * 50)
                
                for i, answer in enumerate(result["answers"], 1):
                    question = payload["questions"][i-1]
                    print(f"\n📝 Question {i}: {question}")
                    print(f"💡 Answer: {answer}")
                    print("-" * 30)
                
                # Performance info
                if 'x-process-time' in response.headers:
                    process_time = float(response.headers['x-process-time'])
                    print(f"\n⚡ Server Processing Time: {process_time:.2f} seconds")
                
                print(f"\n🎯 Test completed successfully!")
                
            else:
                print("❌ Invalid response format - missing 'answers' field")
                print(f"Response: {json.dumps(result, indent=2)}")
                
        else:
            print(f"❌ HTTP Error {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Error text: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>120 seconds)")
        print("💡 This might be normal for the first run (LEGAL-BERT model download)")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed")
        print("💡 Make sure the API is running: python run_api.py")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8004/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("🟢 API Health Check: PASSED")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            return True
        else:
            print(f"🔴 API Health Check: FAILED (Status: {response.status_code})")
            return False
    except:
        print("🔴 API Health Check: FAILED (Connection error)")
        return False

def main():
    """Main test function"""
    print("🚀 LEGAL-BERT Document Query System - Test Suite")
    print("=" * 60)
    
    # Test health endpoint first
    if test_health_endpoint():
        print()
        test_api_endpoint()
    else:
        print("\n💡 Start the API first: python run_api.py")

if __name__ == "__main__":
    main()