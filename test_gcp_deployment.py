#!/usr/bin/env python3
"""
Test script for GCP deployment
"""

import requests
import json
import time

# Update this URL after deployment
BASE_URL = "https://legal-bert-api-95314557912.us-central1.run.app"
ENDPOINT = f"{BASE_URL}/hackrx/run"

def test_deployment():
    """Test the deployed API"""
    
    print("🧪 Testing GCP deployment...")
    print(f"📍 URL: {ENDPOINT}")
    
    # Test data
    test_request = {
        "documents": "testdata/BAJHLIP23020V012223.pdf",
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
        print("⏳ Sending request...")
        start_time = time.time()
        
        response = requests.post(
            ENDPOINT,
            json=test_request,
            headers=headers,
            timeout=60
        )
        
        duration = time.time() - start_time
        print(f"⏱️  Response time: {duration:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📋 Answers received: {len(result['answers'])}")
            
            for i, answer in enumerate(result['answers'], 1):
                print(f"\n🔍 Question {i}: {test_request['questions'][i-1]}")
                print(f"💬 Answer: {answer[:200]}...")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"📊 Status: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("🚀 GCP Deployment Test")
    print("=" * 50)
    
    # Update the URL first
    if "[REPLACE-WITH-ACTUAL-HASH]" in BASE_URL:
        print("❌ Please update BASE_URL with your actual deployment URL")
        print("   Get it from: gcloud run services list")
        exit(1)
    
    test_health()
    print()
    test_deployment()