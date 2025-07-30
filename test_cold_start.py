#!/usr/bin/env python3
"""
Test cold start performance with different document
"""

import requests
import json
import time

BASE_URL = "https://legal-bert-api-95314557912.us-central1.run.app"
ENDPOINT = f"{BASE_URL}/hackrx/run"

def test_cold_start():
    """Test with a different document to simulate cold start"""
    
    print("🧪 Testing COLD START performance...")
    print(f"📍 URL: {ENDPOINT}")
    
    # Use a different document URL to avoid cache
    test_request = {
        "documents": "https://hackrx.in/policies/EDLHLGA23009V012223.pdf",  # Different doc
        "questions": [
            "What is this document about?",
            "What are the main topics covered?",
            "Is this a legal document?"
        ]
    }
    
    headers = {
        "Authorization": "Bearer default-token",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        print("⏳ Sending request with NEW document...")
        start_time = time.time()
        
        response = requests.post(
            ENDPOINT,
            json=test_request,
            headers=headers,
            timeout=120
        )
        
        duration = time.time() - start_time
        print(f"⏱️  COLD START Response time: {duration:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"📋 Answers: {len(result.get('answers', []))}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_multiple_docs():
    """Test with multiple different documents"""
    
    docs = [
        "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf"
    ]
    
    for i, doc_url in enumerate(docs, 1):
        print(f"\n🧪 Test {i}: Different document")
        test_request = {
            "documents": doc_url,
            "questions": ["What is the main topic of this document?"]
        }
        
        headers = {
            "Authorization": "Bearer default-token",
            "Content-Type": "application/json"
        }
        
        try:
            start_time = time.time()
            response = requests.post(ENDPOINT, json=test_request, headers=headers, timeout=120)
            duration = time.time() - start_time
            
            print(f"⏱️  Document {i} time: {duration:.2f} seconds")
            print(f"📊 Status: {response.status_code}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 COLD START Performance Test")
    print("=" * 60)
    test_cold_start()
    print("\n" + "=" * 60)
    test_multiple_docs()