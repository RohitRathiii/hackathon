#!/usr/bin/env python3
import requests
import json

url = "https://legal-bert-api-95314557912.us-central1.run.app/hackrx/run"

payload = {
    "documents": "testdata/BAJHLIP23020V012223.pdf",
    "questions": ["What is the grace period for premium payment?"]
}

headers = {
    "Authorization": "Bearer default-token",
    "Content-Type": "application/json"
}

print("🧪 Testing with 120 second timeout...")
try:
    response = requests.post(url, json=payload, headers=headers, timeout=120)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Answer: {result['answers'][0]}")
    else:
        print(f"❌ Error: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")