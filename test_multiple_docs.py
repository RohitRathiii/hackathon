#!/usr/bin/env python3
"""
Test the optimized API with multiple insurance documents
"""

import requests
import json
import time
import os

BASE_URL = "https://legal-bert-api-95314557912.us-central1.run.app"
ENDPOINT = f"{BASE_URL}/hackrx/run"

# Test documents from testdata folder
TEST_DOCUMENTS = [
    {
        "name": "Bajaj Health Insurance",
        "file": "testdata/BAJHLIP23020V012223.pdf",
        "description": "Bajaj Allianz Health Insurance Policy"
    },
    {
        "name": "Cholamandalam Health",
        "file": "testdata/CHOTGDP23004V012223.pdf", 
        "description": "Cholamandalam Health Insurance"
    },
    {
        "name": "Edelweiss Health",
        "file": "testdata/EDLHLGA23009V012223.pdf",
        "description": "Edelweiss Health Insurance"
    },
    {
        "name": "HDFC Health Insurance",
        "file": "testdata/HDFHLIP23024V072223.pdf",
        "description": "HDFC Health Insurance Policy"
    },
    {
        "name": "ICICI Health Insurance", 
        "file": "testdata/ICIHLIP22012V012223.pdf",
        "description": "ICICI Lombard Health Insurance"
    }
]

# Standard insurance questions to test across all documents
STANDARD_QUESTIONS = [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?",
    "Does this policy cover maternity expenses?",
    "What is the sum insured or coverage amount?",
    "Are there any exclusions mentioned in the policy?"
]

def test_single_document(doc_info, questions):
    """Test a single document with given questions"""
    
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {doc_info['name']}")
    print(f"📄 File: {doc_info['file']}")
    print(f"📋 Description: {doc_info['description']}")
    print(f"❓ Questions: {len(questions)}")
    
    test_request = {
        "documents": doc_info['file'],
        "questions": questions
    }
    
    headers = {
        "Authorization": "Bearer default-token",
        "Content-Type": "application/json"
    }
    
    try:
        print("⏳ Processing...")
        start_time = time.time()
        
        response = requests.post(ENDPOINT, json=test_request, headers=headers, timeout=120)
        duration = time.time() - start_time
        
        print(f"⏱️  Response time: {duration:.2f} seconds")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! {len(result['answers'])} answers received")
            
            # Analyze answer quality
            print(f"\n📝 ANSWER QUALITY ANALYSIS:")
            for i, (question, answer) in enumerate(zip(questions, result['answers']), 1):
                print(f"\n🔍 Q{i}: {question}")
                print(f"💬 A{i}: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                
                # Simple quality metrics
                quality_score = analyze_answer_quality(question, answer)
                print(f"⭐ Quality Score: {quality_score}/5")
                print("-" * 60)
            
            return {
                "success": True,
                "duration": duration,
                "answers": result['answers'],
                "doc_name": doc_info['name']
            }
        else:
            print(f"❌ Error: {response.text}")
            return {
                "success": False,
                "duration": duration,
                "error": response.text,
                "doc_name": doc_info['name']
            }
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {
            "success": False,
            "duration": 0,
            "error": str(e),
            "doc_name": doc_info['name']
        }

def analyze_answer_quality(question, answer):
    """Simple answer quality analysis"""
    score = 0
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    # Check if answer is not empty or error
    if len(answer.strip()) > 10 and "error" not in answer_lower:
        score += 1
    
    # Check for specific information
    if any(keyword in answer_lower for keyword in ["days", "months", "years", "period"]):
        score += 1
    
    # Check for numbers (important in insurance)
    import re
    if re.search(r'\d+', answer):
        score += 1
    
    # Check for relevant keywords based on question
    if "grace period" in question_lower and "grace" in answer_lower:
        score += 1
    elif "waiting period" in question_lower and "waiting" in answer_lower:
        score += 1
    elif "maternity" in question_lower and "maternity" in answer_lower:
        score += 1
    elif "sum insured" in question_lower and ("sum" in answer_lower or "amount" in answer_lower):
        score += 1
    elif "exclusion" in question_lower and "exclusion" in answer_lower:
        score += 1
    
    # Check for detailed answer (not too short)
    if len(answer) > 100:
        score += 1
    
    return min(score, 5)  # Cap at 5

def test_all_documents():
    """Test all documents with standard questions"""
    
    print("🚀 COMPREHENSIVE MULTI-DOCUMENT TEST")
    print("="*80)
    print(f"📄 Documents to test: {len(TEST_DOCUMENTS)}")
    print(f"❓ Questions per document: {len(STANDARD_QUESTIONS)}")
    print(f"🎯 Total tests: {len(TEST_DOCUMENTS) * len(STANDARD_QUESTIONS)}")
    
    results = []
    total_start_time = time.time()
    
    for doc_info in TEST_DOCUMENTS:
        result = test_single_document(doc_info, STANDARD_QUESTIONS)
        results.append(result)
        
        # Brief pause between documents
        time.sleep(2)
    
    total_duration = time.time() - total_start_time
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed tests: {len(failed_tests)}")
    print(f"⏱️  Total time: {total_duration:.2f} seconds")
    print(f"⏱️  Average time per document: {total_duration/len(TEST_DOCUMENTS):.2f} seconds")
    
    if successful_tests:
        avg_response_time = sum(r['duration'] for r in successful_tests) / len(successful_tests)
        print(f"⏱️  Average response time: {avg_response_time:.2f} seconds")
        
        print(f"\n📈 PERFORMANCE BY DOCUMENT:")
        for result in successful_tests:
            print(f"   {result['doc_name']}: {result['duration']:.2f}s")
    
    if failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for result in failed_tests:
            print(f"   {result['doc_name']}: {result['error'][:100]}...")
    
    # Overall assessment
    success_rate = len(successful_tests) / len(results) * 100
    if success_rate >= 80:
        print(f"\n🎉 EXCELLENT: {success_rate:.1f}% success rate!")
    elif success_rate >= 60:
        print(f"\n✅ GOOD: {success_rate:.1f}% success rate")
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT: {success_rate:.1f}% success rate")

def test_hackathon_simulation():
    """Simulate hackathon conditions with one document"""
    
    print(f"\n{'='*80}")
    print("🏆 HACKATHON SIMULATION TEST")
    print(f"{'='*80}")
    
    # Use the first document for hackathon simulation
    doc_info = TEST_DOCUMENTS[0]
    
    # Full hackathon questions
    hackathon_questions = [
        "What is the grace period for premium payment under this policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges?"
    ]
    
    result = test_single_document(doc_info, hackathon_questions)
    
    if result['success']:
        if result['duration'] < 30:
            print(f"🎉 HACKATHON READY: {result['duration']:.2f}s (Under 30s limit!)")
        else:
            print(f"⚠️  OPTIMIZATION NEEDED: {result['duration']:.2f}s (Over 30s limit)")
    else:
        print("❌ HACKATHON SIMULATION FAILED")

if __name__ == "__main__":
    test_all_documents()
    test_hackathon_simulation()