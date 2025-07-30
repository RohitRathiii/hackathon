#!/usr/bin/env python3
"""
Test the improved BM25 insurance keyword matching locally
"""

import sys
import os
sys.path.append('src')

from final_api import extract_answer_insurance_bm25
from document_parser import DocumentChunk

def test_bm25_patterns():
    """Test BM25 patterns with sample insurance content"""
    
    # Sample insurance content chunks
    sample_chunks = [
        DocumentChunk(
            content="A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
            page_number=1,
            chunk_id="chunk_1"
        ),
        DocumentChunk(
            content="There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
            page_number=2,
            chunk_id="chunk_2"
        ),
        DocumentChunk(
            content="Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months.",
            page_number=3,
            chunk_id="chunk_3"
        ),
        DocumentChunk(
            content="The policy has a specific waiting period of two (2) years for cataract surgery.",
            page_number=4,
            chunk_id="chunk_4"
        ),
        DocumentChunk(
            content="A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year.",
            page_number=5,
            chunk_id="chunk_5"
        )
    ]
    
    # Test questions
    test_questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "What is the No Claim Discount (NCD) offered in this policy?"
    ]
    
    print("🧪 Testing BM25 Insurance Keyword Matching")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Question {i}: {question}")
        answer = extract_answer_insurance_bm25(question, sample_chunks)
        print(f"💬 Answer: {answer}")
        print("-" * 40)

if __name__ == "__main__":
    test_bm25_patterns()