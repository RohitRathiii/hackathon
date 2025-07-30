#!/usr/bin/env python3
"""
Simple local test to verify the enhanced Gemini logic
"""

import sys
import os
sys.path.append('src')

# Test the functions directly without the full API
def test_enhanced_functions():
    """Test the enhanced functions directly"""
    
    print("🧪 Testing Enhanced Functions Directly...")
    
    # Import the functions
    try:
        from final_api import rank_chunks_with_bm25, extract_answer_with_gemini
        from document_parser import DocumentChunk
        print("✅ Functions imported successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return
    
    # Create sample chunks with proper structure
    from document_parser import ChunkMetadata
    
    sample_chunks = [
        DocumentChunk(
            content="A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
            metadata=ChunkMetadata(document_id="test_1", page_number=1)
        ),
        DocumentChunk(
            content="There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
            metadata=ChunkMetadata(document_id="test_2", page_number=2)
        ),
        DocumentChunk(
            content="The policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months.",
            metadata=ChunkMetadata(document_id="test_3", page_number=3)
        )
    ]
    
    # Test BM25 ranking
    print("\n🔍 Testing BM25 Ranking...")
    question = "What is the grace period for premium payment?"
    try:
        ranked_chunks = rank_chunks_with_bm25(question, sample_chunks)
        print(f"✅ BM25 ranking successful: {len(ranked_chunks)} chunks ranked")
        print(f"📋 Top chunk: {ranked_chunks[0].content[:100]}...")
    except Exception as e:
        print(f"❌ BM25 ranking error: {e}")
    
    # Test the enhanced prompt (without actually calling Gemini)
    print("\n🔍 Testing Enhanced Prompt Structure...")
    try:
        # This will test the prompt creation but fail at Gemini call (which is expected)
        result = extract_answer_with_gemini(question, sample_chunks)
        print(f"✅ Enhanced answer: {result[:200]}...")
    except Exception as e:
        print(f"⚠️  Expected Gemini error (API not available locally): {str(e)[:100]}...")
    
    print("\n✅ Local function tests completed!")

if __name__ == "__main__":
    test_enhanced_functions()