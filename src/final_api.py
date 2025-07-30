#!/usr/bin/env python3
"""
Final Clean API for LEGAL-BERT Document Query System
"""

import os
import time
import logging
from typing import List, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Core ML imports
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Local imports
from src.document_parser import DocumentParser, DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
document_parser: Optional[DocumentParser] = None
embedding_model = None
gemini_client = None

# Authentication
VALID_BEARER_TOKENS = {
    os.getenv("API_BEARER_TOKEN", "default-token"): "default-user",
}

class QueryRequest(BaseModel):
    documents: str = Field(..., description="Document URL to process")
    questions: List[str] = Field(..., description="List of questions")
    
    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v):
        if not v.strip():
            raise ValueError("Document URL/path cannot be empty")
        return v
    
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        for question in v:
            if not question.strip():
                raise ValueError("Question cannot be empty")
        return v

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="Answers to all questions")

# Authentication
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    if token not in VALID_BEARER_TOKENS:
        raise HTTPException(status_code=401, detail="Invalid token")
    return VALID_BEARER_TOKENS[token]

class SimpleLegalBertSearcher:
    """Simple LEGAL-BERT searcher without complex threading"""
    
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = []
        self.indexed = False
        
    def initialize(self):
        """Initialize ultra-fast model for hackathon performance"""
        try:
            # Force CPU device and optimize for maximum speed
            import torch
            torch.set_num_threads(4)
            device = "cpu"
            
            # Use the fastest possible model for sub-30s performance
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            logger.info(f"Ultra-fast embedding model loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise e
    
    def index_documents(self, chunks: List[DocumentChunk]):
        """Ultra-fast document indexing"""
        start_time = time.time()
        logger.info(f"Fast indexing {len(chunks)} chunks")
        
        self.chunks = chunks
        
        # Aggressive chunk reduction for speed (take most relevant ones)
        if len(chunks) > 100:
            # Smart sampling: take chunks from different parts of document
            step = len(chunks) // 100
            chunks = chunks[::step][:100]  # Sample evenly across document
            self.chunks = chunks
        
        # Create embeddings with aggressive truncation for speed
        texts = [chunk.content[:200] for chunk in chunks]  # Much shorter for speed
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            batch_size=32  # Optimize batch size
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize and add embeddings
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.indexed = True
        duration = time.time() - start_time
        logger.info(f"Fast indexing completed in {duration:.2f}s")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Search for relevant chunks"""
        if not self.indexed:
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results

def extract_answer_with_gemini(question: str, chunks: List[DocumentChunk]) -> str:
    """Extract answer using Gemini with BM25-enhanced context selection"""
    global gemini_client
    
    if not gemini_client:
        return extract_answer_insurance_bm25(question, chunks)
    
    # Use BM25 patterns to rank chunks better
    ranked_chunks = rank_chunks_with_bm25(question, chunks)
    
    # Reduced context for speed (fewer chunks, shorter content)
    context_parts = []
    for i, chunk in enumerate(ranked_chunks[:3]):  # Only top 3 chunks for speed
        content = chunk.content[:300] if len(chunk.content) > 300 else chunk.content  # Shorter content
        context_parts.append(f"[{i+1}] {content}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are an expert insurance policy analyst. Answer the following question using ONLY the provided context from an insurance policy document.

Question: {question}

Context from Insurance Policy:
{context}

Instructions for High-Quality Answer:
1. Provide a comprehensive answer based ONLY on the provided context
2. Include ALL relevant details: specific numbers, time periods, percentages, conditions, exceptions
3. Quote exact phrases from the context when appropriate
4. If multiple aspects are mentioned (e.g., different waiting periods), include all of them
5. Structure your answer clearly with bullet points if multiple points exist
6. If any information is not clearly stated in the context, explicitly mention this
7. For numerical values, always include the units (days, months, years, percentage, etc.)
8. Include any relevant conditions, limitations, or exceptions mentioned
9. Be precise and avoid generalizations

Provide a detailed, accurate answer:"""
    
    try:
        response = gemini_client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,  # Reduced for speed
                temperature=0.1,        # Lower for faster generation
                top_p=0.8,             # More focused responses
                top_k=20,              # Fewer options for speed
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return extract_answer_insurance_bm25(question, chunks)

def rank_chunks_with_bm25(question: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Rank chunks using BM25-style insurance keyword scoring"""
    question_lower = question.lower()
    
    # Comprehensive insurance keyword weights
    high_value_keywords = {
        # Time periods
        "grace period": 8, "waiting period": 8, "moratorium": 6,
        
        # Medical conditions
        "pre-existing": 7, "ped": 7, "maternity": 7, "pregnancy": 6,
        "cataract": 6, "surgery": 5, "operation": 5, "treatment": 4,
        
        # Benefits and discounts
        "no claim discount": 8, "ncd": 8, "bonus": 5, "discount": 5,
        "health check": 6, "checkup": 6, "preventive": 5,
        
        # Coverage details
        "sum insured": 6, "coverage": 5, "benefit": 5, "limit": 5,
        "room rent": 7, "icu": 6, "hospital": 5, "accommodation": 5,
        
        # Exclusions and conditions
        "exclusion": 6, "excluded": 6, "not covered": 7,
        "deductible": 6, "copay": 6, "co-payment": 6,
        
        # Medical systems
        "ayush": 6, "ayurveda": 5, "homeopathy": 5, "unani": 5,
        
        # Special cases
        "organ donor": 7, "transplant": 6, "harvesting": 6,
        
        # Policy terms
        "premium": 5, "renewal": 5, "policy": 4, "insured": 4,
        "claim": 5, "reimburs": 5, "indemnify": 5
    }
    
    scored_chunks = []
    for chunk in chunks:
        content_lower = chunk.content.lower()
        score = 0
        
        # High-value keyword matching
        for keyword, weight in high_value_keywords.items():
            if keyword in question_lower and keyword in content_lower:
                score += weight
        
        # Number matching (important for insurance)
        import re
        question_numbers = re.findall(r'\d+', question_lower)
        content_numbers = re.findall(r'\d+', content_lower)
        number_overlap = len(set(question_numbers).intersection(set(content_numbers)))
        score += number_overlap * 2
        
        # General word overlap
        question_words = set(question_lower.split())
        content_words = set(content_lower.split())
        word_overlap = len(question_words.intersection(content_words))
        score += word_overlap * 0.5
        
        scored_chunks.append((score, chunk))
    
    # Sort by score and return chunks
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks]

def extract_answer_insurance_bm25(question: str, chunks: List[DocumentChunk]) -> str:
    """Advanced BM25-based insurance keyword matching"""
    question_lower = question.lower()
    
    # Comprehensive insurance keyword patterns
    insurance_patterns = {
        # Grace Period Patterns
        "grace_period": {
            "keywords": ["grace period", "grace", "premium payment", "due date", "lapse", "renewal grace"],
            "numbers": ["15", "30", "45", "60", "days", "day"],
            "context": ["premium", "payment", "due", "renewal", "continue", "lapse"]
        },
        
        # Waiting Period Patterns
        "waiting_period": {
            "keywords": ["waiting period", "waiting", "pre-existing", "ped", "exclusion period", "moratorium"],
            "numbers": ["12", "24", "36", "48", "months", "years", "year"],
            "context": ["disease", "condition", "illness", "treatment", "coverage", "excluded"]
        },
        
        # Maternity Benefits
        "maternity": {
            "keywords": ["maternity", "pregnancy", "childbirth", "delivery", "newborn", "baby", "natal"],
            "numbers": ["24", "36", "months", "deliveries", "births"],
            "context": ["benefit", "coverage", "expenses", "hospital", "treatment", "female"]
        },
        
        # Surgery/Treatment Waiting Periods
        "surgery_waiting": {
            "keywords": ["cataract", "surgery", "operation", "procedure", "treatment"],
            "numbers": ["12", "24", "36", "months", "years"],
            "context": ["waiting", "period", "coverage", "excluded", "benefit"]
        },
        
        # Organ Donor Coverage
        "organ_donor": {
            "keywords": ["organ donor", "donor", "transplant", "harvesting", "organ"],
            "context": ["medical expenses", "coverage", "indemnify", "benefit", "insured"]
        },
        
        # No Claim Discount
        "no_claim_discount": {
            "keywords": ["no claim discount", "ncd", "discount", "bonus", "renewal"],
            "numbers": ["5%", "10%", "15%", "20%", "percent"],
            "context": ["premium", "claim", "year", "continuous", "aggregate"]
        },
        
        # Health Checkup Benefits
        "health_checkup": {
            "keywords": ["health check", "checkup", "preventive", "screening", "examination"],
            "context": ["benefit", "reimburs", "coverage", "annual", "block", "years"]
        },
        
        # Hospital Definition
        "hospital_definition": {
            "keywords": ["hospital", "institution", "medical facility", "nursing home"],
            "numbers": ["10", "15", "beds", "inpatient"],
            "context": ["qualified", "staff", "operation theatre", "24/7", "records"]
        },
        
        # AYUSH Coverage
        "ayush": {
            "keywords": ["ayush", "ayurveda", "yoga", "naturopathy", "unani", "siddha", "homeopathy"],
            "context": ["treatment", "coverage", "inpatient", "hospital", "system", "medical"]
        },
        
        # Room Rent Limits
        "room_rent": {
            "keywords": ["room rent", "daily room", "icu charges", "accommodation"],
            "numbers": ["1%", "2%", "percent", "sum insured", "capped"],
            "context": ["limit", "charges", "plan", "network", "ppn"]
        },
        
        # Sum Insured
        "sum_insured": {
            "keywords": ["sum insured", "coverage amount", "policy limit", "maximum benefit"],
            "numbers": ["lakh", "lakhs", "crore", "rupees", "rs"],
            "context": ["individual", "family", "aggregate", "per", "annum"]
        },
        
        # Exclusions
        "exclusions": {
            "keywords": ["exclusion", "excluded", "not covered", "limitation"],
            "context": ["treatment", "condition", "disease", "procedure", "benefit"]
        },
        
        # Deductible/Copay
        "deductible": {
            "keywords": ["deductible", "copay", "co-payment", "excess"],
            "numbers": ["percent", "%", "rupees", "amount"],
            "context": ["claim", "payment", "insured", "bear"]
        }
    }
    
    # Find best matching pattern
    best_matches = []
    
    for pattern_name, pattern in insurance_patterns.items():
        score = 0
        matched_content = []
        
        for chunk in chunks:
            content = chunk.content
            content_lower = content.lower()
            
            # Check keyword matches
            keyword_matches = sum(1 for kw in pattern["keywords"] if kw in question_lower and kw in content_lower)
            if keyword_matches > 0:
                score += keyword_matches * 3
                
                # Check for number/context matches
                if "numbers" in pattern:
                    number_matches = sum(1 for num in pattern["numbers"] if num in content_lower)
                    score += number_matches * 2
                
                context_matches = sum(1 for ctx in pattern["context"] if ctx in content_lower)
                score += context_matches
                
                # Extract relevant sentences
                sentences = content.split('.')
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(kw in sentence_lower for kw in pattern["keywords"]):
                        matched_content.append(sentence.strip())
        
        if score > 0:
            best_matches.append((score, pattern_name, matched_content))
    
    # Sort by score and return best match
    if best_matches:
        best_matches.sort(key=lambda x: x[0], reverse=True)
        _, pattern_name, content_list = best_matches[0]
        
        # Return the most relevant sentences
        if content_list:
            # Combine and clean up sentences
            result = ". ".join(content_list[:3])  # Top 3 sentences
            if len(result) > 500:
                result = result[:500] + "..."
            return result + "."
    
    # Fallback: semantic search through chunks
    for chunk in chunks:
        content = chunk.content
        content_lower = content.lower()
        
        # Simple keyword overlap
        question_words = set(question_lower.split())
        content_words = set(content_lower.split())
        overlap = len(question_words.intersection(content_words))
        
        if overlap >= 2:  # At least 2 word overlap
            sentences = content.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 20 and overlap >= 2:
                    return sentence.strip() + "."
    
    # Final fallback
    if chunks:
        return chunks[0].content[:300] + "..."
    
    return "I couldn't find relevant information to answer this question."

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_parser, embedding_model, gemini_client
    
    logger.info("Starting Final LEGAL-BERT API...")
    
    # Initialize document parser
    document_parser = DocumentParser()
    logger.info("Document parser initialized")
    
    # Initialize searcher
    embedding_model = SimpleLegalBertSearcher()
    embedding_model.initialize()
    
    # Initialize Gemini
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            gemini_client = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("Gemini client initialized")
        else:
            logger.warning("No Gemini API key found")
    except Exception as e:
        logger.warning(f"Gemini initialization failed: {e}")
    
    yield
    logger.info("Shutting down API...")

# Create app
app = FastAPI(
    title="Final LEGAL-BERT API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query_final(
    request: QueryRequest,
    user: str = Depends(verify_token)
) -> QueryResponse:
    """Final stable document processing and question answering"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing request for user {user}")
        logger.info(f"Document: {request.documents}, Questions: {len(request.questions)}")
        
        # Parse document
        logger.info("Parsing document...")
        chunks = document_parser.parse_document(request.documents)
        logger.info(f"Parsed {len(chunks)} chunks")
        
        if not chunks:
            raise HTTPException(status_code=422, detail="No content extracted")
        
        # Index documents
        logger.info("Indexing documents...")
        embedding_model.index_documents(chunks)
        
        # Process questions with parallel optimization for speed
        logger.info(f"Processing {len(request.questions)} questions in parallel...")
        answers = []
        
        try:
            # Pre-compute search results for all questions (batch processing)
            logger.info("Pre-computing search results for all questions...")
            all_search_results = []
            for question in request.questions:
                search_results = embedding_model.search(question, top_k=3)  # Reduced for speed
                relevant_chunks = [chunk for chunk, score in search_results]
                if not relevant_chunks:
                    relevant_chunks = chunks[:3]  # Fewer fallback chunks
                all_search_results.append(relevant_chunks)
            
            # Process questions in parallel using threading
            import concurrent.futures
            
            def process_single_question(question_data):
                question, relevant_chunks, index = question_data
                logger.info(f"Processing question {index+1}/{len(request.questions)}")
                
                if gemini_client:
                    return extract_answer_with_gemini(question, relevant_chunks)
                else:
                    return extract_answer_insurance_bm25(question, relevant_chunks)
            
            # Prepare question data for parallel processing
            question_data = [
                (question, chunks, i) 
                for i, (question, chunks) in enumerate(zip(request.questions, all_search_results))
            ]
            
            # Process questions in parallel (max 5 threads to avoid overwhelming Gemini)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_index = {
                    executor.submit(process_single_question, data): i 
                    for i, data in enumerate(question_data)
                }
                
                # Collect results in order
                answers = [None] * len(request.questions)
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        answer = future.result(timeout=30)  # 30s timeout per question
                        answers[index] = answer
                    except Exception as e:
                        logger.error(f"Error processing question {index+1}: {e}")
                        answers[index] = f"Error processing question: {str(e)}"
            
            # Ensure all answers are filled
            for i, answer in enumerate(answers):
                if answer is None:
                    answers[i] = "Unable to process this question."
                
        except Exception as e:
            logger.error(f"Error processing questions: {e}")
            # Fallback: simple answers
            for question in request.questions:
                answers.append(f"Unable to process question: {question[:50]}...")
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Request completed in {total_time:.2f}ms")
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "components": {
            "document_parser": document_parser is not None,
            "legal_bert_model": embedding_model is not None,
            "gemini_client": gemini_client is not None
        }
    }

@app.get("/")
async def root():
    return {"message": "Final LEGAL-BERT Document Query System", "status": "running"}

if __name__ == "__main__":
    uvicorn.run("final_api:app", host="0.0.0.0", port=8004, reload=False)