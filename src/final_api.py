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
        """Initialize LEGAL-BERT model"""
        try:
            self.model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")
            logger.info("LEGAL-BERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load LEGAL-BERT: {e}")
            # Fallback to a smaller model
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Using fallback model: all-MiniLM-L6-v2")
    
    def index_documents(self, chunks: List[DocumentChunk]):
        """Index documents for search"""
        if self.indexed and len(self.chunks) == len(chunks):
            logger.info("Documents already indexed")
            return
            
        start_time = time.time()
        logger.info(f"Indexing {len(chunks)} chunks with LEGAL-BERT")
        
        self.chunks = chunks
        
        # Create embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize and add embeddings
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.indexed = True
        duration = time.time() - start_time
        logger.info(f"Indexing completed in {duration:.2f}s")
    
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
    """Extract answer using Gemini with context from chunks"""
    global gemini_client
    
    if not gemini_client:
        return extract_answer_simple(question, chunks)
    
    # Prepare context from top chunks
    context_parts = []
    for i, chunk in enumerate(chunks[:5]):
        content = chunk.content[:400] + "..." if len(chunk.content) > 400 else chunk.content
        context_parts.append(f"[{i+1}] {content}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Based on the following context from an insurance policy document, provide a clear and accurate answer to the question.

Question: {question}

Context:
{context}

Instructions:
1. Answer based ONLY on the provided context
2. Be specific and include relevant details like numbers, time periods, conditions
3. If the information is not clearly stated, say so
4. Keep the answer concise but complete

Answer:"""
    
    try:
        response = gemini_client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=250,
                temperature=0.3,
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return extract_answer_simple(question, chunks)

def extract_answer_simple(question: str, chunks: List[DocumentChunk]) -> str:
    """Simple pattern-based answer extraction"""
    question_lower = question.lower()
    
    # Search for relevant information
    for chunk in chunks:
        content = chunk.content
        content_lower = content.lower()
        
        # Grace period questions
        if "grace period" in question_lower and "premium" in question_lower:
            if "grace period" in content_lower and ("15" in content or "30" in content):
                sentences = content.split('.')
                for sentence in sentences:
                    if "grace period" in sentence.lower() and any(num in sentence for num in ["15", "30"]):
                        return sentence.strip() + "."
        
        # Waiting period questions
        elif "waiting period" in question_lower and ("pre-existing" in question_lower or "ped" in question_lower):
            if "waiting period" in content_lower and ("36" in content or "months" in content):
                sentences = content.split('.')
                for sentence in sentences:
                    if "waiting period" in sentence.lower() and ("pre-existing" in sentence.lower() or "36" in sentence):
                        return sentence.strip() + "."
        
        # Maternity questions
        elif "maternity" in question_lower:
            if "maternity" in content_lower:
                sentences = content.split('.')
                for sentence in sentences:
                    if "maternity" in sentence.lower():
                        return sentence.strip() + "."
    
    # Fallback
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
        
        # Process questions
        logger.info(f"Processing {len(request.questions)} questions...")
        answers = []
        
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}: {question[:50]}...")
            
            try:
                # Search for relevant chunks
                search_results = embedding_model.search(question, top_k=10)
                relevant_chunks = [chunk for chunk, score in search_results]
                
                # If no search results, use first chunks
                if not relevant_chunks:
                    relevant_chunks = chunks[:10]
                
                # Extract answer
                if gemini_client:
                    answer = extract_answer_with_gemini(question, relevant_chunks)
                else:
                    answer = extract_answer_simple(question, relevant_chunks)
                
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                answers.append(f"Error processing question: {str(e)}")
        
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