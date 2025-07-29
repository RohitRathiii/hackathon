#!/usr/bin/env python3
"""
Production API Runner for LEGAL-BERT Document Query System
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Run the optimized LEGAL-BERT API"""
    
    # Verify required environment variables
    required_vars = ["GOOGLE_API_KEY", "API_BEARER_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        sys.exit(1)
    
    print("🚀 Starting LEGAL-BERT Document Query System...")
    print("📋 Features:")
    print("   • LEGAL-BERT embeddings for legal/insurance documents")
    print("   • Optimized FAISS vector search with caching")
    print("   • Multi-agent RAG workflow")
    print("   • Enhanced BM25 keyword search")
    print("   • Parallel question processing")
    print("   • Bearer token authentication")
    print()
    print("🌐 API will be available at: http://localhost:8004")
    print("📖 API Documentation: http://localhost:8004/docs")
    print("🔧 Health Check: http://localhost:8004/health")
    print()
    
    # Run the API
    port = int(os.environ.get("PORT", 8004))  # Heroku provides PORT env var
    uvicorn.run(
        "src.final_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )

if __name__ == "__main__":
    main()