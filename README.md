# LEGAL-BERT Document Query System

A high-performance, multi-agent RAG system specifically optimized for legal and insurance document analysis using LEGAL-BERT embeddings.

## 🚀 Features

- **LEGAL-BERT Embeddings**: Domain-specific embeddings trained on legal documents
- **Optimized FAISS Search**: Fast local vector search with intelligent caching
- **Multi-Agent Workflow**: Planner → Retrieval → Judge → Synthesis pipeline
- **Enhanced BM25**: Legal term-aware keyword search with query expansion
- **Parallel Processing**: Concurrent question processing for maximum speed
- **Smart Caching**: Embedding cache for faster subsequent runs
- **Bearer Token Auth**: Secure API access with token authentication

## 📋 System Requirements

- Python 3.8+
- 8GB+ RAM (recommended for LEGAL-BERT)
- macOS/Linux/Windows

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd legal-bert-document-query
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_google_api_key_here

# Required: API Bearer Token for authentication
API_BEARER_TOKEN=your_secure_token_here

# Optional: System Configuration
MAX_MEMORY_GB=8
LOG_LEVEL=INFO
```

## 🚀 Quick Start

1. **Start the API**
   ```bash
   python run_api.py
   ```

2. **Test the API**
   ```bash
   python test_hackathon_endpoint.py
   ```

3. **Access API Documentation**
   - API Docs: http://localhost:8004/docs
   - Health Check: http://localhost:8004/health

## 📡 API Usage

### Endpoint: `/hackrx/run`

**Request:**
```json
{
  "documents": "path/to/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?",
    "Does this policy cover maternity expenses?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "Based on the provided context, the grace period for premium payment is 15 days...",
    "The provided context mentions 'Pre-Existing Disease' but does not specify...",
    "Based on the provided context, the policy covers maternity expenses..."
  ]
}
```

### Authentication

Include Bearer token in headers:
```bash
curl -X POST "http://localhost:8004/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token_here" \
  -d '{"documents": "document.pdf", "questions": ["Your question?"]}'
```

## 🏗️ Architecture

```
Document Input
     ↓
Document Parser (PyMuPDF)
     ↓
LEGAL-BERT Embeddings → FAISS Index
     ↓                      ↓
Enhanced BM25 Index ← Hybrid Search
     ↓
Multi-Agent Workflow:
├── Planner Agent (Query Decomposition)
├── Retrieval Agent (Hybrid Search)
├── Judge Agent (Relevance Filtering)
└── Synthesis Agent (Answer Generation)
     ↓
Final Answer
```

## 📊 Performance

- **Total Processing Time**: ~20 seconds (first run)
- **Cached Runs**: ~10 seconds (with embedding cache)
- **Indexing**: ~1 second (cached)
- **Question Processing**: ~7 seconds (3 questions parallel)

## 📁 Project Structure

```
├── src/
│   ├── ultra_fast_api.py           # Main API application
│   ├── ultra_fast_agents.py        # Multi-agent workflow
│   ├── ultra_fast_embedding_search.py  # LEGAL-BERT + FAISS search
│   ├── document_parser.py          # Document processing
│   └── logic_evaluation.py         # Logic evaluation (optional)
├── testdata/                       # Sample PDF files
├── .env                           # Environment configuration
├── requirements.txt               # Python dependencies
├── run_api.py                    # Production runner
├── test_hackathon_endpoint.py    # API test script
└── README.md                     # This file
```

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_hackathon_endpoint.py
```

Expected output:
```
✅ Success! Got 3 answers
Answer 1: Based on the provided context, the grace period for premium payment is 15 days...
Answer 2: The provided context mentions 'Pre-Existing Disease'...
Answer 3: Based on the provided context, the policy covers maternity expenses...
```

## 🔧 Troubleshooting

### Common Issues

1. **LEGAL-BERT Download**: First run downloads ~440MB model
2. **Memory Issues**: Reduce batch size in `ultra_fast_embedding_search.py`
3. **API Key Errors**: Verify `GOOGLE_API_KEY` in `.env`
4. **Port Conflicts**: Change port in `run_api.py` if 8004 is occupied

### Performance Optimization

- **Embedding Cache**: Automatically created in `.embedding_cache/`
- **Batch Size**: Adjust in embedding manager for your hardware
- **Parallel Workers**: Modify in retrieval agent based on CPU cores

## 📈 Deployment

For production deployment:

1. **Set environment variables** in your deployment platform
2. **Use HTTPS** for secure API access
3. **Configure load balancing** for high availability
4. **Monitor memory usage** for LEGAL-BERT processing
5. **Set up logging** for debugging and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **LEGAL-BERT**: nlpaueb/legal-bert-base-uncased
- **FAISS**: Facebook AI Similarity Search
- **FastAPI**: Modern web framework for APIs
- **Google Gemini**: LLM for text generation