# HackRx 6.0 Requirements Summary

## Key API Requirements

### 1. Endpoint Structure
- **Required Endpoint**: `/hackrx/run` (NOT `/api/v1/hackrx/run`)
- **Method**: POST
- **Authentication**: Bearer token in Authorization header
- **Content-Type**: application/json
- **Accept**: application/json

### 2. Request Format
```json
{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
    "questions": [
        "Question 1",
        "Question 2",
        "..."
    ]
}
```

**Important Notes:**
- `documents` is a **single string** (not array)
- `questions` is an array of strings
- Maximum response time: 30 seconds

### 3. Response Format
```json
{
    "answers": [
        "Answer to question 1",
        "Answer to question 2",
        "..."
    ]
}
```

**Important Notes:**
- Response must be a simple object with `answers` array
- Each answer should be a string
- Order of answers must match order of questions

### 4. Deployment Requirements
- **Public URL**: Must be publicly accessible
- **HTTPS**: SSL certificate required
- **Uptime**: Must be available during evaluation
- **Performance**: Response time < 30 seconds

### 5. Authentication
- Bearer token authentication required
- Format: `Authorization: Bearer <api_key>`
- Token validation must be implemented

## Current Implementation Status

### ✅ Fixed Issues
1. Updated Pydantic validators from V1 to V2 style (`@field_validator`)
2. Changed endpoint from `/api/v1/hackrx/run` to `/hackrx/run`
3. Updated request model: `documents` is now string (not array)
4. Simplified response model to match hackathon format
5. Updated processing logic for single document

### 🔧 Current Setup
- API running on `http://localhost:8000`
- Bearer token: Set in `.env` file as `API_BEARER_TOKEN`
- Test endpoint: `python test_hackathon_endpoint.py`

### 📋 Pre-Submission Checklist
- [ ] API is publicly accessible via HTTPS
- [ ] `/hackrx/run` endpoint responds correctly
- [ ] Bearer token authentication works
- [ ] Response format matches exactly: `{"answers": ["...", "..."]}`
- [ ] Response time < 30 seconds
- [ ] SSL certificate configured
- [ ] Test with provided sample data

### 🚀 Deployment Options
1. **Heroku**: Easy deployment with SSL
2. **Railway**: Simple Python deployment
3. **Render**: Free tier with SSL
4. **Vercel**: Serverless functions
5. **AWS/GCP/Azure**: Cloud platforms
6. **DigitalOcean**: VPS with SSL setup

### 📝 Sample Test Request
```bash
curl -X POST "https://your-domain.com/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
    "questions": [
      "What is the grace period for premium payment?",
      "What is the waiting period for pre-existing diseases?"
    ]
  }'
```

### Expected Response
```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment...",
    "There is a waiting period of thirty-six (36) months..."
  ]
}
```