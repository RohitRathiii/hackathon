#!/bin/bash

echo "🚀 Deploying LEGAL-BERT API to Railway..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    
    # Install Railway CLI
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install railway
    else
        # Linux
        curl -fsSL https://railway.app/install.sh | sh
    fi
fi

# Login to Railway (if not already logged in)
echo "🔐 Logging into Railway..."
railway login --browserless

# Create new Railway project
echo "📋 Creating Railway project..."
railway init

# Set environment variables
echo "🔧 Setting environment variables..."
railway variables set GOOGLE_API_KEY="AIzaSyA3poDdpwsi-rrOnURaLE1Gd6v-uRZeH_Y"
railway variables set API_BEARER_TOKEN="default-token"
railway variables set TORCH_DEVICE="cpu"
railway variables set PYTHONPATH="/app/src"

# Deploy to Railway with lightweight requirements
echo "🚀 Deploying to Railway with optimized build..."
cp requirements_railway.txt requirements.txt
railway up

# Get the deployment URL
echo "✅ Deployment complete!"
echo "🌐 Your Railway URL:"
railway status
echo "📋 Add '/hackrx/run' to the URL above for hackathon submission"

echo ""
echo "🧪 Test your deployment:"
echo "curl -X GET \"https://your-app.railway.app/health\""