#!/bin/bash

# Google Cloud Run Deployment Script
echo "🚀 Deploying LEGAL-BERT API to Google Cloud Run..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please run:"
    echo "   brew install --cask google-cloud-sdk"
    echo "   source ~/.zshrc"
    exit 1
fi

# Auto-generate project ID with timestamp
# Auto-generate project ID
PROJECT_ID="peppy-webbing-466418-k1"
SERVICE_NAME="legal-bert-api"
REGION="us-central1"

echo "📋 Project: $PROJECT_ID"
echo "🌍 Region: $REGION"
echo "🔧 Service: $SERVICE_NAME"

# Login and setup
echo "🔐 Please login to Google Cloud..."
gcloud auth login

# Create project
echo "📋 Creating project..."
gcloud projects create $PROJECT_ID --name="Legal BERT Hackathon API"

# Set project
gcloud config set project $PROJECT_ID

# Enable billing (you'll need to do this manually in console)
echo "💳 Please enable billing for project $PROJECT_ID in the Google Cloud Console"
echo "   Visit: https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
read -p "Press Enter after enabling billing..."

# Enable APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# Deploy
echo "🚀 Deploying to Cloud Run (CPU-optimized)..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars GOOGLE_API_KEY="AIzaSyA3poDdpwsi-rrOnURaLE1Gd6v-uRZeH_Y",API_BEARER_TOKEN="default-token",TORCH_DEVICE="cpu" \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 5

echo "✅ Deployment complete!"
echo "🌐 Your HTTPS API URL:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"
echo "📋 Add '/hackrx/run' to the URL above for hackathon submission"