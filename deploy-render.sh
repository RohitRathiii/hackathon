#!/bin/bash

echo "🚀 Deploying LEGAL-BERT API to Render..."

# Check if we have a git repository
if [ ! -d ".git" ]; then
    echo "📋 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit for Render deployment"
fi

# Create optimized requirements for Render
echo "🔧 Creating optimized requirements for Render..."
cp requirements_render.txt requirements.txt

# Add changes to git
git add .
git commit -m "Optimize for Render deployment" || echo "No changes to commit"

echo "📋 Render Deployment Instructions:"
echo "=================================="
echo ""
echo "1. Go to https://render.com and sign up/login"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub repository"
echo "4. Configure the service:"
echo "   - Name: legal-bert-api"
echo "   - Environment: Python 3"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: python run_api.py"
echo "   - Plan: Starter (Free)"
echo ""
echo "5. Add Environment Variables:"
echo "   GOOGLE_API_KEY=AIzaSyA3poDdpwsi-rrOnURaLE1Gd6v-uRZeH_Y"
echo "   API_BEARER_TOKEN=default-token"
echo "   TORCH_DEVICE=cpu"
echo "   PYTHONPATH=/opt/render/project/src"
echo ""
echo "6. Deploy and wait for build to complete"
echo ""
echo "🌐 Your Render URL will be: https://legal-bert-api-[random].onrender.com"
echo "🎯 Hackathon endpoint: https://legal-bert-api-[random].onrender.com/hackrx/run"
echo ""
echo "📋 Alternative: Use render.yaml for Infrastructure as Code"
echo "   - Upload render.yaml to your repository"
echo "   - Render will auto-detect and deploy"

# Push to GitHub if remote exists
if git remote get-url origin 2>/dev/null; then
    echo ""
    echo "📤 Pushing to GitHub..."
    git push origin main || git push origin master || echo "Push failed - please push manually"
    echo "✅ Ready for Render deployment!"
else
    echo ""
    echo "⚠️  No GitHub remote found. Please:"
    echo "   1. Create a GitHub repository"
    echo "   2. Add remote: git remote add origin <your-repo-url>"
    echo "   3. Push: git push -u origin main"
    echo "   4. Then deploy on Render"
fi