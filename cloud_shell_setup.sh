#!/usr/bin/env bash
# Run this script in Google Cloud Shell

echo "🦉 Setting up Great Gray Owl training in Google Cloud Shell"

# Set your project
export PROJECT_ID="ggoheatmap"
gcloud config set project $PROJECT_ID

# Install Python packages (if not already installed)
echo "▸ Installing Python packages..."
pip3 install --user numpy pandas scikit-learn matplotlib joblib requests

# Clone or upload your project
echo "▸ Setting up project..."
# If you have the code in git:
# git clone <your-repo-url>
# cd GgoHeatMap

# Or if you upload files manually, just run:
echo "Upload your train.py file to Cloud Shell, then run:"
echo "  python3 train.py"

echo "✅ Cloud Shell setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Upload your train.py file using the Cloud Shell file menu"
echo "2. Run: python3 train.py"
echo "3. Run: ./build_and_deploy.sh to deploy your app" 