#!/usr/bin/env bash
set -e

# Configuration
PROJECT_ID="ggoheatmap"
BUCKET_NAME="gs://${PROJECT_ID}-owl-training"
JOB_NAME="owl-habitat-training-$(date +%Y%m%d-%H%M%S)"
REGION="us-central1"

echo "🦉 Training Great Gray Owl model on Google Cloud AI Platform"
echo "Project: $PROJECT_ID"
echo "Bucket: $BUCKET_NAME"
echo "Job: $JOB_NAME"

# Set project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "▸ Enabling required APIs..."
gcloud services enable ml.googleapis.com
gcloud services enable storage.googleapis.com

# Create storage bucket
echo "▸ Creating Cloud Storage bucket..."
gsutil mb -l $REGION $BUCKET_NAME 2>/dev/null || echo "Bucket already exists"

# Create training package
echo "▸ Preparing training package..."
mkdir -p trainer
cp train.py trainer/task.py

# Create setup.py for the training package
cat > trainer/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name='owl-habitat-trainer',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas', 
        'scikit-learn',
        'matplotlib',
        'joblib',
        'requests'
    ]
)
EOF

# Create __init__.py
touch trainer/__init__.py

# Create config.yaml for AI Platform
cat > config.yaml << EOF
trainingInput:
  scaleTier: BASIC
  runtimeVersion: '2.11'
  pythonVersion: '3.7'
  region: $REGION
  jobDir: $BUCKET_NAME/job-dir
EOF

# Upload training package
echo "▸ Uploading training code..."
gsutil -m cp -r trainer $BUCKET_NAME/

# Submit training job
echo "▸ Submitting training job..."
gcloud ai-platform jobs submit training $JOB_NAME \
    --config=config.yaml \
    --module-name=trainer.task \
    --package-path=trainer \
    --staging-bucket=$BUCKET_NAME

echo "▸ Monitoring training job..."
gcloud ai-platform jobs describe $JOB_NAME

# Wait for job completion
echo "▸ Waiting for job to complete..."
while true; do
    STATUS=$(gcloud ai-platform jobs describe $JOB_NAME --format="value(state)")
    echo "Job status: $STATUS"
    
    if [ "$STATUS" = "SUCCEEDED" ]; then
        echo "✅ Training job completed successfully!"
        break
    elif [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "CANCELLED" ]; then
        echo "❌ Training job failed or was cancelled"
        exit 1
    fi
    
    sleep 30
done

# Download results
echo "▸ Downloading trained model..."
gsutil -m cp -r $BUCKET_NAME/job-dir/model.pkl .
gsutil -m cp -r $BUCKET_NAME/job-dir/grid.csv .
gsutil -m cp -r $BUCKET_NAME/job-dir/obs.csv .

# Clean up
echo "▸ Cleaning up..."
rm -rf trainer config.yaml

echo "✅ Cloud AI Platform training complete!"
echo "📁 Model artifacts downloaded:"
echo "   - model.pkl"
echo "   - grid.csv"
echo "   - obs.csv" 