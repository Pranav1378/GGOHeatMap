#!/usr/bin/env bash
# Build the container, push to Artifact Registry, deploy to Cloud Run.

set -e

# Configuration
PROJECT_ID="ggoheatmap" # Pass as first argument or set here
REGION="us-central1"
SERVICE_NAME="owl-habitat"
IMAGE_NAME="owl-habitat-app"

echo "🦉 Deploying Great Gray Owl Habitat App to Google Cloud Run"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"

# Set project
gcloud config set project "$PROJECT_ID"

# Enable required services
echo "▸ Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo "▸ Setting up Artifact Registry..."
REPO_NAME="owl-habitat-repo"
gcloud artifacts repositories describe $REPO_NAME \
    --location=$REGION 2>/dev/null || \
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Owl habitat app container registry"

# Build and push image
echo "▸ Building and pushing container..."
IMAGE_URL="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"

gcloud builds submit --tag "$IMAGE_URL" .

# Deploy to Cloud Run
echo "▸ Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image="$IMAGE_URL" \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=1 \
    --timeout=900 \
    --max-instances=10 \
    --set-env-vars="PYTHONUNBUFFERED=1"

echo "✅ Deployment complete!"
echo "🌐 Your app will be available at:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"
