#!/usr/bin/env bash
# One-shot script: build container, push to Artifact Registry, deploy to Cloud Run.

PROJECT_ID=<ggoheatmap>
REGION=us-central1
IMAGE_NAME=owl-habitat
SERVICE=owl-habitat

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# Create a repo if it doesn't exist
gcloud artifacts repositories describe my-repo --location=$REGION 2>/dev/null \
  || gcloud artifacts repositories create my-repo --repository-format=docker \
       --location=$REGION --description="Docker repo"

# Build & push
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/my-repo/$IMAGE_NAME:latest"

# Deploy
gcloud run deploy "$SERVICE" \
  --image="$REGION-docker.pkg.dev/$PROJECT_ID/my-repo/$IMAGE_NAME:latest" \
  --platform=managed --region="$REGION" \
  --allow-unauthenticated --memory=2Gi --timeout=15m
