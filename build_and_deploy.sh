#!/usr/bin/env bash
# Build the container, push to Artifact Registry, deploy to Cloud Run.

PROJECT_ID=ggoheatmap          #  ← your project
REGION=us-central1
REPO=my-repo
IMAGE=owl-habitat
SERVICE=owl-habitat

set -e

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# create repo if missing
gcloud artifacts repositories describe $REPO --location=$REGION >/dev/null 2>&1 \
  || gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION \
       --description="Docker repo"

# build & push
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:latest"

# deploy
gcloud run deploy "$SERVICE" \
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:latest" \
  --region "$REGION" --platform managed \
  --allow-unauthenticated --memory 2Gi --timeout 15m
