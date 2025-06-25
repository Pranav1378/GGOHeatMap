#!/usr/bin/env bash
set -e

# Configuration
PROJECT_ID="ggoheatmap"
ZONE="us-central1-a"
INSTANCE_NAME="owl-training-vm"
MACHINE_TYPE="e2-standard-2"  # 2 vCPU, 8GB RAM
IMAGE_FAMILY="ubuntu-2004-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

echo "🦉 Setting up Great Gray Owl model training on Google Cloud Compute"
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Instance: $INSTANCE_NAME"

# Set project
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "▸ Enabling Compute Engine API..."
gcloud services enable compute.googleapis.com

# Create the VM instance
echo "▸ Creating Compute Engine instance..."
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=20GB \
    --boot-disk-type=pd-standard \
    --metadata=startup-script='#!/bin/bash
        # Update system
        apt-get update
        apt-get install -y python3 python3-pip git
        
        # Install Python packages
        pip3 install numpy pandas scikit-learn matplotlib joblib requests
        
        # Create working directory
        mkdir -p /home/training
        cd /home/training
        
        echo "VM setup complete - ready for training!"
    ' \
    --tags=training-vm \
    --scopes=https://www.googleapis.com/auth/cloud-platform

echo "▸ Waiting for VM to be ready..."
sleep 30

# Copy training files to the VM
echo "▸ Copying training files to VM..."
gcloud compute scp train.py $INSTANCE_NAME:/home/training/ --zone=$ZONE
gcloud compute scp requirements.txt $INSTANCE_NAME:/home/training/ --zone=$ZONE

# Run training on the VM
echo "▸ Starting training on VM..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    cd /home/training
    echo '🦉 Starting Great Gray Owl model training...'
    python3 train.py
    echo '✅ Training complete!'
    ls -la *.pkl *.csv
"

# Copy results back
echo "▸ Copying trained model back..."
gcloud compute scp $INSTANCE_NAME:/home/training/model.pkl . --zone=$ZONE
gcloud compute scp $INSTANCE_NAME:/home/training/grid.csv . --zone=$ZONE
gcloud compute scp $INSTANCE_NAME:/home/training/obs.csv . --zone=$ZONE

echo "▸ Cleaning up VM..."
gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet

echo "✅ Cloud training complete!"
echo "📁 Model artifacts downloaded:"
echo "   - model.pkl"
echo "   - grid.csv" 
echo "   - obs.csv"
echo ""
echo "🚀 Ready to deploy with: ./build_and_deploy.sh" 