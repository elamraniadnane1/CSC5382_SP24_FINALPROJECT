import os
import subprocess

# Define variables
PROJECT_ID = 'your-gcp-project-id'  # Replace with your GCP project ID
IMAGE_NAME = 'fastapi-bert'
TAG = 'v1'
REGION = 'us-central1'  # Replace with your preferred region

# Build the Docker image
subprocess.run(['docker', 'build', '-t', f'gcr.io/{PROJECT_ID}/{IMAGE_NAME}:{TAG}', '.'], check=True)

# Authenticate with GCP
subprocess.run(['gcloud', 'auth', 'configure-docker'], check=True)

# Push the Docker image to Google Container Registry
subprocess.run(['docker', 'push', f'gcr.io/{PROJECT_ID}/{IMAGE_NAME}:{TAG}'], check=True)

# Deploy the model to Vertex AI
subprocess.run([
    'gcloud', 'ai', 'models', 'upload',
    '--region', REGION,
    '--display-name', IMAGE_NAME,
    '--container-image-uri', f'gcr.io/{PROJECT_ID}/{IMAGE_NAME}:{TAG}',
    '--container-port', '8000'
], check=True)

# Create an endpoint
subprocess.run([
    'gcloud', 'ai', 'endpoints', 'create',
    '--region', REGION,
    '--display-name', f'{IMAGE_NAME}-endpoint'
], check=True)

# Deploy the model to the endpoint
subprocess.run([
    'gcloud', 'ai', 'endpoints', 'deploy-model',
    '--region', REGION,
    '--model', IMAGE_NAME,
    '--display-name', f'{IMAGE_NAME}-deployment',
    '--machine-type', 'n1-standard-2',  # Choose an appropriate machine type
    '--endpoint', f'{IMAGE_NAME}-endpoint',
    '--traffic-split', '0=100'
], check=True)

print(f"Model deployed successfully. You can access it at: https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{IMAGE_NAME}-endpoint")
