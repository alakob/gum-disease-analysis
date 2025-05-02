#!/bin/bash

# Stop and remove any existing container with the same name
echo "Stopping any existing container..."
docker stop gum-disease-app 2>/dev/null
docker rm gum-disease-app 2>/dev/null

# Build the Docker image
echo "Building Docker image..."
docker build -t gum-disease-app .

# Run the container
echo "Starting container..."
docker run -d --name gum-disease-app \
  -p 8505:8502 \
  --env-file .env \
  gum-disease-app

echo "Container started! Access the app at http://localhost:8505"
