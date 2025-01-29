#!/bin/bash

# Export current user's UID and GID
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

echo "Deploying application with UID=$USER_ID and GID=$GROUP_ID"

# Check if infrastructure network exists
if ! docker network ls | grep -q "app-network"; then
    echo "Error: app_network not found. Please deploy infrastructure first."
    exit 1
fi

# Stop any running containers from the app compose
docker-compose -f app-compose.yml down

# Create necessary directories with correct permissions
mkdir -p frontend/node_modules
sudo chown -R $USER_ID:$GROUP_ID frontend/node_modules frontend/src frontend/public

# Clean up any existing node_modules
rm -rf frontend/node_modules/*

# Rebuild containers with current user's UID and GID
docker-compose -f app-compose.yml build --no-cache

# Start containers
docker-compose -f app-compose.yml up -d

# Show logs
docker-compose -f app-compose.yml logs -f
