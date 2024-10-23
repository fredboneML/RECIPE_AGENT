#!/bin/bash

# Export current user's UID and GID
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

echo "Building with UID=$USER_ID and GID=$GROUP_ID"

# Stop any running containers
docker-compose down

# Remove existing volumes
docker-compose down -v

# Create necessary directories with correct permissions
mkdir -p frontend/node_modules
sudo chown -R $USER_ID:$GROUP_ID frontend/node_modules

# Rebuild containers with current user's UID and GID
docker-compose build --no-cache

# Start containers
docker-compose up -d

# Show logs
docker-compose logs -f