#!/bin/bash

# Export current user's UID and GID
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

echo "Deploying infrastructure with UID=$USER_ID and GID=$GROUP_ID"

# Stop any running containers from the infrastructure compose
docker-compose -f infrastructure-compose.yml down

# Remove existing volumes if needed
read -p "Do you want to remove existing database volumes? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose -f infrastructure-compose.yml down -v
fi

# Create necessary directories with correct permissions
mkdir -p database/data
sudo chown -R $USER_ID:$GROUP_ID database/data

# Ensure the backend data directory exists
mkdir -p backend/ai-analyzer/data
sudo chown -R $USER_ID:$GROUP_ID backend/ai-analyzer/data

# Rebuild containers with current user's UID and GID
docker-compose -f infrastructure-compose.yml build --no-cache

# Start containers
docker-compose -f infrastructure-compose.yml up -d

# Show logs
docker-compose -f infrastructure-compose.yml logs -f
