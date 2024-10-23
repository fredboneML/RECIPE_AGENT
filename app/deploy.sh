#!/bin/bash

# Stop running containers
docker-compose down


# Pull latest changes if using git
# git pull

# Build the containers
docker-compose build --no-cache

# Start the containers
docker-compose up -d

# Show container status
docker-compose ps

# Show logs
docker-compose logs -f