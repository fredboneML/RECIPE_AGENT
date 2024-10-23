#!/bin/bash

# Stop running containers
docker-compose down

# Pull latest changes if using git
# git pull

# Build and start containers
docker-compose up -d --build

# Show container status
docker-compose ps

# Show logs
docker-compose logs -f