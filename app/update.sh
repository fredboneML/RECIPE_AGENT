#!/bin/bash

# Update system packages
sudo apt update
sudo apt upgrade -y

# Update docker images
docker-compose pull

# Restart containers
docker-compose down
docker-compose up -d

# Remove unused images and volumes
docker system prune -af --volumes