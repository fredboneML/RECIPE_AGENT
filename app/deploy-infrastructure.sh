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
echo "Building Docker images..."
docker-compose -f infrastructure-compose.yml build --no-cache

# Start containers
echo "Starting containers..."
docker-compose -f infrastructure-compose.yml up -d

# Wait for database to be ready
echo "Waiting for database to be ready..."
max_retries=30
retries=0
while ! docker-compose -f infrastructure-compose.yml exec -T database pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" 2>/dev/null; do
    retries=$((retries + 1))
    if [ $retries -ge $max_retries ]; then
        echo "Error: Database not ready after $max_retries attempts. Check logs with 'docker-compose -f infrastructure-compose.yml logs database'"
        exit 1
    fi
    printf '.'
    sleep 5
done
echo "Database is ready!"

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to be ready..."
max_retries=30
retries=0
while ! curl --output /dev/null --silent --fail http://localhost:6333/healthz; do
    retries=$((retries + 1))
    if [ $retries -ge $max_retries ]; then
        echo "Error: Qdrant not ready after $max_retries attempts. Check logs with 'docker-compose -f infrastructure-compose.yml logs qdrant'"
        exit 1
    fi
    printf '.'
    sleep 5
done
echo "Qdrant is ready!"

# Show logs - use a separate terminal for this
echo "Infrastructure deployed successfully!"
# Show logs
docker-compose -f infrastructure-compose.yml logs -f
