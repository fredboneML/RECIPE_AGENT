#!/bin/bash
# Script to run admin endpoint tests inside Docker container

echo "=========================================="
echo "Running Admin Endpoint Tests"
echo "=========================================="

# Check if we're in Docker or need to run in Docker
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    echo "Running inside Docker container..."
    python3 /usr/src/app/tests/test_admin_endpoints.py
else
    echo "Running tests via Docker exec..."
    cd "$(dirname "$0")/../.." || exit 1
    docker-compose exec backend_app python3 /usr/src/app/tests/test_admin_endpoints.py
fi
