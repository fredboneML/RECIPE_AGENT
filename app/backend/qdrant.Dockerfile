FROM --platform=linux/arm64 qdrant/qdrant:latest

# Install curl for debugging and healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Expose the Qdrant ports
EXPOSE 6333
EXPOSE 6334

# Set up a volume for persistence
VOLUME /qdrant/storage

# Set the health check correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:6333/healthz || exit 1

# Use the default entrypoint from the base image