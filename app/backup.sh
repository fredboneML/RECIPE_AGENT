#!/bin/bash

BACKUP_DIR="/home/$USER/backups/ai-analyzer"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup PostgreSQL database
docker exec postgres_db pg_dump -U $POSTGRES_USER $POSTGRES_DB > $BACKUP_DIR/db_backup_$TIMESTAMP.sql

# Backup volumes
docker run --rm \
    -v postgres_data:/source:ro \
    -v $BACKUP_DIR:/backup \
    alpine \
    tar czf /backup/postgres_data_$TIMESTAMP.tar.gz /source

# Remove backups older than 7 days
find $BACKUP_DIR -type f -mtime +7 -delete