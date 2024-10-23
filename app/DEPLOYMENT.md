# Deployment Guide for AI Analyzer Application

This guide outlines the steps to deploy the AI Analyzer application on a Debian 12 server.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Server Setup](#server-setup)
- [Application Deployment](#application-deployment)
- [Security Configuration](#security-configuration)
- [Backup Configuration](#backup-configuration)
- [Maintenance Scripts](#maintenance-scripts)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites
- Debian 12 server
- Root access or sudo privileges
- OpenAI API key
- Git repository access (if using version control)

## Server Setup

### 1. Update System and Install Dependencies
```bash
# Update the system
sudo apt update
sudo apt upgrade -y

# Install required packages
sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    ufw

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add your user to docker group
sudo usermod -aG docker $USER
```

### 2. Configure Firewall
```bash
# Configure UFW
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 3000
sudo ufw allow 8000
sudo ufw enable
```

## Application Deployment

### 1. Create Project Directory
```bash
# Create directory
mkdir -p ~/ai-analyzer/app
cd ~/ai-analyzer/app
```

### 2. Create Environment File
Create a `.env` file with the following content:
```env
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=ai_analyzer
DB_HOST=database
DB_PORT=5432
OPENAI_API_KEY=your_openai_api_key
AI_ANALYZER_OPENAI_API_KEY=your_openai_api_key
```

### 3. Create Docker Compose File
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  frontend_app:
    build: ./frontend
    container_name: frontend_app
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_BACKEND_URL=http://37.97.226.251:8000
    restart: unless-stopped
    networks:
      - app-network

  backend_app:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: backend_app
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
      - DB_HOST=database
      - DB_PORT=5432
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - AI_ANALYZER_OPENAI_API_KEY=${AI_ANALYZER_OPENAI_API_KEY}
    volumes:
      - ./backend:/usr/src/app
    command: ["uvicorn", "ai_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000"]
    restart: unless-stopped
    networks:
      - app-network

  database:
    image: postgres:13
    container_name: postgres_db
    ports:
      - "5433:5432"
    env_file:
      - .env
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    networks:
      - app-network

volumes:
  postgres_data:
    name: ai_analyzer_postgres_data

networks:
  app-network:
    driver: bridge
```

### 4. Create Deployment Script
Create `deploy.sh`:
```bash
#!/bin/bash

# Stop running containers
docker-compose down

# Build and start containers
docker-compose up -d --build

# Show container status
docker-compose ps

# Show logs
docker-compose logs -f
```

Make it executable:
```bash
chmod +x deploy.sh
```

Run deployment
```bash
./deploy.sh
```

## Backup Configuration

### 1. Create Backup Script
Create `backup.sh`:
```bash
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
```

### 2. Create Update Script
Create `update.sh`:
```bash
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
```

### 3. Configure Automatic Updates and Backups
Add to crontab:
```bash
# Open crontab
crontab -e

# Add these lines
0 3 * * * /home/$USER/ai-analyzer/app/backup.sh
0 4 * * 0 /home/$USER/ai-analyzer/app/update.sh
```

## Monitoring

### Check Application Status
```bash
# View logs
docker-compose logs -f

# Check container status
docker-compose ps

# Monitor system resources
htop

# Monitor docker stats
docker stats
```

## Security Best Practices

1. Use strong passwords in `.env` file
2. Keep the server updated regularly
3. Monitor logs for suspicious activity
4. Configure fail2ban for SSH protection
5. Use SSL/TLS certificates if using a domain
6. Regular security audits
7. Implement rate limiting
8. Regular backups

## Accessing the Application

After deployment, access your application at:
- Frontend: `http://37.97.226.251:3000`
- Backend API: `http://37.97.226.251:8000`

## Troubleshooting

### Common Issues and Solutions

1. Container fails to start:
```bash
# Check container logs
docker-compose logs [service_name]
```

2. Database connection issues:
```bash
# Check if database is running
docker-compose ps database
# Check database logs
docker-compose logs database
```

3. Permission issues:
```bash
# Check file permissions
ls -la
# Fix permissions if needed
chmod -R 755 .
```

### Helpful Commands

```bash
# Restart specific service
docker-compose restart [service_name]

# View running containers
docker ps

# View container logs
docker logs -f [container_name]

# Access container shell
docker exec -it [container_name] sh
```

## Support

For additional support or questions:
1. Check the application documentation
2. Review Docker and Docker Compose documentation
3. Contact the development team
