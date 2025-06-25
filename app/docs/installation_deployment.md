# Installation & Deployment Guide

## Prerequisites

### System Requirements
- **Node.js**: Version 16.0 or higher
- **Python**: Version 3.8 or higher
- **PostgreSQL**: Version 12 or higher
- **Docker**: Version 20.0 or higher
- **Docker Compose**: Version 2.0 or higher
- **Git**: Latest version

### API Keys Required
- OpenAI API Key (required for AI functionality)
- Additional provider API keys (optional, for Groq/HuggingFace)

## Local Development Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ai-analyzer/app
```

### 2. Environment Configuration
Create a `.env` file in the project app and backend directory:

```bash
nano  .env
```

Edit the `.env` file with your configuration:

```env
# Database settings
POSTGRES_USER=ai_analyzer_user
POSTGRES_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432
POSTGRES_DB=ai_analyzer_db

# OpenAI settings
AI_ANALYZER_OPENAI_API_KEY=your_openai_api_key
OPENAI_API_KEY=your_openai_api_key

# Backend URL
REACT_APP_BACKEND_URL=http://localhost:8000

# External API settings
URL=https://dev.10telecom.nl/ai-analyzer-5Ck9gthhdekmn/get_transcriptions.php?
API_KEY=your_external_api_key

# Database users
ADMIN_USER=admin
ADMIN_PASSWORD=admin_password
READ_USER=readonly
READ_USER_PASSWORD=readonly_password
USER_ID=1000
GROUP_ID=1000

# Model Provider Configuration
MODEL_PROVIDER=openai
MODEL_NAME='gpt-4.1-nano'
USE_OPENAI_COMPATIBILITY=true

# Application settings
TENANT_CODE=tientelecom
LAST_ID=0
LIMIT=500

# JWT settings
JWT_SECRET_KEY=your_jwt_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=1440
REFRESH_TOKEN_EXPIRE_DAYS=7
JWT_ALGORITHM=HS256
```

### 3. Database Setup

#### Option A: Local PostgreSQL
1. Install PostgreSQL locally
2. Create database and user:
```sql
CREATE DATABASE ai_analyzer_db;
CREATE USER ai_analyzer_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE ai_analyzer_db TO ai_analyzer_user;
```

#### Option B: Docker PostgreSQL
```bash
docker run --name ai-analyzer-postgres \
  -e POSTGRES_DB=ai_analyzer_db \
  -e POSTGRES_USER=ai_analyzer_user \
  -e POSTGRES_PASSWORD=your_secure_password \
  -p 5432:5432 \
  -d postgres:13
```

### 4. Backend Setup
```bash
# Navigate to backend directory
cd app/backend/ai-analyzer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from ai_analyzer.data_import_postgresql import run_data_import; run_data_import()"

# Start the backend server
uvicorn ai_analyzer.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at: `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### 5. Frontend Setup
```bash
# Navigate to frontend directory
cd app/frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at: `http://localhost:3000`

## Debian Server Deployment

### Prerequisites for Server Deployment
- **Debian Server**: With Docker and Docker Compose installed
- **Domain/Subdomain**: Configured to point to your server
- **SSL Certificate**: For HTTPS (recommended)
- **Firewall**: Configured to allow ports 80, 443, 3000, 8000, 5433, 6333

### Deployment Process

#### 1. Initial Server Setup
```bash
# SSH into your Debian server
ssh user@your-server-ip

# Clone the repository
git clone <repository-url>
cd ai-analyzer/app

# Make deployment scripts executable
chmod +x deploy-infrastructure.sh
chmod +x deploy-app.sh
```

#### 2. Environment Configuration
```bash
# Create and configure environment file
nano .env

# Add all required environment variables (see Local Development Setup above)
# Make sure to update URLs and API keys for production
```

#### 3. Deploy Infrastructure (First Time Only)
```bash
# Deploy databases and core infrastructure
./deploy-infrastructure.sh
```

This script will:
- Deploy PostgreSQL database container
- Deploy Qdrant vector database container
- Set up persistent volumes for data storage
- Configure network connectivity
- Initialize database schemas

**Wait for infrastructure deployment to complete before proceeding.**

#### 4. Deploy Application
```bash
# Deploy the main application
./deploy-app.sh
```

This script will:
- Build and deploy frontend container
- Build and deploy backend container
- Deploy cron job container for background processing
- Set up health checks and monitoring
- Configure inter-service communication

#### 5. Verify Deployment
```bash
# Check all services are running
docker-compose ps

# Check service logs
docker-compose logs -f

# Test health endpoints
curl http://your-server-ip:8000/health
curl http://your-server-ip:3000
```

### Production Deployment Notes

#### Environment Variables for Production
```env
# Update these for production
REACT_APP_BACKEND_URL=https://your-domain.com:8000
HOST=your-domain.com
ALLOWED_ORIGINS=https://your-domain.com,https://www.your-domain.com

# Security settings
JWT_SECRET_KEY=your-very-secure-production-secret
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Database settings (use strong passwords)
POSTGRES_PASSWORD=your-very-secure-db-password
ADMIN_PASSWORD=your-very-secure-admin-password
```

#### SSL/HTTPS Configuration
```bash
# Install Certbot for SSL certificates
sudo apt update
sudo apt install certbot

# Obtain SSL certificate
sudo certbot certonly --standalone -d your-domain.com

# Configure nginx or reverse proxy for HTTPS
# (Additional configuration required)
```

#### Monitoring and Logs
```bash
# View real-time logs
docker-compose logs -f

# Check specific service logs
docker-compose logs -f backend_app
docker-compose logs -f frontend_app
docker-compose logs -f database

# Monitor resource usage
docker stats
```

## Docker Deployment

### 1. Docker Compose Setup
The application includes a `docker-compose.yml` file for easy deployment:

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Docker Services Overview
- **frontend_app**: React application (port 3000)
- **backend_app**: FastAPI application (port 8000)
- **database**: PostgreSQL database (port 5433)
- **qdrant**: Vector database for embeddings (port 6333)
- **cron_app**: Background data processing

### 3. Production Docker Compose
For production deployment, use `docker-compose.prod.yml`:

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Update production environment
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

## Environment Variables Reference

### Database Configuration
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `POSTGRES_USER` | Database username | ✅ | - |
| `POSTGRES_PASSWORD` | Database password | ✅ | - |
| `DB_HOST` | Database host | ✅ | database |
| `DB_PORT` | Database port | ❌ | 5432 |
| `POSTGRES_DB` | Database name | ✅ | - |

### AI Model Configuration
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AI_ANALYZER_OPENAI_API_KEY` | OpenAI API key | ✅ | - |
| `OPENAI_API_KEY` | OpenAI API key (alternative) | ✅ | - |
| `MODEL_NAME` | Model identifier | ❌ | gpt-4.1-nano |
| `USE_OPENAI_COMPATIBILITY` | Enable OpenAI-compatible API | ❌ | true |

### Authentication
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `JWT_SECRET_KEY` | JWT signing key | ✅ | - |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiry time | ❌ | 1440 |
| `JWT_ALGORITHM` | JWT algorithm | ❌ | HS256 |

### Application Settings
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `REACT_APP_BACKEND_URL` | Backend API URL | ✅ | - |
| `TENANT_CODE` | Tenant identifier | ❌ | tientelecom |
| `LIMIT` | Default query limit | ❌ | 500 |
| `ADMIN_USER` | Admin username | ❌ | admin |
| `ADMIN_PASSWORD` | Admin password | ❌ | admin_password |
| `READ_USER` | Read-only username | ❌ | readonly |
| `READ_USER_PASSWORD` | Read-only password | ❌ | readonly_password |

### External API Configuration
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `URL` | External API URL | ❌ | - |
| `API_KEY` | External API key | ❌ | - |

## Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check database logs
docker logs postgres_db

# Test database connection
psql -h localhost -U ai_analyzer_user -d ai_analyzer_db -p 5433
```

#### Port Conflicts
```bash
# Check which process is using a port
lsof -i :8000  # Backend port
lsof -i :3000  # Frontend port
lsof -i :5433  # Database port (Docker)
lsof -i :6333  # Qdrant port

# Kill process if needed
kill -9 <PID>
```

#### Docker Issues
```bash
# Clean Docker system
docker system prune -f

# Rebuild containers
docker-compose down -v
docker-compose up --build

# Check container logs
docker-compose logs <service-name>

# Check specific service health
docker-compose ps
```

#### Frontend Build Issues
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check for dependency conflicts
npm audit
```

#### Backend Issues
```bash
# Check Python dependencies
pip list

# Verify virtual environment
which python
pip show fastapi

# Check backend logs
docker-compose logs backend_app

# Test backend directly
curl http://localhost:8000/health
```

### Performance Optimization

#### Development
- Use `NODE_ENV=development` for detailed error messages
- Enable React development tools
- Use `--reload` flag for FastAPI auto-reload
- Enable hot reload for frontend

#### Production
- Set `NODE_ENV=production`
- Enable gzip compression
- Use environment-specific database pools
- Configure proper logging levels
- Enable Qdrant persistence

## Health Checks

### Backend Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "qdrant": "connected",
  "collections": ["transcriptions", "embeddings"],
  "user": "admin",
  "tenant": "tientelecom"
}
```

### Database Health Check
```bash
# Direct database check
docker exec postgres_db pg_isready -U ai_analyzer_user -d ai_analyzer_db

# Through backend
curl -H "Authorization: Bearer <token>" http://localhost:8000/health
```

### Frontend Health Check
```bash
curl http://localhost:3000
```

### Qdrant Health Check
```bash
curl http://localhost:6333/health
```

## Initial Setup

### Database Initialization
```bash
# Run database initialization
cd app/backend/ai-analyzer
python -c "from ai_analyzer.data_import_postgresql import run_data_import; run_data_import()"
```

This will:
- Create necessary database tables
- Create admin and read-only users
- Set up tenant configuration
- Initialize restricted tables

### Default Users
After initialization, the following users are created:

| Username | Password | Role | Permissions |
|----------|----------|------|-------------|
| `admin` | `admin_password` | admin | Full access |
| `readonly` | `readonly_password` | read_only | Read-only access |

### First Login
1. Navigate to `http://localhost:3000`
2. Use admin credentials:
   - Username: `admin`
   - Password: `admin_password`
3. The system will automatically load initial questions and conversations

## Monitoring and Maintenance

### Log Monitoring
```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend_app
docker-compose logs -f frontend_app
docker-compose logs -f database

# Check for errors
docker-compose logs | grep ERROR
```

### Database Maintenance
```bash
# Backup database
docker exec postgres_db pg_dump -U ai_analyzer_user ai_analyzer_db > backup.sql

# Restore database
docker exec -i postgres_db psql -U ai_analyzer_user ai_analyzer_db < backup.sql

# Check database size
docker exec postgres_db psql -U ai_analyzer_user -d ai_analyzer_db -c "SELECT pg_size_pretty(pg_database_size('ai_analyzer_db'));"
```

### Performance Monitoring
```bash
# Check container resource usage
docker stats

# Check disk usage
docker system df

# Monitor API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
```

---

**Need Help?** Check the troubleshooting section or contact the development team.