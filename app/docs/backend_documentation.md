# Backend Documentation

## Architecture Overview

### Technology Stack
- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL
- **ORM**: SQLAlchemy
- **Authentication**: JWT (JSON Web Tokens)
- **AI Integration**: OpenAI API, Groq, Ollama, HuggingFace
- **Vector Database**: Qdrant
- **API Documentation**: OpenAPI/Swagger

### Project Structure
```
backend/
├── ai-analyzer/
│   ├── ai_analyzer/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── config.py               # Configuration settings
│   │   ├── api.py                  # Additional API endpoints
│   │   ├── agents/                 # AI agent implementations
│   │   │   ├── __init__.py
│   │   │   ├── agent_manager.py
│   │   │   ├── base.py
│   │   │   ├── database_inspector.py
│   │   │   ├── factory.py
│   │   │   ├── initial_questions.py
│   │   │   ├── question_generator.py
│   │   │   ├── response_analyzer.py
│   │   │   ├── sql_generator.py
│   │   │   └── workflow.py
│   │   ├── agents.py               # Agent management
│   │   ├── cache_manager.py        # Database caching
│   │   ├── data_pipeline.py        # Data processing pipeline
│   │   ├── data_import_postgresql.py # Database import utilities
│   │   ├── database_agent_postgresql.py # Database agent
│   │   ├── fetch_data_from_api.py  # External API integration
│   │   ├── make_openai_call.py     # OpenAI integration
│   │   ├── make_openai_call_df.py  # DataFrame processing
│   │   ├── tenant_manager.py       # Multi-tenant support
│   │   ├── utils/                  # Utility functions
│   │   │   ├── __init__.py
│   │   │   ├── embedding_singleton.py
│   │   │   ├── model_logger.py
│   │   │   ├── qdrant_client.py
│   │   │   ├── resilience.py
│   │   │   ├── singleton_embeddings.py
│   │   │   ├── singleton_resources.py
│   │   │   └── vector_db.py
│   │   └── utils.py                # General utilities
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── poetry.lock
├── Dockerfile
├── Dockerfile.cron
└── qdrant.Dockerfile
```

## Database Schema

### Core Models

#### User Model
```python
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default='read_only')
    tenant_code = Column(String, nullable=False)
    
    # Username should be unique per tenant
    __table_args__ = (
        UniqueConstraint('username', 'tenant_code', name='unique_username_per_tenant'),
    )
    
    def has_write_permission(self):
        return self.role in ['admin', 'write']
```

#### Transcription Model
```python
class Transcription(Base):
    __tablename__ = 'transcription'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    transcription_id = Column(String, nullable=False)
    processing_date = Column(DateTime, nullable=False)
    transcription = Column(String, nullable=False)
    summary = Column(String)
    topic = Column(String)
    sentiment = Column(String)
    call_duration_secs = Column(Integer, nullable=True)
    tenant_code = Column(String, nullable=False)
    clid = Column(String, nullable=True)
    telephone_number = Column(String, nullable=True)
    call_direction = Column(String, nullable=True)
    
    __table_args__ = (
        UniqueConstraint(
            'transcription_id',
            'processing_date',
            'tenant_code',
            name='unique_tenant_transcription'
        ),
    )
```

#### UserMemory Model (Conversations)
```python
class UserMemory(Base):
    __tablename__ = 'user_memory'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    conversation_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    message_order = Column(Integer, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    followup_questions = Column(JSON, nullable=True)
    
    __table_args__ = (
        UniqueConstraint('conversation_id', 'message_order', name='unique_message_order'),
    )
```

### Database Relationships Diagram
```
Users (1) ←→ (N) UserMemory (Conversations)
  ↓
Transcriptions (Independent table with tenant_code)
```

## API Reference

### Authentication Endpoints

#### POST /api/login
**Description**: Authenticate user and return access token

**Request Body**:
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**Success Response (200)**:
```json
{
  "success": true,
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "username": "admin",
  "tenant_code": "tientelecom",
  "role": "admin",
  "permissions": {
    "canWrite": true
  }
}
```

**Error Response (401)**:
```json
{
  "detail": "Invalid credentials"
}
```

#### POST /api/refresh-token
**Description**: Refresh access token using current token

**Headers**: `Authorization: Bearer <current_token>`

**Success Response (200)**:
```json
{
  "access_token": "new_jwt_token",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Query and Analysis Endpoints

#### POST /api/query
**Description**: Submit a query for AI analysis

**Request Body**:
```json
{
  "query": "What are the trending topics?",
  "conversation_id": "optional-conversation-uuid"
}
```

**Success Response (200)**:
```json
{
  "success": true,
  "response": "Based on the analysis...",
  "conversation_id": "conversation-uuid",
  "followup_questions": [
    "How do these trends compare to last month?",
    "Which topics have the highest satisfaction scores?"
  ]
}
```

#### GET /api/conversations
**Description**: Get user's conversation history

**Success Response (200)**:
```json
[
  {
    "id": "conversation-uuid",
    "title": "What are the most discussed topics...",
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
]
```

#### GET /api/conversations/{conversation_id}
**Description**: Get messages from a specific conversation

**Success Response (200)**:
```json
[
  {
    "id": "message-id",
    "conversation_id": "conversation-uuid",
    "query": "What are the trending topics?",
    "response": "The trending topics are...",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "followup_questions": [
      "How do these topics compare to last month?",
      "Which topics have the highest satisfaction scores?"
    ]
  }
]
```

#### GET /api/initial-questions
**Description**: Get predefined initial questions for analysis

**Success Response (200)**:
```json
{
  "success": true,
  "categories": {
    "Sentiment Analysis": [
      "What is the overall sentiment trend?",
      "Which calls have the most negative sentiment?"
    ],
    "Topic Analysis": [
      "What are the most discussed topics?",
      "How do topics vary by time period?"
    ]
  }
}
```

#### POST /api/analyze-response
**Description**: Analyze a response to extract insights

**Request Body**:
```json
{
  "transcription_id": "transcription-uuid",
  "question_id": "question-uuid",
  "response": "The analysis shows positive sentiment trends..."
}
```

**Success Response (200)**:
```json
{
  "success": true,
  "analysis": "The response indicates strong positive sentiment trends..."
}
```

#### POST /api/generate-followup
**Description**: Generate follow-up questions based on conversation history

**Request Body**:
```json
{
  "conversation_type": "system",
  "questions": ["What are the top customer issues?"],
  "responses": ["The top issues are billing problems (34%) and technical support (28%)..."]
}
```

**Success Response (200)**:
```json
{
  "success": true,
  "followup_questions": [
    "What specific billing problems are customers reporting?",
    "Which technical support issues take the longest to resolve?"
  ]
}
```

### User Management Endpoints

#### POST /api/add_user
**Description**: Add a new user (Admin only)

**Request Body**:
```json
{
  "username": "newuser",
  "password": "password123",
  "role": "read_only"
}
```

**Success Response (200)**:
```json
{
  "success": true,
  "message": "User added successfully"
}
```

### Health Check Endpoints

#### GET /health
**Description**: Comprehensive health check with authentication

**Success Response (200)**:
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

## Authentication & Authorization

### JWT Implementation

#### Token Structure
```python
# JWT Payload
{
  "sub": "username",           # Subject (username)
  "tenant_code": "tientelecom",
  "role": "admin",
  "exp": 1642680000,          # Expiration timestamp
  "iat": 1642676400           # Issued at timestamp
}
```

#### Security Dependencies
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from jose import JWTError, jwt

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
    """Validate JWT token and return current user"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Role-Based Access Control
- **Admin Users**: Full access to all resources and user management
- **Read-Only Users**: Can only view data and run queries
- **Tenant Isolation**: Users can only access data within their tenant

## AI Model Integration

### Multi-Provider Support

#### AI Service Configuration
```python
class AgentManager:
    def __init__(self, tenant_code: str, session: Session = None, model: str = None):
        self.tenant_code = tenant_code
        self.session = session
        self.model = model or "gpt-4o-mini-2024-07-18"
    
    def generate_initial_questions(self, transcription_id: str) -> List[str]:
        """Generate initial questions for analysis"""
        # Implementation using OpenAI API
        pass
    
    def analyze_response(self, transcription_id: str, question_id: str, response_text: str) -> str:
        """Analyze a response to a question"""
        # Implementation using AI models
        pass
```

#### OpenAI Provider Integration
```python
# Configuration in config.py
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o-mini-2024-07-18')
OPENAI_API_KEY = os.getenv('AI_ANALYZER_OPENAI_API_KEY')

# Usage in agents
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7
)
```

## Configuration Management

### Environment-Based Settings
```python
# Required Configuration
config = {
    'POSTGRES_USER': os.getenv('POSTGRES_USER'),
    'POSTGRES_PASSWORD': os.getenv('POSTGRES_PASSWORD'),
    'DB_HOST': os.getenv('DB_HOST', 'database'),
    'DB_PORT': os.getenv('DB_PORT', '5432'),
    'POSTGRES_DB': os.getenv('POSTGRES_DB'),
    'AI_ANALYZER_OPENAI_API_KEY': os.getenv('AI_ANALYZER_OPENAI_API_KEY'),
    'JWT_SECRET_KEY': os.getenv('JWT_SECRET_KEY'),
    'JWT_ALGORITHM': os.getenv('JWT_ALGORITHM', "HS256"),
    'ACCESS_TOKEN_EXPIRE_MINUTES': int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 60 * 24)),
    'TENANT_CODE': os.getenv('TENANT_CODE'),
    'MODEL_NAME': os.getenv('MODEL_NAME', 'gpt-4o-mini-2024-07-18')
}

# Database URL
DATABASE_URL = f"postgresql://{config['POSTGRES_USER']}:{config['POSTGRES_PASSWORD']}@{config['DB_HOST']}:{config['DB_PORT']}/{config['POSTGRES_DB']}"
```

## Error Handling

### Custom Exception Classes
```python
class AIAnalyzerException(Exception):
    """Base exception for AI Analyzer"""
    pass

class ValidationError(AIAnalyzerException):
    """Validation error"""
    pass

class NotFoundError(AIAnalyzerException):
    """Resource not found"""
    pass

class AuthenticationError(AIAnalyzerException):
    """Authentication failed"""
    pass
```

### Global Exception Handler
```python
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": str(exc)
            }
        }
    )
```

---

**Next**: Check the AI Model Documentation for detailed information about model integration and training.