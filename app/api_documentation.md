# AI Analyzer API Documentation

## Overview

The AI Analyzer is a call center analytics application that provides insights into customer conversations through sentiment analysis, topic extraction, and trend analysis. The application supports multi-tenant architecture with JWT-based authentication.

## Base URLs

- **Local Development**: `http://localhost:8000`
- **Debian Server**: `http://37.97.226.251:8000`
- **Frontend (Local)**: `http://localhost:3000`
- **Frontend (Server)**: `http://37.97.226.251:3000`

## Authentication

The API uses JWT (JSON Web Token) based authentication with Bearer token scheme.

### Login Flow
1. User submits credentials to `/api/login`
2. Server returns JWT token with user information
3. Client includes token in `Authorization` header for subsequent requests
4. Token automatically refreshes before expiration

### Required Headers
```http
Authorization: Bearer <jwt_token>
X-Tenant-Code: <tenant_code>
Content-Type: application/json
Accept: application/json
```

## API Endpoints

### Authentication Endpoints

#### POST /api/login
Authenticate user and receive JWT token.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response (Success):**
```json
{
  "success": true,
  "access_token": "jwt_token_string",
  "token_type": "bearer",
  "username": "user123",
  "tenant_code": "tenant_abc",
  "role": "admin",
  "permissions": {
    "canWrite": true
  }
}
```

**Response (Error):**
```json
{
  "detail": "Invalid credentials"
}
```

#### POST /api/refresh-token
Refresh JWT token before expiration.

**Headers:**
```http
Authorization: Bearer <current_token>
```

**Response:**
```json
{
  "access_token": "new_jwt_token",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Core Analytics Endpoints

#### POST /api/query
Process natural language queries about call data.

**Headers:**
```http
Authorization: Bearer <jwt_token>
X-Tenant-Code: <tenant_code>
```

**Request Body:**
```json
{
  "query": "What are the most discussed topics this month?",
  "conversation_id": "optional-uuid-string"
}
```

**Response:**
```json
{
  "response": "Based on the call data analysis, the top discussed topics this month are:\n1. Technical Support - 45% of calls\n2. Billing Questions - 23% of calls\n3. Product Information - 18% of calls",
  "conversation_id": "uuid-string",
  "followup_questions": [
    "Which technical support issues are most common?",
    "How has billing sentiment changed over time?",
    "What product features are customers asking about most?"
  ]
}
```

#### GET /api/initial-questions
Get categorized initial questions for analysis.

**Headers:**
```http
X-Tenant-Code: <tenant_code>
X-UI-Language: nl|en (optional, defaults to 'nl')
```

**Response:**
```json
{
  "success": true,
  "categories": {
    "Trending Topics": {
      "description": "Analyze popular discussion topics",
      "questions": [
        "What are the most discussed topics this month?",
        "Which topics show increasing trends?",
        "What topics are commonly mentioned in positive calls?"
      ]
    },
    "Customer Sentiment": {
      "description": "Understand customer satisfaction trends",
      "questions": [
        "How has overall sentiment changed over time?",
        "What topics generate the most positive feedback?",
        "Which issues need immediate attention based on sentiment?"
      ]
    }
  }
}
```

### Conversation Management

#### GET /api/conversations
Get user's conversation history.

**Headers:**
```http
X-Tenant-Code: <tenant_code>
```

**Response:**
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
Get messages from a specific conversation.

**Headers:**
```http
X-Tenant-Code: <tenant_code>
```

**Response:**
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

### Analysis Endpoints

#### POST /api/analyze-response
Analyze a response to extract insights.

**Headers:**
```http
X-Tenant-Code: <tenant_code>
```

**Request Body:**
```json
{
  "transcription_id": "transcription-uuid",
  "question_id": "question-uuid",
  "response": "The analysis shows positive sentiment trends..."
}
```

**Response:**
```json
{
  "success": true,
  "analysis": "The response indicates strong positive sentiment trends with a 15% increase in customer satisfaction scores over the past month..."
}
```

#### POST /api/generate-followup
Generate follow-up questions based on conversation history.

**Headers:**
```http
X-Tenant-Code: <tenant_code>
```

**Request Body:**
```json
{
  "conversation_type": "system",
  "questions": [
    "What are the top customer issues?"
  ],
  "responses": [
    "The top issues are billing problems (34%) and technical support (28%)..."
  ]
}
```

**Response:**
```json
{
  "success": true,
  "followup_questions": [
    "What specific billing problems are customers reporting?",
    "Which technical support issues take the longest to resolve?",
    "How do these issues vary by customer segment?"
  ]
}
```

### Health and Monitoring

#### GET /health
Check application health status.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "qdrant": "connected",
  "collections": ["tenant_example1", "tenant_example2"]
}
```

## Data Models

### Query Request
```json
{
  "query": "string",           // Natural language query
  "question": "string",        // Alternative field name (legacy support)
  "conversation_id": "string"  // Optional UUID for conversation continuity
}
```

### User Model
```json
{
  "id": "integer",
  "username": "string",
  "role": "admin|write|read_only",
  "tenant_code": "string"
}
```

### Conversation Message
```json
{
  "id": "string",
  "conversation_id": "string",
  "query": "string",
  "response": "string",
  "timestamp": "ISO 8601 datetime",
  "followup_questions": ["string"],
  "message_order": "integer"
}
```

## Error Handling

### HTTP Status Codes
- `200 OK` - Request successful
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Invalid or missing authentication
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

### Error Response Format
```json
{
  "detail": "Error description",
  "success": false,
  "error": "Detailed error message"
}
```

### Common Error Scenarios
1. **Invalid Token**: `401 Unauthorized` - Token expired or invalid
2. **Missing Tenant**: `401 Unauthorized` - X-Tenant-Code header missing
3. **Query Processing Error**: `500 Internal Server Error` - SQL generation or execution failure
4. **Context Length Exceeded**: Special error for conversation limit reached

## Multi-Tenant Architecture

The application supports multiple tenants with data isolation:

- Each tenant has isolated data in the database
- Tenant code must be provided in `X-Tenant-Code` header
- Vector search collections are tenant-specific (`tenant_{tenant_code}`)
- All SQL queries include automatic tenant filtering

## Rate Limiting and Performance

- Query caching is implemented to improve response times
- Conversation context is limited to prevent token overflow
- Circuit breakers protect against Qdrant connection failures
- Connection pooling for database optimization

## Example Usage

### Complete Authentication Flow
```javascript
// 1. Login
const loginResponse = await fetch('http://37.97.226.251:8000/api/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    username: 'your_username',
    password: 'your_password'
  })
});

const { access_token, tenant_code } = await loginResponse.json();

// 2. Query call data
const queryResponse = await fetch('http://37.97.226.251:8000/api/query', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`,
    'X-Tenant-Code': tenant_code,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    query: 'What are the most common customer complaints this week?'
  })
});

const result = await queryResponse.json();
console.log(result.response); // Analysis results
console.log(result.followup_questions); // Suggested next questions
```

### Using with Docker Environment Variables

For deployment, set the `HOST` environment variable:

```bash
# Local development
HOST=localhost docker-compose up

# Server deployment
HOST=37.97.226.251 docker-compose up
```

The frontend will automatically use the correct backend URL based on the `HOST` environment variable.

## Supported Query Types

The AI Analyzer supports various types of natural language queries:

### Topic Analysis
- "What are the most discussed topics this month?"
- "Which topics are trending upward?"
- "Show me topics with negative sentiment"

### Sentiment Analysis
- "How has customer sentiment changed over time?"
- "What percentage of calls are positive vs negative?"
- "Which topics have the best satisfaction scores?"

### Temporal Analysis
- "Compare this month's data to last month"
- "What are the busiest call times?"
- "Show me weekly trend patterns"

### Call Metrics
- "What's the average call duration by topic?"
- "Which issues take the longest to resolve?"
- "How many calls did we receive today?"

## Language Support

The API supports both English and Dutch:
- Use `X-UI-Language: nl` for Dutch responses
- Use `X-UI-Language: en` for English responses
- Questions can be asked in either language
- Responses will match the language of the question

## Security Considerations

1. **JWT Token Security**: Tokens auto-refresh before expiration
2. **Tenant Isolation**: Strict data separation between tenants
3. **SQL Injection Protection**: Parameterized queries and validation
4. **CORS Configuration**: Configured for specific origins
5. **Input Validation**: All inputs are validated before processing

## Deployment Notes

### Environment Variables Required
```bash
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_DB=your_db_name
OPENAI_API_KEY=your_openai_key
AI_ANALYZER_OPENAI_API_KEY=your_openai_key
JWT_SECRET_KEY=your_secret_key
ADMIN_USER=admin_username
ADMIN_PASSWORD=admin_password
HOST=37.97.226.251  # For server deployment
```

### Docker Compose Deployment
```bash
# Clone the repository
git clone <repository_url>
cd ai-analyzer

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Deploy infrastructure (database, vector DB)
./deploy-infrastructure.sh

# Deploy application
HOST=37.97.226.251 ./deploy-app.sh
```

The application will be available at:
- Frontend: `http://37.97.226.251:3000`
- Backend API: `http://37.97.226.251:8000`
- API Documentation: This document

For local development, use `localhost` instead of the server IP address.