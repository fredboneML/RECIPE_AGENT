# Frontend Documentation

## Project Structure

```
frontend/
├── public/
│   ├── index.html
│   ├── logo.png
│   ├── logo192.png
│   ├── logo512.png
│   ├── manifest.json
│   └── robots.txt
├── src/
│   ├── components/           # Reusable UI components
│   │   └── ChatInterface.js  # Main chat interface component
│   ├── services/            # API and external services
│   │   └── api.js           # API service configuration
│   ├── App.js               # Root component with routing
│   ├── App.css              # Main application styles
│   ├── App.test.js          # Test file
│   ├── index.js             # Entry point
│   ├── index.css            # Global styles
│   ├── Login.js             # Login component
│   ├── Login.css            # Login styles
│   ├── LanguageContext.js   # Language context provider
│   ├── translations.js      # Translation utilities
│   ├── tokenManager.js      # JWT token management
│   ├── reportWebVitals.js   # Performance monitoring
│   └── setupTests.js        # Test configuration
├── package.json
├── package-lock.json
├── Dockerfile
└── README.md
```

## Key Components

### 1. App Component (`src/App.js`)

The main application component that handles:
- Authentication state management
- Conversation management
- Query processing
- Real-time updates

**Key Features:**
- JWT token verification
- Conversation history
- Real-time query processing
- Error handling and retry logic
- Language switching support

**State Management:**
```jsx
const [isAuthChecking, setIsAuthChecking] = useState(true);
const [conversations, setConversations] = useState([]);
const [currentConversation, setCurrentConversation] = useState(null);
const [messages, setMessages] = useState([]);
const [query, setQuery] = useState('');
const [result, setResult] = useState('');
const [isProcessing, setIsProcessing] = useState(false);
const [categories, setCategories] = useState({});
```

### 2. Login Component (`src/Login.js`)

Handles user authentication with:
- Username/password form
- JWT token storage
- Automatic redirect on success
- Error handling and display

**Authentication Flow:**
```jsx
const handleSubmit = async (event) => {
    // Submit credentials to /api/login
    // Store JWT token using tokenManager
    // Store user data in localStorage
    // Verify authentication with health check
    // Redirect to main application
};
```

### 3. ChatInterface Component (`src/components/ChatInterface.js`)

The main chat interface that provides:
- Message display
- Query input
- Real-time responses
- Follow-up questions
- Conversation management

### 4. TokenManager (`src/tokenManager.js`)

Centralized JWT token management with:
- Token storage and retrieval
- Automatic token refresh
- API request interception
- Authentication state management

**Key Methods:**
```jsx
// Token management
setToken(token)
getToken()
clearToken()
isLoggedIn()

// API requests with authentication
get(url, options)
post(url, data, options)
put(url, data, options)
delete(url, options)

// Token refresh
scheduleRefresh()
refreshToken()
```

## State Management

### Context API Structure

#### LanguageContext
```jsx
const LanguageContext = createContext();

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState('nl'); // Default to Dutch
  
  const t = (key) => {
    return translations[language][key] || key;
  };
  
  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};
```

### Local Storage Management

#### User Data Storage
```jsx
// User authentication data
localStorage.setItem('userName', username);
localStorage.setItem('tenantCode', tenantCode);
localStorage.setItem('token', accessToken);

// User profile data
localStorage.setItem('user', JSON.stringify({
  username: data.username,
  role: data.role,
  tenant_code: data.tenant_code,
  permissions: data.permissions
}));
```

## API Integration

### API Service Configuration

#### Base API Setup (`src/services/api.js`)
```jsx
// API base URL configuration
const backendUrl = window.location.hostname === 'localhost' 
  ? 'http://localhost:8000'
  : `http://${window.location.hostname}:8000`;

// Default headers
const getHeaders = () => ({
  'Content-Type': 'application/json',
  'X-Tenant-Code': localStorage.getItem('tenantCode'),
  'Accept': 'application/json',
  'X-UI-Language': language,
  'Authorization': `Bearer ${localStorage.getItem('token')}`,
});
```

### API Endpoints

#### Authentication Endpoints
```jsx
// Login
POST /api/login
{
  "username": "string",
  "password": "string"
}

// Token refresh
POST /api/refresh-token
Headers: Authorization: Bearer <current_token>
```

#### Query and Analysis Endpoints
```jsx
// Submit query
POST /api/query
{
  "query": "string",
  "conversation_id": "optional-uuid"
}

// Get conversations
GET /api/conversations

// Get conversation messages
GET /api/conversations/{conversation_id}

// Get initial questions
GET /api/initial-questions

// Analyze response
POST /api/analyze-response
{
  "transcription_id": "uuid",
  "question_id": "uuid",
  "response": "string"
}

// Generate follow-up questions
POST /api/generate-followup
{
  "conversation_type": "string",
  "questions": ["string"],
  "responses": ["string"]
}
```

#### Health Check
```jsx
// Health check
GET /health
Headers: Authorization: Bearer <token>
```

### Expected Response Formats

#### Success Response
```json
{
  "success": true,
  "response": "Analysis result...",
  "conversation_id": "uuid",
  "followup_questions": [
    "Follow-up question 1",
    "Follow-up question 2"
  ]
}
```

#### Error Response
```json
{
  "detail": "Error message"
}
```

#### Authentication Response
```json
{
  "success": true,
  "access_token": "jwt_token",
  "token_type": "bearer",
  "username": "admin",
  "tenant_code": "tientelecom",
  "role": "admin",
  "permissions": {
    "canWrite": true
  }
}
```

## Styling Approach

### CSS Framework
- **Primary**: CSS Modules for component-specific styles
- **Global**: CSS files for application-wide styles
- **Responsive**: Mobile-first design approach

### Styling Organization

#### Global Styles (`src/index.css`)
```css
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
```

#### Component Styles (`src/App.css`)
```css
/* Main application layout */
.app {
  text-align: center;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Header styles */
.header {
  background-color: #282c34;
  padding: 20px;
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* Chat interface styles */
.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

/* Message styles */
.message {
  margin: 10px 0;
  padding: 15px;
  border-radius: 8px;
  max-width: 80%;
}

.user-message {
  background-color: #007bff;
  color: white;
  align-self: flex-end;
}

.ai-message {
  background-color: #f8f9fa;
  border: 1px solid #dee2e6;
  align-self: flex-start;
}
```

#### Login Styles (`src/Login.css`)
```css
.login-page {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background-color: #f5f5f5;
}

.login-container {
  background: white;
  padding: 40px;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 400px;
}

.input-group {
  margin-bottom: 20px;
}

.input-group label {
  display: block;
  margin-bottom: 5px;
  font-weight: 500;
}

.input-group input {
  width: 100%;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 16px;
}

.error {
  color: #dc3545;
  margin-top: 10px;
  font-size: 14px;
}
```

## Development Guidelines

### Component Best Practices
1. Use functional components with hooks
2. Implement proper error boundaries
3. Follow single responsibility principle
4. Use descriptive naming conventions
5. Implement proper loading states

### Performance Optimization
1. Use `React.memo()` for expensive components
2. Implement proper loading states
3. Use `useMemo()` and `useCallback()` appropriately
4. Optimize re-renders with proper dependencies
5. Implement proper error handling

### Accessibility
1. Use semantic HTML elements
2. Implement proper ARIA labels
3. Ensure keyboard navigation
4. Maintain color contrast ratios
5. Add alt text for images

### Error Handling
```jsx
// Global error handling
useEffect(() => {
  const handleError = (error) => {
    console.error('Global error:', error);
    // Handle authentication errors
    if (error.message === 'Unauthorized') {
      localStorage.clear();
      navigate('/login');
    }
  };
  
  window.addEventListener('error', handleError);
  return () => window.removeEventListener('error', handleError);
}, [navigate]);
```

### Authentication Flow
```jsx
// Token verification on app start
useEffect(() => {
  const verifyAuth = async () => {
    try {
      const userName = localStorage.getItem('userName');
      const tenantCode = localStorage.getItem('tenantCode');
      
      if (!userName || !tenantCode) {
        navigate('/login', { replace: true });
        return;
      }
      
      // Verify token with health check
      const healthResponse = await tokenManager.get('/health');
      if (!healthResponse.ok) {
        throw new Error('Health check failed');
      }
      
      // Load conversations
      const conversationsResponse = await tokenManager.get('/api/conversations');
      if (!conversationsResponse.ok) {
        throw new Error('Auth verification failed');
      }
      
    } catch (error) {
      localStorage.clear();
      navigate('/login', { replace: true });
    }
  };
  
  verifyAuth();
}, [navigate]);
```

---

**Next Steps**: Check the Backend Documentation for API details and integration patterns.