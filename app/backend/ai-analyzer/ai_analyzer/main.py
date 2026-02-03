# app/backend/ai-analyzer/ai_analyzer/main.py
from ai_analyzer.agents.recipe_search_agent import RecipeSearchAgent
from ai_analyzer.agents.data_extractor_router import DataExtractorRouterAgent
from ai_analyzer.cache_manager import DatabaseCacheManager
from ai_analyzer.data_import_postgresql import (
    User,
    UserMemory,
    store_conversation
)
from ai_analyzer.config import (
    config, DATABASE_URL, JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES,
    AZURE_AD_TENANT_ID, AZURE_AD_CLIENT_ID, SSO_ENABLED, LOCAL_AUTH_ENABLED
)
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import hashlib
import time
import logging
import uuid
import os
import tempfile
from datetime import datetime, timedelta
from psycopg2 import connect
from typing import List, Optional, Dict
from pydantic import BaseModel
from jose import JWTError, jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from unstructured.partition.auto import partition

# Azure AD SSO imports
from ai_analyzer.auth import azure_ad_validator, role_mapper
from ai_analyzer.auth.models import AzureADUser, AzureADGroupMapping, CREATE_TABLES_SQL


# JWT Configuration
SECRET_KEY = JWT_SECRET_KEY
ALGORITHM = JWT_ALGORITHM

# Security scheme for token authenticatio
security = HTTPBearer()

# Token models


class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    role: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a new JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
    logger = logging.getLogger(__name__)
    logger.info(f"get_current_user: Received credentials: {credentials}")
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        logger.info(f"get_current_user: Decoding token: {token}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        logger.info(f"get_current_user: Token payload: {payload}")
        if username is None:
            logger.warning(
                "get_current_user: Username missing in token payload")
            raise credentials_exception
        token_data = TokenData(
            username=username,
            role=payload.get("role")
        )
    except JWTError as e:
        logger.error(f"get_current_user: JWTError: {e}")
        raise credentials_exception

    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        logger.warning(
            f"get_current_user: No user found for username: {token_data.username}")
        raise credentials_exception
    logger.info(
        f"get_current_user: Authenticated user: {user.username}, role: {user.role}")
    return user


# Update internal imports

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,  # Increased pool size for more concurrent connections
    max_overflow=30,  # Increased overflow for peak loads
    pool_timeout=30,  # Timeout for getting a connection from the pool
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True,  # Enable connection health checks
    isolation_level="READ COMMITTED"  # Better concurrency handling
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_restricted_tables(db_session):
    """Get list of restricted table names from database"""
    try:
        result = db_session.execute(text("""
            SELECT table_name 
            FROM restricted_tables 
            WHERE added_by = 'system'
        """))
        return [row[0] for row in result]
    except Exception as e:
        logger.error(f"Error fetching restricted tables: {e}")
        # Fallback to minimum set of restricted tables
        return ['users', 'user_memory', 'query_cache', 'query_performance']


def get_initial_restricted_tables():
    """Get initial restricted tables with proper session handling"""
    db = SessionLocal()
    try:
        return get_restricted_tables(db)
    finally:
        db.close()


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize cache manager once


def get_cache_manager():
    """Get a properly initialized cache manager with a session"""
    db = SessionLocal()
    return DatabaseCacheManager(db)


# Initialize cache manager with a session instead of engine
cache_manager = get_cache_manager()

# Get initial restricted tables
restricted_tables = get_initial_restricted_tables()

# Initialize recipe search agent
recipe_search_agent = None
data_extractor_router_agent = None


def initialize_recipe_search_agent():
    """Initialize the recipe search agent"""
    global recipe_search_agent
    try:
        logger.info("Initializing recipe search agent...")
        recipe_search_agent = RecipeSearchAgent()
        logger.info("Recipe search agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing recipe search agent: {e}")
        logger.exception("Detailed error:")
        return False


def initialize_data_extractor_router_agent():
    """Initialize the data extractor router agent"""
    global data_extractor_router_agent
    try:
        logger.info("Initializing data extractor router agent...")
        data_extractor_router_agent = DataExtractorRouterAgent()
        logger.info("Data extractor router agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing data extractor router agent: {e}")
        logger.exception("Detailed error:")
        return False


# Recipe Search Request Model
class RecipeSearchRequest(BaseModel):
    """Request model for recipe search"""
    description: str
    # Optional feature refinement
    features: Optional[List[Dict[str, str]]] = None
    text_top_k: int = 20  # Number of candidates from text search
    final_top_k: int = 3  # Final number of results

    class Config:
        json_schema_extra = {
            "example": {
                "description": "Fruit: Yellow peach particulates, max 12mm in size. Can also use a peach puree. Apricot Puree. Fruit content to be >30%. Flavour profile: Balanced peach and apricot flavours.",
                "features": [
                    {"charactDescr": "Puree/with pieces", "valueCharLong": "puree"},
                    {"charactDescr": "Industry (SD Reporting)",
                     "valueCharLong": "Dairy"}
                ],
                "text_top_k": 20,
                "final_top_k": 3
            }
        }


def wait_for_db(max_retries=10, delay=10):
    """Wait for database to be ready with better error handling"""
    logger = logging.getLogger(__name__)

    for i in range(max_retries):
        try:
            # Try direct PostgreSQL connection first
            conn = connect(
                dbname=config['POSTGRES_DB'],
                user=config['POSTGRES_USER'],
                password=config['POSTGRES_PASSWORD'],
                host=config['DB_HOST'],
                port=config['DB_PORT']
            )
            conn.close()
            logger.info("Direct PostgreSQL connection successful")

            # Then try SQLAlchemy connection
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            logger.info("SQLAlchemy connection successful")

            return True
        except Exception as e:
            logger.error(f"Database connection attempt {
                         i + 1}/{max_retries} failed: {str(e)}")
            if i < max_retries - 1:  # Don't sleep on the last attempt
                time.sleep(delay)
    return False


# Create tables
Base.metadata.create_all(bind=engine)

# Hashing passwords


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def generate_title(query: str) -> str:
    """Generate a title from the first query of conversation"""
    return query[:50] + "..." if len(query) > 50 else query

# Add startup event to handle initial data loading


@app.on_event("startup")
async def startup_event():
    """Startup event handler with proper error handling"""
    try:
        # Wait for database with more retries
        if not wait_for_db(max_retries=10, delay=10):
            logger.error("Failed to connect to database during startup")
            return

        # Set up database session
        db = SessionLocal()
        try:
            # Simple connection test
            db.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
        finally:
            db.close()

        # Initialize recipe search agent
        initialize_recipe_search_agent()

        # Initialize data extractor router agent
        initialize_data_extractor_router_agent()

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error(f"Error during startup: {e}")


# Health check endpoint with JWT authentication
@app.get("/health")
async def health_check(current_user: User = Depends(get_current_user)):
    try:
        # Check database connection
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        logger.info("PostgreSQL connection successful")

        # Check SQLAlchemy connection
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("SQLAlchemy connection successful")

        # Check recipe search service
        global recipe_search_agent
        recipe_status = "available" if recipe_search_agent else "unavailable"

        return {
            "status": "healthy",
            "database": "connected",
            "user": current_user.username,
            "recipe_service": recipe_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.error("Stack trace:", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# API to handle login
@app.post("/api/login")
async def login(request: Request, db: Session = Depends(get_db)):
    # Check if local authentication is enabled
    if not LOCAL_AUTH_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="Local authentication is disabled. Please use SSO."
        )

    try:
        data = await request.json()
        username = data['username']
        password = data['password']

        # Find user
        user = db.query(User).filter(User.username == username).first()

        if not user:
            logger.warning(f"Login attempt with invalid username: {username}")
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )

        # Verify password
        if user.password_hash != hash_password(password):
            logger.warning(
                f"Login attempt with invalid password for user: {username}")
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )

        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={
                "sub": user.username,
                "role": user.role
            },
            expires_delta=access_token_expires
        )

        # If we get here, authentication is successful
        logger.info(
            f"Successful login for user {username}")
        return {
            "success": True,
            "access_token": access_token,
            "token_type": "bearer",
            "username": user.username,
            "role": user.role,
            "permissions": {
                "canWrite": user.has_write_permission()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Invalid credentials")


# ============================================================================
# Azure AD SSO Endpoints
# ============================================================================

class AzureAuthRequest(BaseModel):
    """Request model for Azure AD authentication"""
    id_token: str


class AzureAuthResponse(BaseModel):
    """Response model for Azure AD authentication"""
    success: bool
    access_token: str
    token_type: str = "bearer"
    username: str
    email: str
    role: str
    permissions: dict
    auth_method: str = "azure_ad"


@app.get("/api/auth/config")
async def get_auth_config(request: Request):
    """Return authentication configuration for frontend"""
    base_url = str(request.base_url).rstrip('/')

    return {
        "sso_enabled": SSO_ENABLED,
        "local_auth_enabled": LOCAL_AUTH_ENABLED,
        "azure_ad": {
            "client_id": AZURE_AD_CLIENT_ID,
            "authority": f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}" if AZURE_AD_TENANT_ID else None,
            "redirect_uri": f"{base_url}/auth/callback",
            "scopes": ["openid", "profile", "email", "User.Read"]
        } if SSO_ENABLED and AZURE_AD_TENANT_ID and AZURE_AD_CLIENT_ID else None
    }


@app.post("/api/auth/azure-callback", response_model=AzureAuthResponse)
async def azure_ad_callback(
    auth_request: AzureAuthRequest,
    db: Session = Depends(get_db)
):
    """
    Handle Azure AD authentication callback.
    Validates Azure AD token and issues internal JWT.
    """
    if not SSO_ENABLED:
        raise HTTPException(status_code=400, detail="SSO is not enabled")

    if not azure_ad_validator.is_configured():
        raise HTTPException(
            status_code=500,
            detail="Azure AD is not properly configured"
        )

    logger.info("Processing Azure AD authentication callback")

    # Validate Azure AD token
    claims = await azure_ad_validator.validate_token(auth_request.id_token)

    if not claims:
        logger.warning("Azure AD token validation failed")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired Azure AD token"
        )

    # Extract user info
    user_info = azure_ad_validator.extract_user_info(claims)
    azure_oid = user_info['oid']
    email = user_info['email']
    display_name = user_info.get('name') or email
    groups = user_info['groups']

    if not azure_oid or not email:
        raise HTTPException(
            status_code=401,
            detail="Required claims missing from token"
        )

    logger.info(f"Azure AD user authenticated: {email}, groups: {len(groups)}")

    # Map groups to application role
    app_role = role_mapper.map_groups_to_role(db, groups)

    # Find or create Azure AD user record
    azure_user = db.query(AzureADUser).filter_by(azure_oid=azure_oid).first()

    if not azure_user:
        # Create new Azure AD user
        azure_user = AzureADUser(
            azure_oid=azure_oid,
            email=email,
            display_name=display_name,
            is_active=True
        )
        db.add(azure_user)
        logger.info(f"Created new Azure AD user record for: {email}")
    else:
        # Update last login
        azure_user.last_login = datetime.utcnow()
        azure_user.display_name = display_name

    # Find or create linked local user
    local_user = db.query(User).filter_by(username=email).first()

    if not local_user:
        # Create local user linked to Azure AD
        local_user = User(
            username=email,
            password_hash="AZURE_AD_SSO_USER",  # Marker for SSO users
            role=app_role
        )
        db.add(local_user)
        db.flush()  # Get the ID
        azure_user.local_user_id = local_user.id
        logger.info(f"Created local user for Azure AD user: {email}")
    else:
        # Update role if changed based on group membership
        if local_user.role != app_role:
            logger.info(f"Updating role for {email}: {local_user.role} -> {app_role}")
            local_user.role = app_role
        azure_user.local_user_id = local_user.id

    db.commit()

    # Create internal JWT
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": email,
            "role": app_role,
            "auth_method": "azure_ad",
            "azure_oid": azure_oid
        },
        expires_delta=access_token_expires
    )

    logger.info(f"Issued internal JWT for Azure AD user: {email}")

    return AzureAuthResponse(
        success=True,
        access_token=access_token,
        username=display_name,
        email=email,
        role=app_role,
        permissions={
            "canWrite": app_role in ['admin', 'write']
        }
    )


@app.get("/api/admin/group-mappings")
async def get_group_mappings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all Azure AD group mappings (admin only)"""
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    return role_mapper.get_all_mappings(db)


@app.post("/api/admin/group-mappings")
async def create_group_mapping(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create or update a group mapping (admin only)"""
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    data = await request.json()

    # Validate required fields
    required_fields = ['group_id', 'group_name', 'app_role']
    for field in required_fields:
        if field not in data:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field: {field}"
            )

    try:
        mapping = role_mapper.add_group_mapping(
            db,
            group_id=data['group_id'],
            group_name=data['group_name'],
            app_role=data['app_role']
        )
        return {
            "success": True,
            "mapping": {
                "group_id": mapping.group_id,
                "group_name": mapping.group_name,
                "app_role": mapping.app_role
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/admin/group-mappings/{group_id}")
async def delete_group_mapping(
    group_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a group mapping (admin only)"""
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    success = role_mapper.delete_group_mapping(db, group_id)
    if not success:
        raise HTTPException(status_code=404, detail="Group mapping not found")

    return {"success": True}


# ============================================================================
# End Azure AD SSO Endpoints
# ============================================================================


# Update the QueryRequest model to handle both 'query' and 'question'
class QueryRequest(BaseModel):
    # Allow either 'query' or 'question' field
    query: str | None = None
    question: str | None = None
    conversation_id: str | None = None

    # Add validation to ensure at least one of query/question is present
    @property
    def get_question(self) -> str:
        return self.question or self.query or ""

    # Add validation
    def model_post_init(self, _):
        if not self.question and not self.query:
            raise ValueError("Either 'question' or 'query' field is required")
        # If question is not set but query is, use query as question
        if not self.question and self.query:
            self.question = self.query

    # Add example for documentation
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the most common topics?",  # Old format
                "question": "What are the most common topics?",  # New format
                "conversation_id": "optional-uuid-here"
            }
        }


@app.post("/api/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and extract text from a document (PDF, Word, PowerPoint, Images, etc.)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Document upload from user: {current_user.username}")

    try:
        # Validate file type
        allowed_extensions = {
            '.pdf', '.doc', '.docx', '.ppt', '.pptx',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg',
            '.html', '.htm', '.txt', '.rtf', '.odt'
        }

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Read and write file content
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            logger.info(f"Processing document: {file.filename}")

            # Use unstructured to extract text
            elements = partition(filename=temp_file_path)
            extracted_text = "\n\n".join(
                [str(element) for element in elements])

            logger.info(
                f"Successfully extracted {len(extracted_text)} characters from {file.filename}")

            return {
                "success": True,
                "filename": file.filename,
                "extracted_text": extracted_text,
                "text_length": len(extracted_text)
            }

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document upload: {e}")
        logger.exception("Detailed error:")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@app.post("/api/query")
async def process_query(
    request: Request,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db)
):
    logger = logging.getLogger(__name__)
    logger.info(
        f"/api/query: Received request from user: {getattr(current_user, 'username', None)} role: {getattr(current_user, 'role', None)}")
    try:
        # Parse request body
        body = await request.json()
        logger.info(f"/api/query: Request body: {body}")

        # Extract query parameters
        query = body.get("query", "")
        conversation_id = body.get("conversation_id", "")
        features = body.get("features", None)
        text_top_k = body.get("text_top_k", 20)
        final_top_k = body.get("final_top_k", 3)
        country_filter_raw = body.get("country_filter", None)
        version_filter = body.get("version_filter", None)

        # Handle country_filter: support both single string (backward compatibility) and list
        country_filter = None
        if country_filter_raw is not None:
            if isinstance(country_filter_raw, list):
                # Multiple countries selected
                # Filter out "All" from the list - if "All" is present, it means no filtering
                filtered_countries = [c for c in country_filter_raw if c != "All"]
                if len(filtered_countries) > 0:
                    country_filter = filtered_countries
                    logger.info(f"/api/query: Multiple countries filter received: {country_filter} (count: {len(country_filter)})")
                elif len(country_filter_raw) > 0 and "All" in country_filter_raw:
                    # List contains only "All" or "All" was filtered out
                    logger.info("/api/query: Country filter contains 'All', no country filter applied (searching all countries)")
                else:
                    logger.info("/api/query: Empty country filter list received, treating as 'All Countries'")
            elif isinstance(country_filter_raw, str):
                # Single country (backward compatibility)
                if country_filter_raw and country_filter_raw != "All":
                    country_filter = country_filter_raw
                    logger.info(f"/api/query: Single country filter received: {country_filter}")
                else:
                    logger.info("/api/query: Country filter is 'All' or empty, no filter applied")
            else:
                logger.warning(f"/api/query: Unexpected country_filter type: {type(country_filter_raw)}, value: {country_filter_raw}")

        # Validate input
        if not query:
            logger.warning("/api/query: Query is missing in request body")
            raise HTTPException(status_code=400, detail="Query is required")

        logger.info(f"/api/query: Processing recipe search query: '{query}'")
        if conversation_id:
            logger.info(
                f"/api/query: Continuing conversation: {conversation_id}")

        # Check if recipe search agent is available
        global recipe_search_agent, data_extractor_router_agent
        if not recipe_search_agent:
            logger.error("Recipe search agent not initialized")
            raise HTTPException(
                status_code=503,
                detail="Recipe search service not available. Please try again later."
            )

        if not data_extractor_router_agent:
            logger.error("Data extractor router agent not initialized")
            raise HTTPException(
                status_code=503,
                detail="Data extraction service not available. Please try again later."
            )

        # Extract features from the query using the data extractor router agent
        logger.info("/api/query: Extracting features from query...")
        extraction_result = data_extractor_router_agent.extract_and_route(
            query)

        # Use extracted features for the search
        extracted_features_df = extraction_result.get('features_df')
        text_description = extraction_result.get('text_description', query)

        logger.info(
            f"/api/query: Search type: {extraction_result.get('search_type')}")
        logger.info(
            f"/api/query: Reasoning: {extraction_result.get('reasoning')}")

        # Detailed logging for debugging
        if extracted_features_df is not None and not extracted_features_df.empty:
            features_list = []
            for idx, row in extracted_features_df.iterrows():
                feature_str = f"{row.get('charactDescr', 'N/A')}: {row.get('valueCharLong', 'N/A')}"
                features_list.append(feature_str)
            logger.info(
                f"/api/query: Extracted {len(extracted_features_df)} features: {', '.join(features_list)}")
        else:
            logger.info("/api/query: Extracted 0 features")

        logger.info(f"/api/query: Extracted description: {text_description}")

        # Extract numerical filters for range queries (e.g., Brix > 40, pH < 4.1)
        numerical_filters = extraction_result.get('numerical_filters', {})
        if numerical_filters:
            logger.info(f"/api/query: Extracted {len(numerical_filters)} numerical constraint(s):")
            for field_code, range_spec in numerical_filters.items():
                logger.info(f"/api/query:   - {field_code}: {range_spec}")

        # Extract categorical filters for exact-match queries (e.g., Preservative: No, Halal: Yes)
        categorical_filters = extraction_result.get('categorical_filters', {})
        if categorical_filters:
            logger.info(f"/api/query: Extracted {len(categorical_filters)} categorical constraint(s):")
            for field_code, match_spec in categorical_filters.items():
                logger.info(f"/api/query:   - {field_code}: {match_spec}")

        # Search for recipes with extracted features
        # Pass original query for language detection (the extracted description may be in English)
        if isinstance(country_filter, list):
            logger.info(f"/api/query: Applying multi-country filter: {country_filter} ({len(country_filter)} countries)")
        elif country_filter:
            logger.info(f"/api/query: Applying single country filter: {country_filter}")
        else:
            logger.info("/api/query: No country filter applied (searching all countries)")
        logger.info(f"/api/query: Version filter: {version_filter}")
        results, metadata, formatted_response, detected_language, comparison_table = recipe_search_agent.search_recipes(
            description=text_description,
            features=extracted_features_df,
            text_top_k=text_top_k,
            final_top_k=final_top_k,
            original_query=query,
            country_filter=country_filter,
            version_filter=version_filter,
            numerical_filters=numerical_filters,
            categorical_filters=categorical_filters
        )

        # Use the formatted response from the agent
        response = formatted_response

        # Generate follow-up questions in the detected language
        followup_questions = recipe_search_agent.generate_followup_questions(
            results, query, detected_language)

        # Store conversation in database using user_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        try:
            logger.info(
                f"/api/query: Storing conversation with ID: {conversation_id}")
            store_conversation(
                db_session,
                str(current_user.id),  # Convert user ID to string
                conversation_id,
                query,
                response,
                followup_questions=followup_questions
            )
            logger.info("/api/query: Conversation stored successfully")
        except Exception as e:
            logger.error(f"/api/query: Error storing conversation: {e}")
            logger.exception("/api/query: Detailed error:")
            # Continue even if storage fails

        # Return a response with the exact fields the frontend expects
        logger.info(
            f"/api/query: Returning response for conversation_id: {conversation_id}")
        
        # Debug: Log comparison table structure
        if comparison_table:
            logger.info(f"/api/query: comparison_table has_data: {comparison_table.get('has_data')}")
            logger.info(f"/api/query: comparison_table has field_definitions: {'field_definitions' in comparison_table}")
            if 'field_definitions' in comparison_table:
                logger.info(f"/api/query: field_definitions count: {len(comparison_table.get('field_definitions', []))}")
            logger.info(f"/api/query: comparison_table recipes count: {len(comparison_table.get('recipes', []))}")
        
        return {
            "response": response,
            "conversation_id": conversation_id,
            "followup_questions": followup_questions,
            "search_results": results,  # Include raw results for frontend
            "metadata": metadata,
            "comparison_table": comparison_table  # Include comparison table
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/api/query: Error processing query: {e}")
        logger.exception("/api/query: Detailed error:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recipe-service-status")
async def get_recipe_service_status(current_user: User = Depends(get_current_user)):
    """Get the status of the recipe search service"""
    global recipe_search_agent

    if not recipe_search_agent:
        return {
            "status": "unavailable",
            "message": "Recipe service not initialized",
            "total_recipes": 0
        }

    try:
        return recipe_search_agent.get_service_status()
    except Exception as e:
        logger.error(f"Error getting recipe service status: {e}")
        return {
            "status": "error",
            "message": f"Error getting service status: {str(e)}",
            "total_recipes": 0
        }


@app.post("/api/recipe-search")
async def search_recipes(
    request: RecipeSearchRequest,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db)
):
    """Search for similar recipes based on description and optional features"""
    logger = logging.getLogger(__name__)
    logger.info(f"Recipe search request from user: {current_user.username}")

    try:
        global recipe_search_agent

        # Check if recipe manager is initialized
        if not recipe_search_agent:
            logger.error("Recipe search agent not initialized")
            raise HTTPException(
                status_code=503,
                detail="Recipe search service not available. Please try again later."
            )

        # Validate input
        if not request.description.strip():
            raise HTTPException(
                status_code=400,
                detail="Recipe description is required"
            )

        logger.info(
            f"Searching recipes for description: '{request.description[:100]}...'")

        # Search for recipes
        # Pass original description for language detection
        results, metadata, formatted_response, detected_language, comparison_table = recipe_search_agent.search_recipes(
            description=request.description,
            features=request.features,
            text_top_k=request.text_top_k,
            final_top_k=request.final_top_k,
            original_query=request.description
        )

        if not results:
            logger.info("No recipes found matching the description")
            return {
                "success": True,
                "message": formatted_response,
                "results": [],
                "metadata": metadata
            }

        # Format results for API response
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = {
                "rank": i,
                "recipe_id": result.get("id", f"recipe_{i}"),
                "description": result.get("description", ""),
                "text_score": round(result.get("text_score", 0), 4),
                "feature_score": round(result.get("feature_score", 0), 4) if result.get("feature_score") else None,
                "combined_score": round(result.get("combined_score", result.get("text_score", 0)), 4),
                "features": result.get("features", []),
                "values": result.get("values", []),
                "feature_text": result.get("feature_text", ""),
                "metadata": result.get("metadata", {})
            }
            formatted_results.append(formatted_result)

        logger.info(f"Found {len(formatted_results)} recipes")

        return {
            "success": True,
            "message": formatted_response,
            "results": formatted_results,
            "metadata": {
                "search_type": metadata.get("search_type", "two_step"),
                "text_candidates": metadata.get("text_candidates", request.text_top_k),
                "final_results": metadata.get("final_results_count", len(formatted_results)),
                "feature_refinement": metadata.get("refinement_completed", False),
                "total_recipes_searched": recipe_search_agent.recipe_manager.get_stats().get("total_recipes", 0) if recipe_search_agent.recipe_manager else 0
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recipe search: {e}")
        logger.exception("Detailed error:")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching recipes: {str(e)}"
        )


@app.get("/api/conversations")
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(
        f"get_conversations: Endpoint called with user: {current_user.username if current_user else 'None'}")
    try:
        logger.info(f"Getting conversations for user: {current_user.username}")

        # Get distinct conversation IDs with their first message (for title)
        # First, get all distinct conversation IDs
        try:
            distinct_conversations = db.query(
                UserMemory.conversation_id,
                func.min(UserMemory.message_order).label(
                    'first_message_order'),
                func.max(UserMemory.timestamp).label('latest_timestamp')
            ).filter(
                UserMemory.is_active,
                UserMemory.expires_at > datetime.utcnow(),
                UserMemory.user_id == str(current_user.id)
            ).group_by(
                UserMemory.conversation_id
            ).order_by(
                func.max(UserMemory.timestamp).desc()
            ).limit(10).all()
        except Exception as db_error:
            logger.error(f"Database query error in conversations: {db_error}")
            logger.exception("Database error details:")
            # Return empty list if there's a database error
            return []

        logger.info(
            f"Found {len(distinct_conversations)} conversations for user {current_user.username}")

        # Now get the first message of each conversation for the title
        result = []
        for conv_id, first_order, latest_ts in distinct_conversations:
            # Get the first message (for title)
            first_message = db.query(UserMemory).filter(
                UserMemory.conversation_id == conv_id,
                UserMemory.message_order == first_order,
                UserMemory.user_id == str(current_user.id)
            ).first()

            if first_message:
                result.append({
                    "id": conv_id,
                    "title": first_message.title or first_message.query[:50],
                    "timestamp": latest_ts.isoformat()  # Use the latest timestamp for sorting
                })

        logger.info(f"Returning {len(result)} conversations")
        return result
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}")
        logger.exception("Detailed error:")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve conversations"
        )


@app.get("/api/initial-questions")
async def get_initial_questions(
    request: Request,
    current_user: User = Depends(get_current_user),
    transcription_id: Optional[str] = None,
    db_session: Session = Depends(get_db)
):
    """Generate initial questions for recipe search"""
    try:
        # Create a hardcoded response with categories structured exactly as the frontend expects
        categories = {
            "Recipe Categories": {
                "description": "Explore different types of recipes",
                "questions": [
                    "What fruit-based recipes do you have?",
                    "Show me dairy product recipes",
                    "What dessert recipes are available?",
                    "Do you have any beverage recipes?",
                    "What savory snack recipes exist?"
                ]
            },
            "Ingredient Search": {
                "description": "Find recipes by specific ingredients",
                "questions": [
                    "Find recipes containing peach",
                    "Show me recipes with apricot puree",
                    "What recipes use strawberry?",
                    "Find recipes with banana",
                    "Show me recipes containing tropical fruits"
                ]
            },
            "Product Characteristics": {
                "description": "Search by product features and properties",
                "questions": [
                    "Find recipes with puree texture",
                    "Show me recipes with pieces",
                    "What recipes are GMO-free?",
                    "Find organic recipe options",
                    "Show me recipes with natural flavors"
                ]
            },
            "Industry Applications": {
                "description": "Explore recipes by industry use",
                "questions": [
                    "What recipes are for dairy industry?",
                    "Show me food service recipes",
                    "Find retail product recipes",
                    "What industrial recipes exist?",
                    "Show me consumer product recipes"
                ]
            },
            "Flavor Profiles": {
                "description": "Discover recipes by taste characteristics",
                "questions": [
                    "Find recipes with balanced flavors",
                    "Show me sweet recipe options",
                    "What tart recipes are available?",
                    "Find recipes with tropical flavors",
                    "Show me recipes with berry flavors"
                ]
            },
            "Dietary Requirements": {
                "description": "Find recipes meeting specific dietary needs",
                "questions": [
                    "What sugar-free recipes exist?",
                    "Show me low-calorie options",
                    "Find allergen-free recipes",
                    "What vegan recipes are available?",
                    "Show me gluten-free recipes"
                ]
            }
        }

        # Return in the exact format the frontend expects
        return {
            "success": True,
            "categories": categories
        }
    except Exception as e:
        logger.error(f"Error generating initial questions: {e}")
        logger.exception("Detailed error:")

        # Return a fallback response with the same structure
        fallback_categories = {
            "General Recipe Search": {
                "description": "Basic recipe search questions",
                "questions": [
                    "What recipes do you have available?",
                    "Show me fruit-based recipes",
                    "Find recipes with specific ingredients"
                ]
            }
        }

        return {
            "success": False,
            "categories": fallback_categories
        }


@app.post("/api/generate-followup")
async def generate_followup(
    request: Request,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db)
):
    """Generate follow-up questions based on previous conversation"""
    try:
        data = await request.json()

        # Extract parameters
        questions = data.get("questions", [])
        responses = data.get("responses", [])

        if not questions or not responses:
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters: questions, responses"
            )

        # Generate follow-up questions based on recipe search context
        followup_questions = [
            "Would you like to refine your search with specific features?",
            "Are you looking for recipes with similar ingredients?",
            "Do you want to see more recipes in this category?",
            "Would you like to filter by any specific characteristics?"
        ]

        return {
            "success": True,
            "followup_questions": followup_questions
        }
    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        logger.exception("Detailed error:")
        return {
            "success": False,
            "error": str(e),
            "followup_questions": [
                "What type of recipe are you looking for?",
                "Do you have any specific ingredients or dietary requirements?",
                "What cuisine style interests you?"
            ]
        }


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(
            f"Getting conversation {conversation_id} for user: {current_user.username}")

        # Get messages for conversation with user isolation
        messages = db.query(UserMemory)\
            .filter(
                UserMemory.conversation_id == conversation_id,
                UserMemory.user_id == str(
                    current_user.id),  # Add user isolation
                UserMemory.is_active
        )\
            .order_by(UserMemory.message_order.asc())\
            .all()

        if not messages:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found or access denied"
            )

        # Process messages to make errors user-friendly
        processed_messages = []
        for msg in messages:
            # Create a copy of the message data
            message_data = {
                "id": msg.id,
                "conversation_id": msg.conversation_id,
                "query": msg.query,
                "timestamp": msg.timestamp.isoformat(),
                "followup_questions": msg.followup_questions
            }

            # Check if this is an error response
            if msg.response and msg.response.startswith("Error") or "error" in msg.response.lower():
                # Detect language (Dutch vs English)
                is_dutch = any(dutch_word in msg.query.lower() for dutch_word in
                               ['wat', 'hoe', 'waarom', 'welke', 'kunnen', 'waar', 'wie', 'wanneer', 'onderwerp'])

                # Replace error message with user-friendly message
                if is_dutch:
                    message_data["response"] = "Er was een probleem bij het beantwoorden van deze vraag. Probeer het opnieuw of stel een andere vraag."
                else:
                    message_data["response"] = "There was an issue answering this question. Please try again or ask a different question."
            else:
                # Keep the original response
                message_data["response"] = msg.response

            processed_messages.append(message_data)

        return processed_messages

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve conversation"
        )


# Optional: API to add a new user
@app.post("/api/add_user")
async def add_user(request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Check if current user has admin privileges
    if not current_user.has_write_permission():
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    data = await request.json()
    username = data['username']
    password = data['password']
    # Default to read_only if not specified
    role = data.get('role', 'read_only')

    # Validate role
    if role not in ['admin', 'write', 'read_only']:
        raise HTTPException(status_code=400, detail="Invalid role specified")

    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(
        username=username,
        password_hash=hash_password(password),
        role=role
    )
    db.add(new_user)
    db.commit()
    return {"success": True, "message": "User added successfully"}


# Add this function to create conversation tables
def create_conversation_tables(engine):
    """Create tables for storing conversations if they don't exist"""
    try:
        # Create conversations table
        engine.execute(text("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            title TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """))

        # Create conversation_messages table
        engine.execute(text("""
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id SERIAL PRIMARY KEY,
            conversation_id VARCHAR(255) NOT NULL,
            message_order INTEGER NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            followup_questions JSONB,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )
        """))

        logger.info("Conversation tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating conversation tables: {e}")
        logger.exception("Detailed error:")
        return False


@app.post("/api/refresh-token")
async def refresh_token(request: Request, db: Session = Depends(get_db)):
    try:
        auth_header = request.headers.get('authorization')
        if not auth_header or not auth_header.lower().startswith('bearer '):
            raise HTTPException(
                status_code=401, detail="Missing or invalid authorization header")
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            role = payload.get("role")
            if not username or not role:
                raise HTTPException(
                    status_code=401, detail="Invalid token payload")
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
        # Optionally, check user still exists and is active
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        # Issue new token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_token = create_access_token(
            data={
                "sub": user.username,
                "role": user.role
            },
            expires_delta=access_token_expires
        )
        return {
            "access_token": new_token,
            "token_type": "bearer",
            "expires_in": access_token_expires.total_seconds()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refresh token error: {e}")
        raise HTTPException(status_code=401, detail="Could not refresh token")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
