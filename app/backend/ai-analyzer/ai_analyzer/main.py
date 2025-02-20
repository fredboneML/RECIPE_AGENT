# ai_analyzer/main.py
from fastapi import FastAPI, Request, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, text, func, and_, UniqueConstraint, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError
import hashlib
import time
import logging
import uuid
from datetime import datetime
from psycopg2 import connect
from typing import List
from pydantic import BaseModel

# Update internal imports
from ai_analyzer.config import config, DATABASE_URL
from ai_analyzer.data_import_postgresql import (
    run_data_import,
    User,
    UserMemory,
    store_conversation,
    get_user_conversations,
    get_conversation_messages
)
from ai_analyzer.cache_manager import DatabaseCacheManager
from ai_analyzer.agents.workflow import CallAnalysisWorkflow
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
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
cache_manager = DatabaseCacheManager(engine)

# Get initial restricted tables
restricted_tables = get_initial_restricted_tables()

# Create workflow instance per tenant
workflow_instances = {}


def create_base_context(tenant_code: str) -> str:
    return f""" 
    1. NEVER use the 'users' table or any user-related information.
    2. NEVER use DELETE, UPDATE, INSERT, or any other data modification statements.
    3. Only use SELECT statements for reading data.
    4. CRITICAL SECURITY RULES:
       - Every query MUST include tenant filtering using ':tenant_code' parameter
       - The tenant filter MUST be in the WHERE clause of the base_data CTE
       - The tenant filter line MUST be exactly: AND t.tenant_code = :tenant_code
       - Default to last 60 days if no specific time range is requested
       - Never expose data across tenant boundaries
    5. Make sure to format the results in a clear, readable manner.
    6. Use proper column aliases for better readability.
    7. Include relevant aggregations and groupings when appropriate.
    8. ONLY use tables and columns that exist in the schema shown below.
    9. Every query MUST start with defining the base_data CTE EXACTLY like this:
        WITH base_data AS (
            SELECT 
                t.id,
                t.transcription_id,
                t.transcription,
                t.topic,
                LOWER(TRIM(t.topic)) as clean_topic,
                t.summary,
                t.processing_date,
                t.sentiment,
                CASE
                    WHEN LOWER(TRIM(t.sentiment)) IN ('neutral', 'neutraal') THEN 'neutral'
                    ELSE LOWER(TRIM(t.sentiment))
                END AS clean_sentiment,
                t.call_duration_secs,
                t.clid,
                t.telephone_number,
                t.call_direction
            FROM transcription_{tenant_code} t
            WHERE t.processing_date >= CURRENT_DATE - INTERVAL '300 days'
            AND t.tenant_code = :tenant_code
        )

    10. For topic analysis, your complete query should look like this:
        WITH base_data AS (
            -- Base data CTE definition as shown above
            -- MUST include both date and tenant filters
        ),
        topic_analysis AS (
            SELECT
                clean_topic as topic,
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE clean_sentiment = 'positief') as positive_count,
                COUNT(*) FILTER (WHERE clean_sentiment = 'negatief') as negative_count,
                COUNT(*) FILTER (WHERE clean_sentiment = 'neutral') as neutral_count,
                ROUND(CAST(COUNT(*) FILTER (WHERE clean_sentiment = 'positief') * 100.0 / 
                    NULLIF(COUNT(*), 0) AS NUMERIC), 2) as satisfaction_rate
            FROM base_data
            GROUP BY clean_topic
            HAVING COUNT(*) > 0
        )
        SELECT * FROM topic_analysis ...

    11. For time-based analysis, your complete query should look like this:
        WITH base_data AS (
            -- Base data CTE definition as shown above
        ),
        time_based_data AS (
            SELECT 
                id,
                transcription_id,
                clean_topic,
                clean_sentiment,
                clid,
                processing_date,
                CASE
                    WHEN processing_date >= CURRENT_DATE - INTERVAL '7 days' THEN 'Current Week'
                    WHEN processing_date >= CURRENT_DATE - INTERVAL '14 days' THEN 'Previous Week'
                    WHEN processing_date >= CURRENT_DATE - INTERVAL '30 days' THEN 'Current Month'
                    WHEN processing_date >= CURRENT_DATE - INTERVAL '60 days' THEN 'Previous Month'
                END AS time_period
            FROM base_data
            WHERE processing_date >= CURRENT_DATE - INTERVAL '300 days'
        )
        SELECT * FROM time_based_data ...

    12. For text comparisons, ALWAYS use these patterns:
        - Exact match: clean_topic = 'value' or clean_sentiment = 'value'
        - Partial match: clean_topic LIKE '%value%'

    13. For calculations ALWAYS use:
        - NULLIF(value, 0) for division
        - COALESCE(value, 0) for NULL handling
        - ROUND(CAST(value AS NUMERIC), 2) for decimals

    14. For filtering dates ALWAYS use:
        - WHERE processing_date >= CURRENT_DATE - INTERVAL 'X days'

    15. For aggregations ALWAYS use:
        - COUNT(*) FILTER (WHERE condition) for conditional counting
        - SUM(CASE WHEN condition THEN 1 ELSE 0 END) for counting matches

    16. Never use:
        - Raw tables directly (always go through base_data)
        - Raw topic or sentiment columns (always use clean_topic and clean_sentiment)
        - Calculations without CAST and ROUND
        - Division without NULLIF
        - Date comparisons without INTERVAL
        - Any tenant-related columns or filters (handled automatically)

    17. ALWAYS include proper ordering:
        ORDER BY [columns] {{"ASC" | "DESC"}} NULLS LAST

    18. For limiting results:
        LIMIT [number]

    19. Make sure to answer the question using the same language used by the user to ask it.

    20. Focus on these types of analysis:
        - Topic trends and patterns
        - Sentiment analysis
        - Call duration statistics
        - Time-based patterns
        - Customer interaction analysis
        - Call direction analysis

    21. CRITICAL REMINDERS:
        - EVERY query MUST start with WITH base_data AS (...)
        - The base_data CTE MUST include 't.tenant_code = :tenant_code' in WHERE clause
        - NEVER modify or remove the tenant_code filter line
        - ALWAYS use the exact base_data CTE structure shown above
        - NEVER expose or filter by tenant-related information outside base_data
        - COPY and PASTE the exact base_data CTE structure shown above

    22. NEVER execute a DELETE, UPDATE, INSERT, DROP, or any other data modification statements.

    23. Always use the last 2 months as default value when generating the SQL query and only change it if required by the user.

    23. Generate questions only about:
        - Call topics and their trends
        - Sentiment patterns
        - Call durations and patterns
        - Time-based analysis
        - Customer interaction patterns
        - Call direction statistics
        
    24. NEVER generate questions about:
        - Tenant-specific data
        - User access or permissions
        - Data partitioning
        - System administration
    """


def get_workflow_for_tenant(tenant_code: str) -> CallAnalysisWorkflow:
    """Get or create workflow instance for tenant"""
    try:
        if tenant_code not in workflow_instances:
            logger.info(
                f"Creating new workflow instance for tenant: {tenant_code}")
            workflow_instances[tenant_code] = CallAnalysisWorkflow(
                db_url=DATABASE_URL,
                model_provider="openai",
                model_name=os.getenv('MODEL_NAME', 'gpt-4'),
                api_key=os.getenv('AI_ANALYZER_OPENAI_API_KEY'),
                restricted_tables=restricted_tables,
                base_context=create_base_context(tenant_code),
                cache_manager=cache_manager
            )
            logger.info(
                f"Successfully created workflow instance for tenant: {tenant_code}")
        else:
            logger.info(
                f"Using existing workflow instance for tenant: {tenant_code}")

        return workflow_instances[tenant_code]
    except Exception as e:
        logger.error(
            f"Error getting workflow for tenant {tenant_code}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing workflow for tenant {tenant_code}"
        )


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


# User model
# Remove the current User model in main.py and replace it with:

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default='read_only')
    tenant_code = Column(String, nullable=False)

    # Username should be unique per tenant
    __table_args__ = (
        UniqueConstraint('username', 'tenant_code',
                         name='uix_username_tenant'),
    )

    def has_write_permission(self):
        return self.role in ['admin', 'write']


# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get the database session


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Hashing passwords


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def store_conversation(session, user_id, conversation_id, query, response, message_order=None, followup_questions=None):
    """Store a conversation message with followup questions"""
    try:
        if message_order is None:
            # Get the last message order for this conversation
            last_message = session.query(UserMemory)\
                .filter(UserMemory.conversation_id == conversation_id)\
                .order_by(UserMemory.message_order.desc())\
                .first()
            message_order = (last_message.message_order +
                             1) if last_message else 0

        memory = UserMemory(
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
            response=response,
            title=generate_title(query),
            message_order=message_order,
            # Handle followup_questions
            followup_questions=followup_questions if followup_questions else []
        )
        session.add(memory)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error storing conversation: {e}")
        return False


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

        logger.info("Application startup completed successfully")

    except Exception as e:
        logger.error(f"Error during startup: {e}")


# Health check endpoint with detailed debugging
@app.get("/health")
async def health_check():
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
        logger.info("PostgreSQL connection successful")

        # Then try SQLAlchemy connection with proper text() wrapper
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        logger.info("SQLAlchemy connection successful")

        return {
            "status": "healthy",
            "database": {
                "host": config['DB_HOST'],
                "port": config['DB_PORT'],
                "name": config['POSTGRES_DB'],
                "user": config['POSTGRES_USER']
            }
        }
    except Exception as e:
        error_msg = f"Health check failed: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "unhealthy",
            "error": error_msg,
            "database_config": {
                "host": config['DB_HOST'],
                "port": config['DB_PORT'],
                "name": config['POSTGRES_DB'],
                "user": config['POSTGRES_USER']
            }
        }


# API to handle login
@app.post("/api/login")
async def login(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        username = data['username']
        password = data['password']

        # Find user and their tenant code
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

        # If we get here, authentication is successful
        logger.info(
            f"Successful login for user {username} with tenant {user.tenant_code}")
        return {
            "success": True,
            "role": user.role,
            "username": user.username,
            "tenant_code": user.tenant_code,  # Return tenant_code from user record
            "permissions": {
                "canWrite": user.has_write_permission()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Invalid credentials")


# Add this function near the top with other utility functions
def get_default_followup_questions() -> List[str]:
    """Get default followup questions when error occurs"""
    return [
        "What are the most common topics in our calls?",
        "How has customer sentiment changed over time?",
        "Can you show me the breakdown of call topics by sentiment?"
    ]


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


@app.post("/api/query")
async def process_query(
    request: Request,
    db: Session = Depends(get_db)
):
    """Process a user query"""
    try:
        # Get tenant code from headers
        tenant_code = request.headers.get('X-Tenant-Code')
        if not tenant_code:
            raise HTTPException(status_code=401, detail="Tenant code required")

        # Get or create workflow for tenant
        workflow = get_workflow_for_tenant(tenant_code)

        # Parse request body
        try:
            body = await request.json()
            # Validate with Pydantic model
            query_request = QueryRequest(**body)
        except Exception as e:
            logger.error(f"Error parsing request body: {e}")
            raise HTTPException(
                status_code=422,
                detail="Invalid request format. Required fields: either 'query' or 'question' (string), optional: 'conversation_id' (string)"
            )

        # Use validated request data
        question = query_request.get_question
        conversation_id = query_request.conversation_id or str(uuid.uuid4())

        logger.info(f"Processing query for tenant: {tenant_code}")
        logger.info(f"Question: {question}")
        logger.info(f"Conversation ID: {conversation_id}")

        # Process query through workflow
        result = await workflow.process_user_question(
            question=question,
            conversation_id=conversation_id,
            db_session=db,
            tenant_code=tenant_code
        )

        if not result['success'] and result.get('context_exceeded'):
            # Special handling for context length exceeded
            return {
                "success": False,
                "error": result['error'],
                "context_exceeded": True,
                "followup_questions": result['followup_questions']
            }

        logger.info(f"Workflow result: {result}")

        if result.get('success', False):
            # Store successful conversation
            store_success = store_conversation(
                session=db,
                user_id=tenant_code,  # Use tenant_code as user_id for now
                conversation_id=conversation_id,
                query=question,
                response=result['response'],
                message_order=None,
                followup_questions=result.get('followup_questions', [])
            )

            if not store_success:
                logger.warning("Failed to store conversation")

            return {
                "success": True,
                "result": result['response'],
                "conversation_id": conversation_id,
                "followup_questions": result.get('followup_questions', [])
            }
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            reformulated = result.get('reformulated_question', '')
            followup = result.get('followup_questions',
                                  get_default_followup_questions())

            # Store the error response
            store_conversation(
                session=db,
                user_id=tenant_code,  # Use tenant_code as user_id for now
                conversation_id=conversation_id,
                query=question,
                response=f"Error: {error_msg}\n" +
                (f"Try instead: {reformulated}" if reformulated else ""),
                message_order=None,
                followup_questions=followup
            )

            return {
                "success": False,
                "error": error_msg,
                "reformulated_question": reformulated,
                "followup_questions": followup,
                "conversation_id": conversation_id
            }

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "success": False,
            "error": str(e),
            "followup_questions": get_default_followup_questions()
        }


@app.get("/api/conversations")
async def get_conversations(request: Request, db: Session = Depends(get_db)):
    try:
        # Get tenant code from headers
        tenant_code = request.headers.get('X-Tenant-Code')
        if not tenant_code:
            raise HTTPException(status_code=401, detail="Tenant code required")

        # Get recent conversations for current tenant
        conversations = db.query(UserMemory)\
            .filter(
                UserMemory.is_active == True,
                UserMemory.expires_at > datetime.utcnow(),
                UserMemory.user_id == tenant_code  # Add tenant isolation
        )\
            .order_by(UserMemory.timestamp.desc())\
            .limit(10)\
            .all()

        return [
            {
                "id": conv.conversation_id,
                "title": conv.title,
                "timestamp": conv.timestamp.isoformat()
            }
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve conversations"
        )


@app.get("/api/initial-questions")
async def get_initial_questions(request: Request, db: Session = Depends(get_db)):
    """Get initial suggested questions"""
    try:
        # Get tenant code from headers
        tenant_code = request.headers.get('X-Tenant-Code')
        if not tenant_code:
            raise HTTPException(status_code=401, detail="Tenant code required")

        # Get or create workflow for tenant
        workflow = get_workflow_for_tenant(tenant_code)

        # Get initial questions from workflow
        initial_questions = await workflow.get_initial_questions()

        return {
            "success": True,
            "categories": initial_questions
        }
    except Exception as e:
        logger.error(f"Error getting initial questions: {e}")
        return {
            "success": False,
            "categories": {
                "General Analysis": {
                    "description": "Basic analysis questions",
                    "questions": [
                        "What are the most common topics in our calls?",
                        "How has customer sentiment changed over time?",
                        "Show me the breakdown of call topics by sentiment"
                    ]
                }
            }
        }


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        # Get tenant code from headers
        tenant_code = request.headers.get('X-Tenant-Code')
        if not tenant_code:
            raise HTTPException(status_code=401, detail="Tenant code required")

        # Get messages for conversation with tenant isolation
        messages = db.query(UserMemory)\
            .filter(
                UserMemory.conversation_id == conversation_id,
                UserMemory.user_id == tenant_code,  # Add tenant isolation
                UserMemory.is_active == True
        )\
            .order_by(UserMemory.timestamp.asc())\
            .all()

        if not messages:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found or access denied"
            )

        return [
            {
                "id": msg.id,
                "conversation_id": msg.conversation_id,
                "query": msg.query,
                "response": msg.response,
                "timestamp": msg.timestamp.isoformat(),
                "followup_questions": msg.followup_questions
            }
            for msg in messages
        ]

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
async def add_user(request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_db)):
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


@app.get("/api/conversations")
async def get_conversations(db: Session = Depends(get_db)):
    try:
        user_id = "current_user"  # Replace with actual user ID from auth
        conversations = get_user_conversations(db, user_id)
        return [{"id": conv.conversation_id,
                "title": conv.title,
                 "timestamp": conv.timestamp} for conv in conversations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    try:
        messages = get_conversation_messages(db, conversation_id)
        return [{"query": msg.query,
                "response": msg.response,
                 "timestamp": msg.timestamp} for msg in messages]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
