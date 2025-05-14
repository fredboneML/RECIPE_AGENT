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
from datetime import datetime, timezone, timedelta
from psycopg2 import connect
from typing import List
from pydantic import BaseModel
import psycopg2
import json
import os

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
from ai_analyzer.agents import AgentManager  # Import our new AgentManager
from ai_analyzer.agents.workflow import CallAnalysisWorkflow  # Add this import
from ai_analyzer.data_pipeline import get_db_session
from ai_analyzer.utils.singleton_resources import ResourceManager  # Fix the import path

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
    """Get or create a workflow instance for a tenant"""
    try:
        logger.info(f"Creating workflow instance for tenant: {tenant_code}")
        # Create a new workflow instance
        workflow = CallAnalysisWorkflow(tenant_code=tenant_code)
        logger.info(
            f"Successfully created workflow instance for tenant: {tenant_code}")
        return workflow
    except Exception as e:
        logger.error(f"Error creating workflow for tenant {tenant_code}: {e}")
        logger.exception("Detailed error:")
        # Return a basic workflow as fallback
        return CallAnalysisWorkflow(tenant_code=tenant_code)


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

        # Create a title from the query for new conversations
        title = query[:50] + "..." if len(query) > 50 else query

        # Store in UserMemory table
        memory = UserMemory(
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
            response=response,
            title=title if message_order == 0 else None,  # Only set title for first message
            message_order=message_order,
            # Handle followup_questions
            followup_questions=followup_questions if followup_questions else []
        )

        session.add(memory)
        session.commit()

        logger.info(
            f"Stored conversation in UserMemory: {conversation_id}, order: {message_order}")
        return True

    except Exception as e:
        logger.error(f"Error storing conversation: {e}")
        logger.exception("Detailed error:")
        session.rollback()
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

        # Check Qdrant connection
        resource_manager = ResourceManager()
        qdrant_client = resource_manager.get_qdrant_client()
        collections = qdrant_client.get_collections()
        logger.info("Qdrant connection successful")

        return {
            "status": "healthy",
            "database": "connected",
            "qdrant": "connected",
            "collections": [c.name for c in collections.collections]
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
async def process_query(request: Request, db_session: Session = Depends(get_db)):
    """Process a natural language query"""
    try:
        # Parse request body
        body = await request.json()
        query = body.get("query", "")
        tenant_code = body.get("tenant_code", "")
        conversation_id = body.get("conversation_id", "")

        # Validate input
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Get tenant code from request or use default
        if not tenant_code:
            tenant_code = get_tenant_code_from_request(request)

        logger.info(f"Processing query: '{query}' for tenant: {tenant_code}")
        if conversation_id:
            logger.info(f"Continuing conversation: {conversation_id}")

        # Create agent manager
        agent_manager = AgentManager(
            tenant_code=tenant_code, session=db_session)

        # Process the query using the hybrid approach
        response = agent_manager.process_query(query, conversation_id)

        # Generate follow-up questions
        followup_questions = agent_manager.generate_followup_questions(
            "query", [query], [response])

        # Store conversation in database using tenant_code as user_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        try:
            logger.info(f"Storing conversation with ID: {conversation_id}")
            store_conversation(
                db_session,
                tenant_code,  # Use tenant_code as user_id
                conversation_id,
                query,
                response,
                followup_questions=followup_questions
            )
            logger.info("Conversation stored successfully")
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            logger.exception("Detailed error:")
            # Continue even if storage fails

        # Return a response with the exact fields the frontend expects
        return {
            "response": response,
            "conversation_id": conversation_id,
            "followup_questions": followup_questions
        }

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.exception("Detailed error:")
        raise HTTPException(status_code=500, detail=str(e))


# Keep the old endpoint for backward compatibility
@app.post("/api/query_sql")
async def query_sql(request: Request, db_session: Session = Depends(get_db)):
    """Legacy endpoint that uses SQL-based workflow"""
    try:
        data = await request.json()
        question = data.get("question", "")
        conversation_id = data.get("conversation_id", "")
        tenant_code = data.get("tenant_code", "")

        workflow = get_workflow_for_tenant(tenant_code)
        result = await workflow.process_user_question(
            question,
            conversation_id,
            db_session,
            tenant_code
        )

        logger.info(f"Workflow result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.exception("Detailed error:")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/conversations")
async def get_conversations(request: Request, db: Session = Depends(get_db)):
    try:
        # Get tenant code from headers
        tenant_code = request.headers.get('X-Tenant-Code')
        if not tenant_code:
            raise HTTPException(status_code=401, detail="Tenant code required")

        # Get distinct conversation IDs with their first message (for title)
        # First, get all distinct conversation IDs
        distinct_conversations = db.query(
            UserMemory.conversation_id,
            func.min(UserMemory.message_order).label('first_message_order'),
            func.max(UserMemory.timestamp).label('latest_timestamp')
        ).filter(
            UserMemory.is_active == True,
            UserMemory.expires_at > datetime.utcnow(),
            UserMemory.user_id == tenant_code
        ).group_by(
            UserMemory.conversation_id
        ).order_by(
            func.max(UserMemory.timestamp).desc()
        ).limit(10).all()

        # Now get the first message of each conversation for the title
        result = []
        for conv_id, first_order, latest_ts in distinct_conversations:
            # Get the first message (for title)
            first_message = db.query(UserMemory).filter(
                UserMemory.conversation_id == conv_id,
                UserMemory.message_order == first_order,
                UserMemory.user_id == tenant_code
            ).first()

            if first_message:
                result.append({
                    "id": conv_id,
                    "title": first_message.title or first_message.query[:50],
                    "timestamp": latest_ts.isoformat()  # Use the latest timestamp for sorting
                })

        return result
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve conversations"
        )


@app.get("/api/initial-questions")
async def get_initial_questions(
    request: Request,
    transcription_id: str = None,
    db_session: Session = Depends(get_db)
):
    """Generate initial questions for a transcription"""
    try:
        # Get tenant code from headers
        tenant_code = request.headers.get('X-Tenant-Code')
        if not tenant_code:
            raise HTTPException(status_code=401, detail="Tenant code required")

        # Get UI language preference from headers (default to Dutch)
        ui_language = request.headers.get('X-UI-Language', 'nl')

        # Create a hardcoded response with categories structured exactly as the frontend expects
        categories = {
            "Trending Topics": {
                "description": "Analyze popular discussion topics",
                "questions": [
                    "What are the most discussed topics this month?",
                    "Which topics show increasing trends?",
                    "What topics are commonly mentioned in positive calls?",
                    "How have topic patterns changed over time?",
                    "What are the emerging topics from recent calls?"
                ]
            },
            "Customer Sentiment": {
                "description": "Understand customer satisfaction trends",
                "questions": [
                    "How has overall sentiment changed over time?",
                    "What topics generate the most positive feedback?",
                    "Which issues need immediate attention based on sentiment?",
                    "Show me the distribution of sentiments across topics",
                    "What topics have improving sentiment trends?"
                ]
            },
            "Call Analysis": {
                "description": "Analyze call patterns and duration",
                "questions": [
                    "What is the average call duration by topic?",
                    "Which topics tend to have longer calls?",
                    "Show me the call volume trends by time of day",
                    "What's the distribution of call directions by topic?",
                    "Which days have the highest call volumes?"
                ]
            },
            "Topic Correlations": {
                "description": "Discover relationships between topics",
                "questions": [
                    "Which topics often appear together?",
                    "What topics are related to technical issues?",
                    "Show me topics that commonly lead to follow-up calls",
                    "What topics frequently occur with complaints?",
                    "Which topics have similar sentiment patterns?"
                ]
            },
            "Performance Metrics": {
                "description": "Analyze key performance indicators",
                "questions": [
                    "What's our overall customer satisfaction rate?",
                    "Show me topics with the highest resolution rates",
                    "Which topics need more attention based on metrics?",
                    "What are our best performing areas?",
                    "Show me trends in call handling efficiency"
                ]
            },
            "Time-based Analysis": {
                "description": "Understand temporal patterns",
                "questions": [
                    "What are the busiest times for calls?",
                    "How do topics vary by time of day?",
                    "Show me weekly trends in call volumes",
                    "What patterns emerge during peak hours?",
                    "Which days show the best sentiment scores?"
                ]
            }
        }

        # Dutch translations
        dutch_categories = {
            "Trending Onderwerpen": {
                "description": "Analyseer populaire gespreksonderwerpen",
                "questions": [
                    "Wat zijn de meest besproken onderwerpen deze maand?",
                    "Welke onderwerpen vertonen stijgende trends?",
                    "Welke onderwerpen worden vaak genoemd in positieve gesprekken?",
                    "Hoe zijn onderwerppatronen in de loop van de tijd veranderd?",
                    "Wat zijn de opkomende onderwerpen uit recente gesprekken?"
                ]
            },
            "Klantsentiment": {
                "description": "Begrijp trends in klanttevredenheid",
                "questions": [
                    "Hoe is het algehele sentiment in de loop van de tijd veranderd?",
                    "What topics generate the most positive feedback?",
                    "Which issues need immediate attention based on sentiment?",
                    "Show me the distribution of sentiments across topics",
                    "What topics have improving sentiment trends?"
                ]
            },
            "Gesprekanalyse": {
                "description": "Analyseer gesprekspatronen en -duur",
                "questions": [
                    "Wat is de gemiddelde gespreksduur per onderwerp?",
                    "Welke onderwerpen leiden meestal tot langere gesprekken?",
                    "Toon me de trends in gespreksvolume per dagdeel",
                    "Wat is de verdeling van gespreksrichtingen per onderwerp?",
                    "Welke dagen hebben de hoogste gespreksvolumes?"
                ]
            },
            "Onderwerpscorrelaties": {
                "description": "Ontdek relaties tussen onderwerpen",
                "questions": [
                    "Welke onderwerpen komen vaak samen voor?",
                    "Welke onderwerpen zijn gerelateerd aan technische problemen?",
                    "Toon me onderwerpen die vaak leiden tot vervolgoproepen",
                    "Welke onderwerpen komen vaak voor bij klachten?",
                    "Welke onderwerpen hebben vergelijkbare sentimentpatronen?"
                ]
            },
            "Prestatiemetrieken": {
                "description": "Analyseer belangrijke prestatie-indicatoren",
                "questions": [
                    "Wat is ons algemene percentage klanttevredenheid?",
                    "Toon me onderwerpen met de hoogste oplossingspercentages",
                    "Welke onderwerpen hebben meer aandacht nodig op basis van metrieken?",
                    "Wat zijn onze best presterende gebieden?",
                    "Toon me trends in efficiëntie van gespreksafhandeling"
                ]
            },
            "Tijdgebaseerde Analyse": {
                "description": "Begrijp tijdelijke patronen",
                "questions": [
                    "Wat zijn de drukste tijden voor gesprekken?",
                    "Hoe variëren onderwerpen per tijdstip van de dag?",
                    "Toon me wekelijkse trends in gespreksvolumes",
                    "Welke patronen ontstaan tijdens piekuren?",
                    "Welke dagen tonen de beste sentimentscores?"
                ]
            }
        }

        # Return in the exact format the frontend expects
        return {
            "success": True,
            "categories": dutch_categories if ui_language == 'nl' else categories
        }
    except Exception as e:
        logger.error(f"Error generating initial questions: {e}")
        logger.exception("Detailed error:")

        # Return a fallback response with the same structure
        fallback_categories = {
            "General Analysis": {
                "description": "Basic analysis questions",
                "questions": [
                    "What are the most common topics in our calls?",
                    "How has customer sentiment changed over time?",
                    "Show me the breakdown of call topics by sentiment"
                ]
            }
        } if ui_language == 'en' else {
            "Algemene Analyse": {
                "description": "Basis analysevragen",
                "questions": [
                    "Wat zijn de meest voorkomende onderwerpen in onze gesprekken?",
                    "Hoe is het klantsentiment veranderd in de loop van de tijd?",
                    "Toon me de verdeling van gespreksonderwerpen per sentiment"
                ]
            }
        }

        return {
            "success": False,
            "categories": fallback_categories
        }


@app.post("/api/analyze-response")
async def analyze_response(request: Request, db_session: Session = Depends(get_db)):
    """Analyze a response to a question"""
    try:
        data = await request.json()

        # Extract parameters
        transcription_id = data.get("transcription_id")
        question_id = data.get("question_id")
        response_text = data.get("response")
        tenant_code = request.headers.get('X-Tenant-Code')

        if not all([transcription_id, question_id, response_text, tenant_code]):
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters: transcription_id, question_id, response, tenant_code"
            )

        # Create agent manager
        agent_manager = AgentManager(
            tenant_code=tenant_code, session=db_session)

        # Analyze response
        analysis = agent_manager.analyze_response(
            transcription_id, question_id, response_text
        )

        return {
            "success": True,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error analyzing response: {e}")
        logger.exception("Detailed error:")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/generate-followup")
async def generate_followup(request: Request, db_session: Session = Depends(get_db)):
    """Generate follow-up questions based on previous conversation"""
    try:
        data = await request.json()

        # Extract parameters
        conversation_type = data.get("conversation_type", "system")
        questions = data.get("questions", [])
        responses = data.get("responses", [])
        tenant_code = request.headers.get('X-Tenant-Code')

        if not tenant_code:
            raise HTTPException(status_code=401, detail="Tenant code required")

        if not questions or not responses:
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters: questions, responses"
            )

        # Create agent manager
        agent_manager = AgentManager(
            tenant_code=tenant_code, session=db_session)

        # Generate follow-up questions
        followup_questions = agent_manager.generate_followup_questions(
            conversation_type, questions, responses
        )

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
            "followup_questions": get_default_followup_questions()
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


# Add this function to extract tenant code from request
def get_tenant_code_from_request(request: Request) -> str:
    """Extract tenant code from request headers or default to a fallback tenant"""
    # Try to get from headers first
    tenant_code = request.headers.get('X-Tenant-Code', '')

    # If not in headers, try to get from cookies
    if not tenant_code:
        cookies = request.cookies
        tenant_code = cookies.get('tenant_code', '')

    # If still not found, use a default tenant
    if not tenant_code:
        tenant_code = config.get('DEFAULT_TENANT', 'default')
        logger.warning(
            f"No tenant code found in request, using default: {tenant_code}")

    return tenant_code


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
