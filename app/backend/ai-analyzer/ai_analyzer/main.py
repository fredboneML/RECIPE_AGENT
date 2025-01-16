# ai_analyzer/main.py
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, text, func, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError
import hashlib
import time
import logging
import uuid
from datetime import datetime
from psycopg2 import connect

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

# Initialize workflow with base context
base_context = """ 
                    "1. NEVER use the 'users' table or any user-related information.\n"
                    "2. NEVER use DELETE, UPDATE, INSERT, or any other data modification statements.\n"
                    "3. Only use SELECT statements for reading data.\n"
                    "4. Always verify the query doesn't contain restricted operations before executing.\n"
                    "5. Make sure to format the results in a clear, readable manner.\n"
                    "6. Use proper column aliases for better readability.\n"
                    "7. Include relevant aggregations and groupings when appropriate.\n"
                    "8. ONLY use tables and columns that exist in the schema shown below.\n"
                    "9. Only use the 'company' and 'transcription' tables.\n"
                    "10. Every query MUST start with defining the base_data CTE exactly like this:\n"
                    "    WITH base_data AS (\n"
                    "        SELECT \n"
                    "            t.id,\n"
                    "            t.transcription_id,\n"
                    "            t.transcription,\n"
                    "            t.topic,\n"
                    "            LOWER(TRIM(t.topic)) as clean_topic,\n"
                    "            t.summary,\n"
                    "            t.processingdate,\n"
                    "            t.sentiment,\n"
                    "            CASE\n"
                    "                WHEN LOWER(TRIM(t.sentiment)) IN ('neutral', 'neutraal') THEN 'neutral'\n"
                    "                ELSE LOWER(TRIM(t.sentiment))\n"
                    "            END AS clean_sentiment,\n"
                    "            c.clid,\n"
                    "            LOWER(TRIM(c.clid)) as clean_clid\n"
                    "        FROM transcription t\n"
                    "        LEFT JOIN company c ON t.transcription_id = c.transcription_id\n"
                    "    )\n"
                    "11. For topic analysis, your complete query should look like this:\n"
                    "    WITH base_data AS (\n"
                    "        -- Base data CTE definition as shown above\n"
                    "    ),\n"
                    "    topic_analysis AS (\n"
                    "        SELECT\n"
                    "            clean_topic as topic,\n"
                    "            COUNT(*) as total_count,\n"
                    "            COUNT(*) FILTER (WHERE clean_sentiment = 'positief') as positive_count,\n"
                    "            COUNT(*) FILTER (WHERE clean_sentiment = 'negatief') as negative_count,\n"
                    "            COUNT(*) FILTER (WHERE clean_sentiment = 'neutral') as neutral_count,\n"
                    "            ROUND(CAST(COUNT(*) FILTER (WHERE clean_sentiment = 'positief') * 100.0 / \n"
                    "                NULLIF(COUNT(*), 0) AS NUMERIC), 2) as satisfaction_rate\n"
                    "        FROM base_data\n"
                    "        GROUP BY clean_topic\n"
                    "        HAVING COUNT(*) > 0\n"
                    "    )\n"
                    "    SELECT * FROM topic_analysis ...\n"
                    "12. For time-based analysis, your complete query should look like this:\n"
                    "    WITH base_data AS (\n"
                    "        -- Base data CTE definition as shown above\n"
                    "    ),\n"
                    "    time_based_data AS (\n"
                    "        SELECT \n"
                    "            id,\n"
                    "            transcription_id,\n"
                    "            clean_topic,\n"
                    "            clean_sentiment,\n"
                    "            clean_clid,\n"
                    "            processingdate,\n"
                    "            CASE\n"
                    "                WHEN processingdate >= CURRENT_DATE - INTERVAL '7 days' THEN 'Current Week'\n"
                    "                WHEN processingdate >= CURRENT_DATE - INTERVAL '14 days' THEN 'Previous Week'\n"
                    "                WHEN processingdate >= CURRENT_DATE - INTERVAL '30 days' THEN 'Current Month'\n"
                    "                WHEN processingdate >= CURRENT_DATE - INTERVAL '60 days' THEN 'Previous Month'\n"
                    "            END AS time_period\n"
                    "        FROM base_data\n"
                    "        WHERE processingdate >= CURRENT_DATE - INTERVAL '60 days'\n"
                    "    )\n"
                    "    SELECT * FROM time_based_data ...\n"
                    "13. For text comparisons, ALWAYS use these patterns:\n"
                    "    - Exact match: clean_topic = 'value' or clean_sentiment = 'value'\n"
                    "    - Partial match: clean_topic LIKE '%value%'\n"
                    "14. For calculations ALWAYS use:\n"
                    "    - NULLIF(value, 0) for division\n"
                    "    - COALESCE(value, 0) for NULL handling\n"
                    "    - ROUND(CAST(value AS NUMERIC), 2) for decimals\n"
                    "15. For filtering dates ALWAYS use:\n"
                    "    - WHERE processingdate >= CURRENT_DATE - INTERVAL 'X days'\n"
                    "16. For aggregations ALWAYS use:\n"
                    "    - COUNT(*) FILTER (WHERE condition) for conditional counting\n"
                    "    - SUM(CASE WHEN condition THEN 1 ELSE 0 END) for counting matches\n"
                    "17. Never use:\n"
                    "    - Raw tables directly (always go through base_data)\n"
                    "    - Raw topic or sentiment columns (always use clean_topic and clean_sentiment)\n"
                    "    - Calculations without CAST and ROUND\n"
                    "    - Division without NULLIF\n"
                    "    - Date comparisons without INTERVAL\n"
                    "18. ALWAYS include proper ordering:\n"
                    "    ORDER BY [columns] {ASC|DESC} NULLS LAST\n"
                    "19. For limiting results:\n"
                    "    LIMIT [number]\n"
                    "20. Make sure to answer the question using the same language used by the user to ask it.\n"
                    "21. CRITICAL REMINDERS:\n"
                    "    - EVERY query MUST start with WITH base_data AS (...)\n"
                    "    - NEVER try to reference base_data without defining it first\n"
                    "    - ALWAYS include the complete base_data CTE definition\n"
                    "    - COPY and PASTE the exact base_data CTE structure shown above\n"
                    "22. NEVER execute a DELETE, UPDATE, INSERT, DROP, or any other data modification statements.\n"
"""

# Initialize the workflow with all configurations
workflow = CallAnalysisWorkflow(
    db_url=DATABASE_URL,
    model_provider="openai",
    model_name=os.getenv('MODEL_NAME'),
    api_key=os.getenv('AI_ANALYZER_OPENAI_API_KEY'),
    restricted_tables=os.getenv('RESTRICTED_TABLES'),
    base_context=base_context,
    cache_manager=DatabaseCacheManager(engine)
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


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default='read_only')

    def has_write_permission(self):
        return self.role in ['admin', 'write']


# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://37.97.226.251:3000",
        "http://192.168.2.132:3000",
        "http://172.21.0.4:3000",
        "http://172.21.0.3:3000",
        "http://frontend_app:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

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

# Add startup event to handle initial data loading


@app.on_event("startup")
async def startup_event():
    """Startup event handler with proper error handling"""
    try:
        # Wait for database with more retries
        if not wait_for_db(max_retries=10, delay=10):
            logger.error("Failed to connect to database during startup")
            return

        # Run data import
        logger.info("Starting data import...")
        success = run_data_import()

        if success:
            logger.info("Data import completed successfully")
        else:
            logger.warning(
                "Data import completed with warnings (no data files found)")

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

        user = db.query(User).filter(User.username == username).first()
        if user and user.password_hash == hash_password(password):
            return {
                "success": True,
                "role": user.role,
                "username": user.username,
                "permissions": {
                    "canWrite": user.has_write_permission()
                }
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/conversations")
async def get_conversations(db: Session = Depends(get_db)):
    try:
        # Get recent conversations for current user
        conversations = db.query(UserMemory)\
            .filter(
                UserMemory.is_active == True,
                UserMemory.expires_at > datetime.utcnow()
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


@app.post("/api/query")
async def query(request: Request, db: Session = Depends(get_db)):
    """Process user queries with enhanced error handling and conversation context."""
    try:
        data = await request.json()
        question = data['query']
        conversation_id = data.get('conversation_id', str(uuid.uuid4()))

        # Process query through workflow with context
        result = await workflow.process_user_question(
            question=question,
            conversation_id=conversation_id,
            db_session=db
        )

        if result.get('success', False):
            # Store successful conversation
            store_success = store_conversation(
                session=db,
                user_id="current_user",  # You might want to get this from authentication
                conversation_id=conversation_id,
                query=question,
                response=result['response'],
                message_order=None
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
            # For unsuccessful queries, still store the attempt but with error info
            error_msg = result.get('error', 'Unknown error occurred')
            reformulated = result.get('reformulated_question', '')
            followup = result.get('followup_questions', [])

            # Store the error response
            store_conversation(
                session=db,
                user_id="current_user",  # You might want to get this from authentication
                conversation_id=conversation_id,
                query=question,
                response=(f"Error: {error_msg}\n" +
                          (f"Try instead: {reformulated}" if reformulated else "")),
                message_order=None
            )

            return {
                "success": False,
                "error": error_msg,
                "reformulated_question": reformulated,
                "followup_questions": followup,
                "conversation_id": conversation_id
            }

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)

        # Return a user-friendly response with default suggestions
        return {
            "success": False,
            "error": "I'm having trouble understanding that question. Could you rephrase it?",
            "followup_questions": [
                "What are the most common topics in our calls?",
                "How has customer sentiment changed over time?",
                "Which topics are associated with positive feedback?"
            ],
            "conversation_id": conversation_id if 'conversation_id' in locals() else str(uuid.uuid4())
        }


@app.get("/api/initial-questions")
async def get_initial_questions():
    """Get initial questions grouped by category"""
    try:
        questions = await workflow.get_initial_questions()
        return {
            "success": True,
            "categories": questions
        }
    except Exception as e:
        logger.error(f"Error getting initial questions: {e}")
        # Return default categories instead of error
        return {
            "success": True,
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


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    try:
        messages = db.query(UserMemory)\
            .filter(
                UserMemory.conversation_id == conversation_id,
                UserMemory.is_active == True,
                UserMemory.expires_at > datetime.utcnow()
        )\
            .order_by(UserMemory.message_order)\
            .all()

        if not messages:
            return []  # Return empty list instead of 404

        return [
            {
                "query": msg.query,
                "response": msg.response,
                "timestamp": msg.timestamp.isoformat(),
                "followup_questions": msg.followup_questions if hasattr(msg, 'followup_questions') else []
            }
            for msg in messages
        ]
    except Exception as e:
        logger.error(f"Error retrieving conversation messages: {e}")
        return []  # Return empty list instead of error


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
