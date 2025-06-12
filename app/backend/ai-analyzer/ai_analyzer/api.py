# app/backend/ai-analyzer/ai_analyzer/api.py
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
import uuid
from jose import JWTError, jwt
from datetime import timedelta

from ai_analyzer.agents import AgentManager
from ai_analyzer.data_pipeline import get_db_session
from ai_analyzer.config import JWT_SECRET_KEY, JWT_ALGORITHM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Analyzer API", description="API for AI call analysis")

# JWT Configuration
SECRET_KEY = JWT_SECRET_KEY
ALGORITHM = JWT_ALGORITHM

# Security scheme for token authentication
security = HTTPBearer()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User model for type hints (should match the one in main.py)


class User:
    def __init__(self, username: str, tenant_code: str, role: str):
        self.username = username
        self.tenant_code = tenant_code
        self.role = role

# Token data model


class TokenData:
    def __init__(self, username: str = None, tenant_code: str = None, role: str = None):
        self.username = username
        self.tenant_code = tenant_code
        self.role = role

# Dependency to get database session


def get_session():
    session = get_db_session()
    try:
        yield session
    finally:
        session.close()

# Authentication dependency


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_session)) -> User:
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
            tenant_code=payload.get("tenant_code"),
            role=payload.get("role")
        )
    except JWTError as e:
        logger.error(f"get_current_user: JWTError: {e}")
        raise credentials_exception

    # For this simplified version, we'll create a User object from token data
    # In a real implementation, you'd query the database to verify the user exists
    if not all([token_data.username, token_data.tenant_code, token_data.role]):
        logger.warning(f"get_current_user: Incomplete token data")
        raise credentials_exception

    user = User(
        username=token_data.username,
        tenant_code=token_data.tenant_code,
        role=token_data.role
    )

    logger.info(
        f"get_current_user: Authenticated user: {user.username}, tenant: {user.tenant_code}, role: {user.role}")
    return user


@app.get("/health")
def health_check(current_user: User = Depends(get_current_user)):
    """Health check endpoint with JWT authentication"""
    return {
        "status": "healthy",
        "user": current_user.username,
        "tenant": current_user.tenant_code
    }


@app.get("/api/initial-questions")
async def get_initial_questions(
    request: Request,
    current_user: User = Depends(get_current_user),
    transcription_id: str = None,
    db_session: Session = Depends(get_session)
):
    """Generate initial questions for a transcription"""
    try:
        # Use tenant code from the authenticated user
        tenant_code = current_user.tenant_code

        # Create a hardcoded response with unique IDs for each question
        initial_questions = {
            "Trending Topics": {
                "description": "Questions about trending topics",
                "questions": [
                    {
                        "id": "trending_1",
                        "text": "What are the most discussed topics this month?"
                    },
                    {
                        "id": "trending_2",
                        "text": "Which topics have seen the biggest increase in mentions?"
                    },
                    {
                        "id": "trending_3",
                        "text": "What are the emerging topics in customer conversations?"
                    },
                    {
                        "id": "trending_4",
                        "text": "How have topic trends changed over the past quarter?"
                    },
                    {
                        "id": "trending_5",
                        "text": "Which topics are most frequently mentioned together?"
                    }
                ]
            },
            "Customer Sentiment": {
                "description": "Questions about customer sentiment",
                "questions": [
                    {
                        "id": "sentiment_1",
                        "text": "How has overall sentiment changed over time?"
                    },
                    {
                        "id": "sentiment_2",
                        "text": "Which topics have the most negative sentiment?"
                    },
                    {
                        "id": "sentiment_3",
                        "text": "What percentage of calls show positive sentiment?"
                    },
                    {
                        "id": "sentiment_4",
                        "text": "How does sentiment vary by call duration?"
                    },
                    {
                        "id": "sentiment_5",
                        "text": "Which topics have improved in sentiment recently?"
                    }
                ]
            },
            "Call Analysis": {
                "description": "Questions about call patterns",
                "questions": [
                    {
                        "id": "call_1",
                        "text": "What is the average call duration by topic?"
                    },
                    {
                        "id": "call_2",
                        "text": "Which topics lead to the longest calls?"
                    },
                    {
                        "id": "call_3",
                        "text": "How does call volume vary by day of week?"
                    },
                    {
                        "id": "call_4",
                        "text": "What percentage of calls are inbound vs outbound?"
                    },
                    {
                        "id": "call_5",
                        "text": "Which call directions show the most negative sentiment?"
                    }
                ]
            }
        }

        # Return in the exact format the frontend expects
        return {
            "success": True,
            "categories": initial_questions
        }
    except Exception as e:
        logger.error(f"Error generating initial questions: {e}")
        logger.exception("Detailed error:")

        # Return a fallback response with the same structure
        return {
            "success": False,
            "categories": {
                "General Analysis": {
                    "description": "Basic analysis questions",
                    "questions": [
                        {
                            "id": "general_1",
                            "text": "What are the most common topics in our calls?"
                        },
                        {
                            "id": "general_2",
                            "text": "How has customer sentiment changed over time?"
                        },
                        {
                            "id": "general_3",
                            "text": "Show me the breakdown of call topics by sentiment"
                        }
                    ]
                }
            }
        }


@app.post("/api/analyze-response")
def analyze_response(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Analyze a response to a question"""
    try:
        # Use tenant code from the authenticated user
        tenant_code = current_user.tenant_code
        transcription_id = request.get("transcription_id")
        question_id = request.get("question_id")
        response_text = request.get("response")

        if not all([transcription_id, question_id, response_text]):
            raise HTTPException(
                status_code=400, detail="Missing required fields: transcription_id, question_id, response")

        # Create agent manager
        agent_manager = AgentManager(tenant_code=tenant_code, session=session)

        # Analyze response
        analysis = agent_manager.analyze_response(
            transcription_id, question_id, response_text)

        return {
            "success": True,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error analyzing response: {e}")
        logger.exception("Detailed error:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-followup-questions")
def generate_followup_questions(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Generate follow-up questions based on previous responses"""
    try:
        # Use tenant code from the authenticated user
        tenant_code = current_user.tenant_code
        transcription_id = request.get("transcription_id")
        previous_questions = request.get("previous_questions", [])
        previous_responses = request.get("previous_responses", [])

        if not transcription_id:
            raise HTTPException(
                status_code=400, detail="Missing transcription_id")

        if len(previous_questions) != len(previous_responses):
            raise HTTPException(
                status_code=400, detail="Mismatched questions and responses")

        # Create agent manager
        agent_manager = AgentManager(tenant_code=tenant_code, session=session)

        # Generate follow-up questions
        questions = agent_manager.generate_followup_questions(
            transcription_id,
            previous_questions,
            previous_responses
        )

        return {
            "success": True,
            "questions": questions
        }
    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        logger.exception("Detailed error:")
        raise HTTPException(status_code=500, detail=str(e))
