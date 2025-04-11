#!/usr/bin/env python3
import os
import logging
from typing import List, Dict, Any, Optional
import uuid
import re
from sqlalchemy import text
import time
import hashlib
from datetime import datetime, timedelta

from ai_analyzer.agents.factory import AgentFactory
from ai_analyzer.config import config
from ai_analyzer.data_pipeline import create_agent_tables
from sqlalchemy import create_engine
from ai_analyzer.config import DATABASE_URL
from sqlalchemy.orm import Session

from ai_analyzer.agents.database_inspector import DatabaseInspectorAgent
from ai_analyzer.agents.sql_generator import SQLGeneratorAgent
from ai_analyzer.utils.model_logger import get_model_config_from_env
from ai_analyzer.data_import_postgresql import UserMemory
from ai_analyzer.utils.qdrant_client import get_qdrant_client, get_embedding_model, search_qdrant_with_retry
from pybreaker import CircuitBreaker
from ai_analyzer.utils.singleton_resources import get_qdrant_client, get_embedding_model
from ai_analyzer.utils.resilience import search_qdrant_safely

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create circuit breaker
qdrant_breaker = CircuitBreaker(fail_max=5, reset_timeout=30)

# Define default model
DEFAULT_MODEL = "gpt-3.5-turbo"


class AgentManager:
    """Manager for Agno agents"""

    def __init__(self, tenant_code: str, session: Session, model: str = DEFAULT_MODEL):
        self.tenant_code = tenant_code
        self.session = session
        self.model = model
        self.agent_team = None
        self.followup_agent = None
        self.sql_agent = None

        # Get singleton instances - don't create new ones
        self.qdrant_client = get_qdrant_client()
        self.embedding_model = get_embedding_model()
        self.collection_name = f"tenant_{tenant_code}"

    def _initialize_agents_if_needed(self):
        """Initialize agents if they haven't been initialized yet"""
        if self.agent_team is None:
            try:
                self.agent_team = AgentFactory.create_agent_team(
                    self.tenant_code)
            except Exception as e:
                logger.error(f"Error initializing agent team: {e}")
                logger.exception("Detailed error:")

        if self.followup_agent is None:
            try:
                self.followup_agent = AgentFactory.create_followup_agent(
                    self.tenant_code)
            except Exception as e:
                logger.error(f"Error initializing followup agent: {e}")
                logger.exception("Detailed error:")

        if self.sql_agent is None:
            try:
                # Get model configuration
                model_config = get_model_config_from_env()

                # Log model configuration
                logger.info(
                    f"Initializing SQL agent with model configuration:")
                logger.info(f"Provider: {model_config['provider']}")
                logger.info(f"Model: {model_config['model_name']}")
                if model_config['provider'] == 'groq':
                    logger.info(
                        f"Groq model: {model_config['groq_model_name']}")
                    logger.info(
                        f"Groq OpenAI compatibility: {model_config['groq_use_openai_compatibility']}")

                # Create SQL agent with model configuration
                self.sql_agent = SQLGeneratorAgent(
                    model_provider=model_config['provider'],
                    model_name=model_config['model_name'],
                    api_key=model_config.get('api_key'),
                    base_context=None
                )
            except Exception as e:
                logger.error(f"Error initializing SQL agent: {e}")
                logger.exception("Detailed error:")

    def generate_initial_questions(self, transcription_id: str) -> List[str]:
        """Return hardcoded initial questions instead of generating them"""
        try:
            # Use hardcoded questions instead of generating them
            questions = [
                # Trending Topics
                "What are the most discussed topics this month?",
                "Which topics show increasing trends?",
                "What topics are commonly mentioned in positive calls?",
                "How have topic patterns changed over time?",
                "What are the emerging topics from recent calls?",

                # Customer Sentiment
                "How has overall sentiment changed over time?",
                "What topics generate the most positive feedback?",
                "Which issues need immediate attention based on sentiment?",
                "Show me the distribution of sentiments across topics",
                "What topics have improving sentiment trends?",

                # Call Analysis
                "What is the average call duration by topic?",
                "Which topics tend to have longer calls?",
                "Show me the call volume trends by time of day",
                "What's the distribution of call directions by topic?",
                "Which days have the highest call volumes?",

                # Topic Correlations
                "Which topics often appear together?",
                "What topics are related to technical issues?",
                "Show me topics that commonly lead to follow-up calls",
                "What topics frequently occur with complaints?",
                "Which topics have similar sentiment patterns?",

                # Performance Metrics
                "What's our overall customer satisfaction rate?",
                "Show me topics with the highest resolution rates",
                "Which topics need more attention based on metrics?",
                "What are our best performing areas?",
                "Show me trends in call handling efficiency",

                # Time-based Analysis
                "What are the busiest times for calls?",
                "How do topics vary by time of day?",
                "Show me weekly trends in call volumes",
                "What patterns emerge during peak hours?",
                "Which days show the best sentiment scores?"
            ]

            # Save questions to database (keeping SQL for metadata)
            question_ids = self._save_questions_to_db(
                transcription_id, questions, "initial")

            return questions
        except Exception as e:
            logger.error(f"Error generating initial questions: {e}")
            logger.exception("Detailed error:")
            return []

    def analyze_response(self, transcription_id: str, question_id: str, response_text: str) -> str:
        """Analyze a response to a question"""
        try:
            # Get question from database
            question = self._get_question_from_db(question_id)

            # Detect language (Dutch vs English)
            is_dutch = any(dutch_word in question.lower() for dutch_word in
                           ['wat', 'hoe', 'waarom', 'welke', 'kunnen', 'waar', 'wie', 'wanneer', 'onderwerp'])

            # Create agent
            agent = AgentFactory.create_response_analyzer_agent(
                self.tenant_code)

            # Generate analysis
            prompt = f"""
            Transcription ID: {transcription_id}
            
            Question:
            {question}
            
            Response:
            {response_text}
            
            {'Analyseer deze reactie in het Nederlands:' if is_dutch else 'Please analyze this response:'}
            """

            analysis = agent.get_response(prompt)

            # Save analysis to database
            analysis_id = self._save_analysis_to_db(
                transcription_id, question_id, analysis)

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing response: {e}")
            logger.exception("Detailed error:")
            return f"Error analyzing response: {str(e)}"

    def generate_followup_questions(self, conversation_type: str, questions: List[str], responses: List[str]) -> List[str]:
        """Generate follow-up questions based on conversation history"""
        try:
            # Initialize agents if needed
            self._initialize_agents_if_needed()

            # Get the appropriate agent
            agent = self.followup_agent if self.followup_agent else self._get_agent()

            if agent is None:
                logger.error(
                    "No agent available for generating follow-up questions")
                return self._get_default_followup_questions()

            # Format the conversation history
            conversation = ""
            for i, (q, r) in enumerate(zip(questions, responses)):
                conversation += f"Question {i+1}: {q}\nResponse {i+1}: {r}\n\n"

            # Detect language from the last question
            last_question = questions[-1] if questions else ""
            is_dutch = any(dutch_word in last_question.lower() for dutch_word in
                           ['wat', 'hoe', 'waarom', 'welke', 'kunnen', 'waar', 'wie', 'wanneer', 'onderwerp'])

            # Create a prompt for generating follow-up questions
            prompt = f"""
            Based on the following conversation, generate 3 follow-up questions that would help continue and deepen the conversation:
            
            {conversation}
            
            {'Genereer 3 vervolgvragen in het Nederlands:' if is_dutch else 'FOLLOW-UP QUESTIONS:'}
            """

            try:
                # Run the agent to generate follow-up questions
                response = agent.run(prompt)

                # Extract the content if it's a RunResponse object
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)

                # Parse the response to extract questions
                followup_questions = self._parse_questions(response_text)

                # Ensure we have at least some questions
                if not followup_questions:
                    followup_questions = self._get_default_followup_questions()

                return followup_questions[:3]  # Return at most 3 questions

            except Exception as e:
                logger.error(
                    f"Error running agent for followup questions: {e}")
                logger.exception("Detailed error:")
                return self._get_default_followup_questions()

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            logger.exception("Detailed error:")
            return self._get_default_followup_questions()

    @qdrant_breaker
    def search_transcriptions_safe(self, query: str) -> List[Dict[str, Any]]:
        """Circuit-breaker protected vector search"""
        try:
            # First attempt with circuit breaker protection
            return self.search_transcriptions(query)
        except Exception as e:
            logger.error(f"Protected search failed, will not retry: {e}")
            return []  # Return empty results if the search fails

    def search_transcriptions(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant transcriptions using vector search"""
        try:
            # Create the agent if it doesn't exist
            agent = self._get_agent()

            if agent is None:
                logger.error("No agent found")
                return []

            # Try agent-based search first
            try:
                # Use knowledge base search via the agent
                # This would typically use the agent's knowledge retrieval capabilities
                logger.info(f"Search query: '{query}'")

                # Implementation varies based on your agent setup, typically:
                # search_results = agent.search_knowledge(query)
                # or similar approach

                # Fallback to direct Qdrant search if the agent-based search fails or returns no results

                # Use the singleton instances initialized in __init__
                collection_info = self.qdrant_client.get_collection(
                    self.collection_name)
                vector_names = list(
                    collection_info.config.params.vectors.keys())

                if not vector_names:
                    logger.error(
                        f"No vector names found in collection {self.collection_name}")
                    return []

                vector_name = vector_names[0]  # Use the first vector name
                logger.info(f"Using vector name: {vector_name} for search")

                # Generate embeddings for the query using the singleton model
                embeddings = list(self.embedding_model.embed([query]))
                query_vector = embeddings[0].tolist()

                # Search Qdrant with retry logic and circuit breaker
                search_results = search_qdrant_safely(
                    self.qdrant_client,
                    self.collection_name,
                    query_vector,
                    limit=2
                )

                # Log the search results
                logger.info(f"Found {len(search_results)} results in Qdrant")

                # Convert results to the expected format
                formatted_results = []
                for i, result in enumerate(search_results):
                    # Log each result with its score and text
                    logger.info(f"Result {i+1} (score: {result.score:.4f}):")

                    # Log the entire payload for debugging
                    logger.info(
                        f"  Full payload keys: {list(result.payload.keys())}")

                    # Log the entire payload
                    logger.info(f"  Full payload: {result.payload}")

                    # Try to find text in various fields
                    text = result.payload.get("text", "")
                    if not text:
                        # Try alternative fields that might contain the text
                        text = result.payload.get("content", "")
                        if not text:
                            text = result.payload.get("transcript", "")
                            if not text:
                                text = result.payload.get("transcription", "")
                                if not text:
                                    # Try to find any field that might contain text
                                    for key, value in result.payload.items():
                                        if isinstance(value, str) and len(value) > 50:
                                            text = value
                                            logger.info(
                                                f"  Found text in field: {key}")
                                            break
                                if not text:
                                    # We already logged the full payload above
                                    pass

                    # Log the first 100 characters of the text to avoid huge logs
                    logger.info(f"  Text: {text[:100]}..." if len(
                        text) > 100 else f"  Text: {text}")

                    # If we still don't have text, use a placeholder
                    if not text:
                        text = "No text content found in this result."

                    # Create a metadata dictionary with all payload fields
                    metadata = {
                        "id": result.payload.get("id", ""),
                        "call_id": result.payload.get("call_id", ""),
                        "timestamp": result.payload.get("timestamp", ""),
                        "score": result.score
                    }

                    # Add all other payload fields to metadata
                    for key, value in result.payload.items():
                        if key not in metadata:
                            metadata[key] = value

                    formatted_results.append({
                        "text": text,
                        "metadata": metadata
                    })

                return formatted_results

            except Exception as e:
                logger.error(f"Error in agent-based search: {e}")
                logger.exception("Detailed error:")

                # Don't attempt fallback search here - we'll let the circuit breaker
                # handle retry logic through search_transcriptions_safe
                raise

        except Exception as e:
            logger.error(f"Error in search_transcriptions: {e}")
            logger.exception("Detailed error:")
            return []

    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate a response to a query based on search results"""
        try:
            # Create the agent if it doesn't exist
            agent = self._get_agent()

            if agent is None:
                return "I'm sorry, I couldn't generate a response. The agent could not be initialized."

            # If we have search results, format them as context for the agent
            if search_results:
                # Format the search results as context
                context = "Here are some relevant call transcriptions:\n\n"
                for i, result in enumerate(search_results):
                    context += f"Transcript {i+1}:\n{result['text']}\n\n"

                # Create a prompt with the context and query
                prompt = f"""
                {context}
                
                Based on the above call transcriptions, please answer the following question:
                {query}
                """

                # Use the agent's run method to generate a response
                try:
                    response_obj = agent.run(prompt)

                    # Extract the text content from the RunResponse object
                    if hasattr(response_obj, 'content'):
                        return response_obj.content
                    elif isinstance(response_obj, str):
                        return response_obj
                    else:
                        # Convert the response object to a string if it's not already a string
                        return str(response_obj)

                except Exception as e:
                    logger.error(f"Error running agent: {e}")
                    logger.exception("Detailed error:")
                    return "I'm sorry, I couldn't generate a response based on the available data."
            else:
                # No search results, so generate a response asking for more information
                return "I don't have enough information to answer that question. Could you please provide more details or specific call transcriptions to analyze?"

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.exception("Detailed error:")
            return f"I'm sorry, I couldn't generate a response. Error: {str(e)}"

    def _parse_questions(self, response: str) -> List[str]:
        """Parse questions from agent response"""
        try:
            # Split the response into lines
            lines = response.strip().split('\n')

            # Extract questions (lines that end with a question mark or are numbered)
            followup_questions = []
            for line in lines:
                # Remove any numbering or bullet points
                clean_line = line.strip()
                if clean_line and not clean_line.startswith("FOLLOW-UP QUESTIONS:"):
                    # Remove numbering like "1.", "2.", etc.
                    if clean_line[0].isdigit() and clean_line[1:].startswith('. '):
                        clean_line = clean_line[3:]
                    # Remove bullet points
                    if clean_line.startswith('- '):
                        clean_line = clean_line[2:]

                    followup_questions.append(clean_line)

            # Ensure we have at least some questions
            if not followup_questions:
                followup_questions = self._get_default_followup_questions()

            return followup_questions[:3]  # Return at most 3 questions

        except Exception as e:
            logger.error(f"Error parsing questions: {e}")
            logger.exception("Detailed error:")
            return self._get_default_followup_questions()

    def _get_default_followup_questions(self) -> List[str]:
        """Get default follow-up questions"""
        return [
            "What other information would you like to know about the call data?",
            "Would you like to explore a specific topic in more detail?",
            "Is there a particular time period you'd like to analyze?"
        ]

    def _get_question_from_db(self, question_id: str) -> str:
        """Get question text from user_memory table"""
        if not self.session:
            return ""

        try:
            # Try to get the question directly by UUID
            result = self.session.execute(
                text("""
                    SELECT query FROM user_memory 
                    WHERE response LIKE :response_pattern AND user_id = :tenant_code
                """),
                {
                    "response_pattern": f"%Question ID: {question_id}%",
                    "tenant_code": self.tenant_code
                }
            ).fetchone()

            if result:
                # Remove the prefix if it exists
                query = result[0]
                if query.startswith('['):
                    # Extract the actual question after the prefix
                    parts = query.split('] ', 1)
                    if len(parts) > 1:
                        return parts[1]
                return query
            return ""
        except Exception as e:
            logger.error(f"Error getting question from database: {e}")
            logger.exception("Detailed error:")
            return ""

    def _save_questions_to_db(self, transcription_id: str, questions: List[str], question_type: str = "initial") -> List[str]:
        """Save questions to database using the user_memory table"""
        question_ids = []

        # Always generate UUIDs for questions
        question_ids = [str(uuid.uuid4()) for _ in questions]

        # If we don't have a session or questions, just return the IDs
        if not self.session or not questions:
            return question_ids

        try:
            # Check the structure of user_memory table
            result = self.session.execute(
                text(
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'user_memory'")
            ).fetchall()

            # Create a mapping of column names to data types
            column_types = {col: dtype for col, dtype in result}

            # Check if id is an integer type
            id_is_integer = column_types.get('id', '').startswith('int')

            # Generate a unique conversation ID with timestamp to avoid conflicts
            timestamp = int(time.time())
            conversation_id = f"initial_questions_{transcription_id}_{timestamp}"

            # If we need integer IDs, find the current max ID to avoid conflicts
            start_id = 1000000
            if id_is_integer:
                try:
                    # Get the current max ID from the user_memory table
                    max_id_result = self.session.execute(
                        text("SELECT MAX(id) FROM user_memory")
                    ).fetchone()

                    if max_id_result and max_id_result[0]:
                        # Start from max_id + 1 to avoid conflicts
                        start_id = max(int(max_id_result[0]) + 1, start_id)
                        logger.info(f"Starting ID sequence from {start_id}")
                except Exception as e:
                    logger.warning(f"Error getting max ID, using default: {e}")

            for i, (question_id, question) in enumerate(zip(question_ids, questions)):
                # Store each question as a separate memory entry
                # If id is integer, generate a numeric ID instead of UUID
                params = {
                    "user_id": self.tenant_code,
                    "conversation_id": conversation_id,
                    "query": f"[{question_type.upper()}] {question}",
                    "response": f"Question ID: {question_id}",
                    "message_order": i,
                    "followup_questions": [],
                    "title": f"Initial Question {i+1}"  # Add title field
                }

                if id_is_integer:
                    # Use a dynamic incrementing integer ID
                    numeric_id = start_id + i
                    params["id"] = numeric_id
                    # Store the UUID in the response for reference
                    params["response"] = f"Question ID: {question_id} (numeric_id: {numeric_id})"
                else:
                    params["id"] = question_id

                self.session.execute(
                    text("""
                        INSERT INTO user_memory 
                        (id, user_id, conversation_id, query, response, message_order, is_active, timestamp, followup_questions, title, expires_at)
                        VALUES 
                        (:id, :user_id, :conversation_id, :query, :response, :message_order, TRUE, NOW(), :followup_questions, :title, NOW() + INTERVAL '30 days')
                    """),
                    params
                )

            self.session.commit()
            logger.info(
                f"Saved {len(questions)} questions to user_memory table")

        except Exception as e:
            logger.error(f"Error saving questions to database: {e}")
            logger.exception("Detailed error:")
            if self.session:
                self.session.rollback()

        return question_ids

    def _save_analysis_to_db(self, transcription_id: str, question_id: str, analysis: str) -> str:
        """Save analysis to database using user_memory table"""
        analysis_id = str(uuid.uuid4())

        if not self.session:
            return analysis_id

        try:
            # Get the original question
            question = self._get_question_from_db(question_id)

            # Create a conversation ID for the analysis
            conversation_id = f"analysis_{transcription_id}_{question_id}"

            # Store the analysis as a response in user_memory
            self.session.execute(
                text("""
                    INSERT INTO user_memory 
                    (id, user_id, conversation_id, query, response, message_order, is_active, timestamp, followup_questions, title, expires_at)
                    VALUES 
                    (:id, :user_id, :conversation_id, :query, :response, 0, TRUE, NOW(), :followup_questions, :title, NOW() + INTERVAL '30 days')
                """),
                {
                    "id": analysis_id,
                    "user_id": self.tenant_code,
                    "conversation_id": conversation_id,
                    "query": f"Analysis of: {question}",
                    "response": analysis,
                    "followup_questions": [],
                    # Add title field
                    "title": f"Analysis of Question {question_id[:8]}"
                }
            )
            self.session.commit()
            logger.info(f"Saved analysis with ID {analysis_id}")

        except Exception as e:
            logger.error(f"Error saving analysis to database: {e}")
            logger.exception("Detailed error:")
            if self.session:
                self.session.rollback()

        return analysis_id

    def _get_agent(self):
        """Get the agent team or followup agent"""
        # Initialize agents if needed
        self._initialize_agents_if_needed()

        if self.agent_team:
            return self.agent_team
        elif self.followup_agent:
            return self.followup_agent
        else:
            logger.error("No agent found")
            return None

    def _detect_language(self, text: str) -> str:
        """Detect if the text is in Dutch or English"""
        # Get total word count and cleaned words
        words = text.lower().split()
        total_words = len(words)
        logger.info(f"Total words in text: {total_words}")

        # Count English words
        english_words = ['what', 'which', 'how', 'where', 'when', 'who', 'why', 'did', 'does', 'has', 'had', 'it', 'there', 'they', 'show',
                         'is', 'are', 'was', 'were', 'the', 'this', 'that', 'these', 'those', 'our', 'your', 'an', 'a', 'top', 'give', 'do']
        nb_en = sum(1 for word in words if word in english_words)
        logger.info(
            f"English words found: {[word for word in words if word in english_words]}")

        # Count Dutch words
        dutch_words = ['wat', 'hoe', 'waarom', 'welke', 'kunnen', 'waar', 'wie', 'wanneer', 'onderwerp',
                       'kun', 'kunt', 'je', 'jij', 'u', 'bent', 'zijn', 'waar', 'wat', 'wie', 'hoe',
                       'waarom', 'wanneer', 'welk', 'welke', 'het', 'de', 'een', 'het', 'deze', 'dit',
                       'die', 'dat', 'mijn', 'uw', 'jullie', 'ons', 'onze', 'geen', 'niet', 'met',
                       'over', 'door', 'om', 'op', 'voor', 'na', 'bij', 'aan', 'in', 'uit', 'te',
                       'bedrijf', 'waarom', 'tevreden', 'graag', 'gaan', 'wordt', 'komen', 'zal']
        nb_dutch = sum(1 for word in words if word in dutch_words)
        logger.info(
            f"Dutch words found: {[word for word in words if word in dutch_words]}")

        logger.info(
            f"number of Dutch words: {nb_dutch}, number of English words: {nb_en}")

        # Set language based on majority
        if nb_dutch > nb_en:
            return "Dutch"
        elif nb_en > nb_dutch:
            return "English"
        else:
            # Default to English if no clear majority
            return "English"

    def process_query(self, query: str, conversation_id: Optional[str] = None) -> str:
        """Process query with proper transaction handling and context management."""
        try:
            # Start a new transaction for the entire process if none exists
            if not self.session.in_transaction():
                self.session.begin()

            try:
                # Check cache first
                cached_response = self._check_query_cache(query)
                if cached_response:
                    logger.info("Using cached response")
                    return cached_response

                # Detect language of the query
                detected_language = self._detect_language(query)
                logger.info(f"Detected language: {detected_language}")

                # Get conversation history if conversation_id is provided
                conversation_context = ""
                previous_identifiers = set()  # Store any identifiers from previous questions
                if conversation_id and self.session:
                    try:
                        # Query previous messages from this conversation
                        previous_messages = self.session.query(UserMemory)\
                            .filter(
                                UserMemory.conversation_id == conversation_id,
                                UserMemory.user_id == self.tenant_code,
                                UserMemory.is_active == True
                        )\
                            .order_by(UserMemory.message_order.desc())\
                            .limit(10)\
                            .all()

                        if previous_messages:
                            conversation_context = "Previous conversation history:\n\n"
                            # Process in chronological order (oldest first)
                            for msg in reversed(previous_messages):
                                conversation_context += f"User: {msg.query}\n"
                                conversation_context += f"Assistant: {msg.response}\n\n"

                                # Extract identifiers from previous messages
                                # Look for phone numbers, IDs, or other specific identifiers
                                # Match 10 or more digits
                                phone_pattern = r'\b\d{10,}\b'
                                identifiers = re.findall(
                                    phone_pattern, msg.query)
                                previous_identifiers.update(identifiers)
                                logger.info(
                                    f"Found phone number identifiers: {identifiers}")

                                # Look for other common identifier patterns
                                # UUID pattern
                                id_pattern = r'\b[A-Za-z0-9]{8}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{12}\b'
                                identifiers = re.findall(id_pattern, msg.query)
                                previous_identifiers.update(identifiers)
                                logger.info(
                                    f"Found UUID identifiers: {identifiers}")

                            logger.info(
                                f"Retrieved {len(previous_messages)} previous messages from conversation {conversation_id}")
                            if previous_identifiers:
                                logger.info(
                                    f"Found previous identifiers: {previous_identifiers}")
                    except Exception as e:
                        logger.error(
                            f"Error retrieving conversation history: {e}")
                        logger.exception("Detailed error:")

                # 1. Search for relevant transcriptions using vector search with circuit breaker
                similar_transcripts = self.search_transcriptions_safe(query)

                if not similar_transcripts:
                    return "I couldn't find any relevant transcriptions to answer your question. Could you please provide more details or try a different query?"

                # 2. Check for entity corrections (e.g., partial names)
                entity_corrections = self._detect_entity_corrections(
                    query, similar_transcripts)

                # 3. Generate SQL query based on the natural language query and similar transcripts
                # Add retry logic for SQL generation
                max_sql_retries = 3
                sql_retry_count = 0
                last_sql_error = None
                sql_query = None

                while sql_retry_count < max_sql_retries:
                    try:
                        # Create error context for SQL retries
                        error_context = ""
                        if last_sql_error and sql_retry_count > 0:
                            logger.info(
                                f"SQL generation retry attempt {sql_retry_count+1}/{max_sql_retries}")
                            error_context = f"Previous SQL error: {last_sql_error}"

                        # Add context about previous identifiers to the SQL generation
                        identifier_context = ""
                        if previous_identifiers:
                            identifier_context = "\nIMPORTANT: The previous conversation mentioned these identifiers: " + \
                                ", ".join(previous_identifiers) + \
                                "\nMake sure to include these in your WHERE clause if relevant to the current query."

                        # Generate SQL query with error context and identifier context
                        sql_query = self.generate_sql_query(
                            query,
                            similar_transcripts,
                            error_context + identifier_context
                        )

                        # Try to execute the query - this will validate it
                        sql_results = self._execute_sql_query(sql_query)

                        # If we reach here, the query succeeded
                        logger.info(
                            f"SQL query executed successfully after {sql_retry_count+1} attempts")
                        break

                    except ValueError as e:
                        # Store the error for the next retry
                        last_sql_error = str(e)
                        sql_retry_count += 1
                        logger.warning(
                            f"SQL query execution failed (attempt {sql_retry_count}/{max_sql_retries}): {last_sql_error}")

                        # If we've reached max retries, try with a simplified approach
                        if sql_retry_count >= max_sql_retries:
                            logger.error(
                                f"Failed to generate valid SQL after {max_sql_retries} attempts")
                            # ... rest of the error handling code ...

                # If we somehow got here without a successful query or explicit error handling
                if sql_query is None:
                    logger.error(
                        "SQL query generation failed with no valid result")
                    sql_results = "I couldn't generate a valid SQL query to answer your question."

                # Log the generated SQL query
                logger.info(f"Generated SQL query: {sql_query}")

                # Log the first 2 rows of SQL results for debugging
                if sql_results and isinstance(sql_results, str):
                    # Split the results by newlines to get individual rows
                    result_lines = sql_results.split('\n')
                    # Log the first 2 rows (or fewer if there are less than 2 rows)
                    rows_to_log = min(2, len(result_lines))
                    logger.info(f"First {rows_to_log} rows of SQL results:")
                    for i in range(rows_to_log):
                        logger.info(f"Row {i+1}: {result_lines[i]}")
                else:
                    logger.info(
                        "No SQL results to log or results are not in string format")

                # 4. Format examples for the agent
                examples_text = self._format_example_transcriptions(
                    similar_transcripts)

                # 5. Initialize agents if needed
                self._initialize_agents_if_needed()

                # Get the agent
                agent = self._get_agent()
                if agent:
                    logger.info("Using agent for processing query")
                else:
                    logger.error("No agent available for processing query")

                # 6. Execute SQL query and get results - already done above with retry logic

                # 7. Get conversation history if conversation_id is provided
                conversation_history = ""
                if conversation_id and self.session:
                    try:
                        # Query previous messages from this conversation
                        previous_messages = self.session.query(UserMemory)\
                            .filter(
                                UserMemory.conversation_id == conversation_id,
                                UserMemory.user_id == self.tenant_code,
                                UserMemory.is_active == True
                        )\
                            .order_by(UserMemory.message_order.desc())\
                            .limit(10)\
                            .all()

                        if previous_messages:
                            conversation_history = "Previous conversation history:\n\n"
                            # Process in chronological order (oldest first)
                            for msg in reversed(previous_messages):
                                conversation_history += f"User: {msg.query}\n"
                                conversation_history += f"Assistant: {msg.response}\n\n"

                            logger.info(
                                f"Retrieved {len(previous_messages)} previous messages from conversation {conversation_id}")
                    except Exception as e:
                        logger.error(
                            f"Error retrieving conversation history: {e}")
                        logger.exception("Detailed error:")
                        # Continue without conversation history if there's an error

                # 8. Create a prompt with the context, conversation history, query, entity corrections, and SQL results
                prompt = f"""
                I need to answer this user question:
                "{query}"
                
                {conversation_history}
                
                {entity_corrections}
                
                Here are some relevant call transcriptions that might help:
                {examples_text}
                
                SQL Query Results:
                {sql_results}
                
                Please provide a comprehensive answer based ONLY on the SQL results above.
                The call transcriptions are provided only to help with context, but your response should be based entirely on the SQL query results.
                If the SQL results don't contain enough information to answer the question fully, 
                acknowledge this limitation in your response. Make sure to answer in the language of the question!
                
                Make your response user-friendly by avoiding SQL terminology - present the data in plain language without mentioning SQL, queries, or database terms.
                
                IMPORTANT: Respond in {detected_language} language to match the user's question language.
                
                {'' if not entity_corrections else 'IMPORTANT: Make sure to acknowledge the entity correction at the beginning of your response.'}
                """

                # 9. Use the agent to generate a response
                try:
                    response_obj = agent.run(prompt)

                    # Extract the text content from the RunResponse object
                    if hasattr(response_obj, 'content'):
                        response = response_obj.content
                    elif isinstance(response_obj, str):
                        response = response_obj
                    else:
                        # Convert the response object to a string if it's not already a string
                        response = str(response_obj)

                    logger.info(f"Final response: {response[:100]}...")

                    # Cache the response
                    self._cache_query_response(query, response)

                    # Store conversation if needed
                    if conversation_id:
                        self._store_conversation(
                            conversation_id, query, response)

                    # Commit only if we started the transaction
                    if not self.session.in_transaction():
                        self.session.commit()
                    return response

                except Exception as e:
                    logger.error(f"Error running agent: {e}")
                    logger.exception("Detailed error:")
                    return f"I'm sorry, I couldn't process your query. Error: {str(e)}"

            except Exception as e:
                if not self.session.in_transaction():
                    self.session.rollback()
                raise ValueError(f"Error processing query: {str(e)}")

        except Exception as e:
            raise ValueError(f"Error processing query: {str(e)}")

    def _check_query_cache(self, query: str) -> str:
        """Check if a similar query exists in the cache"""
        try:
            if not self.session:
                return None

            # First, rollback any failed transaction
            self.session.rollback()

            # Get the actual column names from the query_cache table
            try:
                # Check if the table exists and get its columns
                columns_result = self.session.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'query_cache'
                """)).fetchall()

                column_names = [col[0] for col in columns_result]
                logger.info(f"Found query_cache columns: {column_names}")

                # Normalize query for better matching
                normalized_query = self._normalize_query(query)

                # Check for exact match first - use the correct column names
                cache_result = self.session.execute(
                    text("""
                    SELECT result, created_at
                    FROM query_cache
                    WHERE tenant_code = :tenant_code
                      AND question ILIKE :query
                      AND created_at > NOW() - INTERVAL '5 minutes'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """),
                    {
                        "tenant_code": self.tenant_code,
                        "query": f"%{normalized_query}%"
                    }
                ).fetchone()

                if cache_result:
                    logger.info(
                        f"Found exact match in cache from {cache_result[1]}")
                    return cache_result[0]

                return None

            except Exception as e:
                logger.error(f"Error checking query_cache schema: {e}")
                logger.exception("Detailed error:")
                return None

        except Exception as e:
            logger.error(f"Error checking query cache: {e}")
            logger.exception("Detailed error:")
            # Make sure to rollback on error
            if self.session:
                self.session.rollback()
            return None

    def _normalize_query(self, query: str) -> str:
        """Normalize a query for better cache matching"""
        # Convert to lowercase
        normalized = query.lower()

        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def _query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        # Simple word overlap similarity
        words1 = set(query1.split())
        words2 = set(query2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _cache_query_response(self, query: str, response: str) -> None:
        """Cache a query and its response"""
        try:
            if not self.session:
                return

            # First, rollback any failed transaction
            self.session.rollback()

            # Create a hash key for the query
            hash_key = self._create_hash_key(query)

            # Calculate expiry date (7 days from now)
            expires_at = datetime.utcnow() + timedelta(days=7)

            # Insert into cache using the correct column names
            # Include hash_key in the INSERT statement
            stmt = text("""
                INSERT INTO query_cache 
                (tenant_code, question, sql, result, created_at, expires_at, hash_key)
                VALUES 
                (:tenant_code, :query, :query, :response, NOW(), :expires_at, :hash_key)
                ON CONFLICT (hash_key, tenant_code) 
                DO UPDATE SET 
                    result = EXCLUDED.result,
                    created_at = CURRENT_TIMESTAMP,
                    expires_at = EXCLUDED.expires_at
            """)

            self.session.execute(stmt, {
                "tenant_code": self.tenant_code,
                "query": query,
                "response": response,
                "expires_at": expires_at,
                "hash_key": hash_key
            })

            self.session.commit()
            logger.info("Cached query response")

        except Exception as e:
            logger.error(f"Error caching query response: {e}")
            logger.exception("Detailed error:")
            # Make sure to rollback on error
            if self.session:
                self.session.rollback()

    def _create_hash_key(self, text: str) -> str:
        """Create a hash key for caching"""
        # Normalize the text by removing extra whitespace
        normalized = ' '.join(text.lower().split())
        # Create SHA-256 hash
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _check_sql_cache(self, sql_query: str) -> str:
        """Check if SQL query results are cached"""
        try:
            if not self.session:
                return None

            # First, rollback any failed transaction
            self.session.rollback()

            # Normalize SQL query
            normalized_sql = self._normalize_sql(sql_query)

            # Check cache using the correct column names
            cache_result = self.session.execute(
                text("""
                SELECT result, created_at
                FROM query_cache
                WHERE tenant_code = :tenant_code
                  AND sql ILIKE :sql_query
                  AND created_at > NOW() - INTERVAL '5 minutes'
                ORDER BY created_at DESC
                LIMIT 1
                """),
                {
                    "tenant_code": self.tenant_code,
                    "sql_query": f"%{normalized_sql}%"
                }
            ).fetchone()

            if cache_result:
                logger.info(
                    f"Found SQL query in cache from {cache_result[1]}")
                return cache_result[0]

            return None

        except Exception as e:
            logger.error(f"Error checking SQL cache: {e}")
            logger.exception("Detailed error:")
            # Make sure to rollback on error
            if self.session:
                self.session.rollback()
            return None

    def _normalize_sql(self, sql_query: str) -> str:
        """Normalize SQL query for better cache matching"""
        # Convert to lowercase
        normalized = sql_query.lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def _cache_sql_results(self, sql_query: str, results: str) -> None:
        """Cache SQL query results"""
        try:
            if not self.session:
                return

            # First, rollback any failed transaction
            self.session.rollback()

            # Create a hash key for the query
            hash_key = self._create_hash_key(sql_query)

            # Calculate expiry date (7 days from now)
            expires_at = datetime.utcnow() + timedelta(days=7)

            # Insert into cache with SQL prefix to distinguish from regular queries
            # Include hash_key in the INSERT statement
            stmt = text("""
                INSERT INTO query_cache 
                (tenant_code, question, sql, result, created_at, expires_at, hash_key)
                VALUES 
                (:tenant_code, :sql_query, :sql_query, :results, NOW(), :expires_at, :hash_key)
                ON CONFLICT (hash_key, tenant_code) 
                DO UPDATE SET 
                    result = EXCLUDED.result,
                    created_at = CURRENT_TIMESTAMP,
                    expires_at = EXCLUDED.expires_at
            """)

            self.session.execute(stmt, {
                "tenant_code": self.tenant_code,
                "sql_query": sql_query,
                "results": results,
                "expires_at": expires_at,
                "hash_key": hash_key
            })

            self.session.commit()
            logger.info("Cached SQL results")

        except Exception as e:
            logger.error(f"Error caching SQL results: {e}")
            logger.exception("Detailed error:")
            # Make sure to rollback on error
            if self.session:
                self.session.rollback()

    def _detect_entity_corrections(self, query: str, transcripts: List[Dict[str, Any]]) -> str:
        """Detect and suggest corrections for entity names in the query"""
        try:
            # Extract potential entity names from the query
            # This is a simple implementation - in a production system, you might use NER
            query_words = query.lower().split()
            potential_entities = []

            # Look for capitalized words in transcripts that might be entities
            entity_candidates = {}

            for transcript in transcripts:
                text = transcript.get("text", "")
                # Extract metadata
                metadata = transcript.get("metadata", {})
                summary = metadata.get("summary", "")

                # Combine text and summary for entity extraction
                content = f"{text} {summary}"

                # Simple entity extraction - look for capitalized words
                words = content.split()
                for i, word in enumerate(words):
                    if word and word[0].isupper() and i < len(words) - 1:
                        # Check if this might be a name (two capitalized words in sequence)
                        if i + 1 < len(words) and words[i+1] and words[i+1][0].isupper():
                            potential_entity = f"{word} {words[i+1]}"
                            entity_candidates[potential_entity.lower(
                            )] = potential_entity
                        else:
                            entity_candidates[word.lower()] = word

            # Check if any words in the query might be partial matches to entities
            corrections = []
            for query_word in query_words:
                if len(query_word) >= 3:  # Only consider words of reasonable length
                    for entity_lower, entity_original in entity_candidates.items():
                        # Check for partial matches
                        if query_word in entity_lower and query_word != entity_lower:
                            corrections.append((query_word, entity_original))

            # Generate correction text
            if corrections:
                correction_text = "I noticed you mentioned: "
                for partial, full in corrections:
                    correction_text += f"'{partial}' which I'm assuming refers to '{full}'. "
                return correction_text

            return ""

        except Exception as e:
            logger.error(f"Error detecting entity corrections: {e}")
            logger.exception("Detailed error:")
            return ""

    def _format_example_transcriptions(self, transcripts: List[Dict[str, Any]]) -> str:
        """Format transcriptions as examples for the agent"""
        formatted_examples = ""

        for i, transcript in enumerate(transcripts):
            # Extract text and metadata
            text = transcript.get("text", "")
            metadata = transcript.get("metadata", {})

            # Format the example with metadata
            formatted_examples += f"Example {i+1}:\n"

            # Include all metadata fields
            for key, value in metadata.items():
                # Format the value based on its type
                if isinstance(value, float) and key == "score":
                    formatted_value = f"{value:.4f}"
                elif isinstance(value, dict) or isinstance(value, list):
                    formatted_value = str(value)
                else:
                    formatted_value = str(value)

                formatted_examples += f"{key}: {formatted_value}\n"

            formatted_examples += f"Text:\n{text}\n\n"

        return formatted_examples

    def generate_sql_query(self, query: str, similar_transcripts: List[Dict[str, Any]], error_context: str = "") -> str:
        """Generate a SQL query based on the user's natural language query, similar transcripts, and error context"""
        try:
            # Initialize agents if needed
            self._initialize_agents_if_needed()

            # Get the SQL generation agent
            sql_agent = AgentFactory.create_sql_agent(self.tenant_code)

            if sql_agent is None:
                logger.error("No SQL agent found")
                return ""

            # Format examples for the agent
            examples_text = self._format_example_transcriptions(
                similar_transcripts)

            # Get database schema information
            db_schema = self._get_database_schema()

            # Extract entity names from the query for better filtering
            entity_names = self._extract_entity_names_from_query(query)
            entity_filter_hint = ""
            if entity_names:
                entity_filter_hint = f"""
                IMPORTANT: The query is asking about: {', '.join(entity_names)}
                Make sure to filter results to only include records related to these entities.
                Use ILIKE with wildcards for partial name matching (e.g., '%Hendrik%Stuiver%').
                For company names or identifiers, check both telephone_number and clid fields.
                """

            # Create a prompt with the context, query, and error information
            prompt = f"""
            Generate a PostgreSQL query for this user question:
            "{query}"
            
            Use these example transcripts as reference:
            {examples_text}
            
            Database schema information:
            {db_schema}
            
            {entity_filter_hint}
            
            {error_context}
            
            CRITICAL REQUIREMENTS:
            1. ALWAYS include "tenant_code = '{self.tenant_code}'" in the WHERE clause
            2. NEVER use DELETE, UPDATE, INSERT or other data-modifying statements
            3. Use PostgreSQL syntax
            4. ALWAYS use LIKE with wildcards for text matching, especially for:
            - Phone numbers: Use "telephone_number LIKE '%{{phone_number}}%'" instead of equals
            - Names: Use "name LIKE '%{{name}}%'" instead of equals
            - Any entity identifiers: Always use LIKE with wildcards for flexible matching
            - Addresses: Use LIKE with wildcards for address components
            5. For people's names, use ILIKE with wildcards
            6. ONLY use tables that exist in the schema information provided
            7. The main table for call data is "transcription" (NOT "transcripts")
            8. DO NOT add LIMIT clauses unless specifically needed for the query
            9. SELECT only the most relevant columns for the question
            10. Focus on finding SPECIFIC records that answer the question, not general statistics
            11. DO NOT add semicolons in the middle of the query
            12. When checking for telephone numbers, ALWAYS check both the telephone_number and clid fields (e.g., "WHERE (telephone_number LIKE '%{{phone_number}}%' OR clid LIKE '%{{phone_number}}%')")
            13. For call direction analysis, use the following categories:
                - 'IN': Incoming calls (calls received by the system)
                - 'OUT': Outgoing calls (calls initiated by the system)
                - 'LOCAL': Local/internal calls (calls within the system)
                - None: Unspecified or unknown call direction
            14. THE 'score' COLUMN DOES NOT EXIST. Use sentiment data instead.
            15. To analyze sentiment, use the clean_sentiment field with values like 'positief', 'negatief', 'neutral'
            16. For calculations involving sentiment, use CASE statements like:
                CASE WHEN clean_sentiment = 'positief' THEN 1 WHEN clean_sentiment = 'negatief' THEN -1 ELSE 0 END
            17. When the query mentions a specific company, person, or entity:
                - ALWAYS add a WHERE clause to filter for that entity
                - Check both telephone_number and clid fields using ILIKE with wildcards
                - Example: "WHERE (telephone_number ILIKE '%{{entity}}%' OR clid ILIKE '%{{entity}}%')"
                - This applies regardless of the query language (English or Dutch)
            18. For entity-specific queries, focus on finding records that match the entity exactly, not general statistics
            
            Generate ONLY the SQL query, no explanations.
            """

            # Generate SQL query using run method instead of get_response
            response_obj = sql_agent.run(prompt)

            # Extract the text content from the response object
            if hasattr(response_obj, 'content'):
                sql_query = response_obj.content
            elif isinstance(response_obj, str):
                sql_query = response_obj
            else:
                # Convert the response object to a string if it's not already a string
                sql_query = str(response_obj)

            # Clean up the SQL query
            sql_query = self._sanitize_sql_query(sql_query)

            return sql_query
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            logger.exception("Detailed error:")
            raise ValueError(f"Error generating SQL query: {str(e)}")

    def _sanitize_sql_query(self, sql_query: str) -> str:
        """Sanitize SQL query to fix common issues"""
        try:
            # Remove any markdown formatting
            if sql_query.startswith("```") and sql_query.endswith("```"):
                sql_query = sql_query[3:-3].strip()
            elif sql_query.startswith("```sql") and sql_query.endswith("```"):
                sql_query = sql_query[6:-3].strip()
            elif sql_query.startswith("```"):
                # Handle case where only opening ``` is present
                sql_query = sql_query[3:].strip()

            # Remove any remaining markdown formatting
            sql_query = re.sub(r'^```sql\s*', '', sql_query)
            sql_query = re.sub(r'^```\s*', '', sql_query)
            sql_query = re.sub(r'\s*```$', '', sql_query)

            if sql_query.startswith("sql"):
                sql_query = sql_query[3:].strip()

            # Fix issue with semicolons before LIMIT
            sql_query = sql_query.replace("; LIMIT", " LIMIT")

            # Remove any trailing semicolons
            sql_query = sql_query.rstrip(";")

            # Ensure there's only one LIMIT clause at the end
            if "LIMIT" in sql_query:
                # Split by LIMIT, keeping only the first part and the last LIMIT clause
                parts = sql_query.split("LIMIT")
                if len(parts) > 2:
                    sql_query = parts[0] + "LIMIT" + parts[-1]

            # Convert exact matches for phone numbers to LIKE statements
            # Look for patterns like: telephone_number = '1234567890' or telephone_number='1234567890'
            phone_pattern = re.compile(
                r"(telephone_number\s*=\s*['\"]([\d\+]+)['\"])")
            matches = phone_pattern.findall(sql_query)
            for match, phone_number in matches:
                replacement = f"telephone_number LIKE '%{phone_number}%'"
                sql_query = sql_query.replace(match, replacement)

            # Convert exact matches for other common entity fields to LIKE statements
            entity_fields = ['name', 'address', 'email',
                             'clid', 'customer_id', 'account_number']
            for field in entity_fields:
                # Match pattern: field = 'value' or field='value'
                field_pattern = re.compile(
                    f"({field}\\s*=\\s*['\"](.*?)['\"])")
                matches = field_pattern.findall(sql_query)
                for match, value in matches:
                    replacement = f"{field} LIKE '%{value}%'"
                    sql_query = sql_query.replace(match, replacement)

            # Remove any explanatory text after the SQL query
            # Look for common patterns like "This query will..." or "The query..."
            explanatory_pattern = re.compile(
                r'```\s*(This query|The query|This SQL|The SQL).*$', re.DOTALL | re.IGNORECASE)
            sql_query = explanatory_pattern.sub('', sql_query).strip()

            # Remove any remaining backticks
            sql_query = sql_query.replace('```', '').strip()

            return sql_query
        except Exception as e:
            logger.error(f"Error sanitizing SQL query: {e}")
            logger.exception("Detailed error:")
            return sql_query  # Return original query if sanitization fails

    def _get_database_schema(self) -> str:
        """Get database schema with proper transaction handling."""
        try:
            # Clean up any existing transaction
            if self.session.in_transaction():
                self.session.rollback()

            # Start fresh transaction
            self.session.begin()
            try:
                # Get table names and their columns
                result = self.session.execute(text("""
                    SELECT 
                        t.table_name,
                        string_agg(c.column_name || ' ' || c.data_type, ', ' ORDER BY c.ordinal_position) as columns
                    FROM 
                        information_schema.tables t
                    JOIN 
                        information_schema.columns c ON t.table_name = c.table_name
                    WHERE 
                        t.table_schema = 'public'
                    GROUP BY 
                        t.table_name
                    ORDER BY 
                        t.table_name
                """))

                # Format the schema information
                schema_info = []
                for row in result:
                    table_name = row[0]
                    columns = row[1]
                    schema_info.append(
                        f"Table: {table_name}\nColumns: {columns}\n")

                self.session.commit()
                return "\n".join(schema_info)
            except Exception as e:
                self.session.rollback()
                raise ValueError(f"Error getting database schema: {str(e)}")
        except Exception as e:
            if self.session.in_transaction():
                self.session.rollback()
            raise ValueError(f"Error getting database schema: {str(e)}")

    def _get_restricted_tables(self) -> List[str]:
        """Get list of restricted table names"""
        try:
            if not self.session:
                return ["users", "user_memory", "query_cache", "query_performance"]

            # Try to get restricted tables from database
            try:
                result = self.session.execute(text("""
                    SELECT table_name 
                    FROM restricted_tables 
                    WHERE added_by = 'system'
                """))
                return [row[0] for row in result]
            except Exception:
                # Fallback to minimum set of restricted tables
                return ["users", "user_memory", "query_cache", "query_performance"]

        except Exception as e:
            logger.error(f"Error getting restricted tables: {e}")
            logger.exception("Detailed error:")
            # Fallback to minimum set of restricted tables
            return ["users", "user_memory", "query_cache", "query_performance"]

    def _extract_entity_names_from_query(self, query: str) -> List[str]:
        """Extract potential entity names from the query"""
        # Simple extraction based on capitalized words
        words = query.split()
        entities = []

        for i, word in enumerate(words):
            # Check if word starts with a capital letter
            if word and word[0].isupper():
                entities.append(word)
            # Check for words that might be part of names but not capitalized
            elif word.lower() in ['hendrik', 'stu', 'stuiver', 'gerbrand', 'luc']:
                entities.append(word)

        return entities

    def _execute_sql_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute SQL query with proper transaction handling and parameter binding."""
        if params is None:
            params = {}

        try:
            # Clean up any existing transaction
            if self.session.in_transaction():
                self.session.rollback()

            # Start fresh transaction
            self.session.begin()
            try:
                # Ensure tenant_code is always included
                if 'tenant_code' not in params:
                    params['tenant_code'] = self.tenant_code

                # Execute query with proper parameter binding
                result = self.session.execute(text(query), params)
                self.session.commit()

                # Convert result to list of dictionaries
                if result.returns_rows:
                    return [dict(row._mapping) for row in result]
                return []

            except Exception as e:
                self.session.rollback()
                if "statement timeout" in str(e).lower():
                    raise ValueError(
                        "Query execution timed out. Please try again with a more specific query.")
                raise ValueError(f"Error executing query: {str(e)}")
        except Exception as e:
            if self.session.in_transaction():
                self.session.rollback()
            raise ValueError(f"Error executing query: {str(e)}")

    def _store_conversation(self, conversation_id: str, query: str, response: str) -> None:
        """Store conversation in database with proper id handling."""
        try:
            # Clean up any existing transaction
            if self.session.in_transaction():
                self.session.rollback()

            # Start fresh transaction
            self.session.begin()
            try:
                # Get the next available message_order for this conversation
                last_message = self.session.execute(
                    text("""
                        SELECT MAX(message_order) 
                        FROM user_memory 
                        WHERE conversation_id = :conversation_id
                    """),
                    {"conversation_id": conversation_id}
                ).fetchone()

                next_message_order = (
                    last_message[0] + 1) if last_message and last_message[0] is not None else 0

                # Check if id column is integer type
                result = self.session.execute(text("""
                    SELECT data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'user_memory' 
                    AND column_name = 'id'
                """)).fetchone()

                id_is_integer = result and result[0].startswith('int')

                # Generate appropriate ID based on column type
                if id_is_integer:
                    # Get the current max ID
                    max_id_result = self.session.execute(text("""
                        SELECT MAX(id) FROM user_memory
                    """)).fetchone()

                    # Start from max_id + 1 or 1000000 if no records exist
                    memory_id = max(int(max_id_result[0] or 0) + 1, 1000000)
                else:
                    # Use UUID for non-integer id columns
                    memory_id = str(uuid.uuid4())

                # Store the conversation
                self.session.execute(
                    text("""
                        INSERT INTO user_memory 
                        (id, user_id, conversation_id, query, response, message_order, is_active, timestamp, followup_questions, title, expires_at)
                        VALUES 
                        (:id, :user_id, :conversation_id, :query, :response, :message_order, TRUE, NOW(), :followup_questions, :title, NOW() + INTERVAL '30 days')
                    """),
                    {
                        'id': memory_id,
                        'user_id': self.tenant_code,
                        'conversation_id': conversation_id,
                        'query': f"Conversation history for query: {query}",
                        'response': response,
                        'message_order': next_message_order,
                        'followup_questions': [],
                        'title': 'Conversation History'
                    }
                )
                self.session.commit()
            except Exception as e:
                self.session.rollback()
                raise ValueError(f"Error storing conversation: {str(e)}")
        except Exception as e:
            if self.session.in_transaction():
                self.session.rollback()
            raise ValueError(f"Error storing conversation: {str(e)}")
