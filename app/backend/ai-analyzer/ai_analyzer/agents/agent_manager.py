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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentManager:
    """Manager for Agno agents"""

    def __init__(self, tenant_code: str, session=None):
        """Initialize the agent manager"""
        self.tenant_code = tenant_code
        self.session = session
        logger.info(f"Initialized AgentManager for tenant: {tenant_code}")

        # Don't initialize agent_team in constructor to avoid errors
        self.agent_team = None
        self.followup_agent = None

        # Ensure tables exist
        if self.session:
            try:
                # Get engine from session
                engine = self.session.get_bind()
                # Create tables if they don't exist
                create_agent_tables(engine)
            except Exception as e:
                logger.error(f"Error creating tables: {e}")
                logger.exception("Detailed error:")

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
            
            Please analyze this response:
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

            # Create a prompt for generating follow-up questions
            prompt = f"""
            Based on the following conversation, generate 3 follow-up questions that would help continue and deepen the conversation:
            
            {conversation}
            
            FOLLOW-UP QUESTIONS:
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

    def search_transcriptions(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant transcriptions using vector search"""
        try:
            # Create the agent if it doesn't exist
            agent = self._get_agent()

            if agent is None:
                logger.error("No agent found")
                return []

            # Add more detailed debugging
            if hasattr(agent, 'knowledge'):
                logger.info(f"Agent has knowledge: {type(agent.knowledge)}")
                if hasattr(agent.knowledge, 'retriever'):
                    logger.info(
                        f"Knowledge has retriever: {type(agent.knowledge.retriever)}")
                else:
                    logger.error("Knowledge doesn't have retriever attribute")
            else:
                logger.error("Agent doesn't have knowledge attribute")

            # Try to search directly with the Qdrant client as a fallback
            try:
                # If agent retriever fails, try direct Qdrant search
                from ai_analyzer.utils import get_qdrant_client
                from fastembed import TextEmbedding
                import numpy as np

                qdrant_client = get_qdrant_client()
                collection_name = f"tenant_{self.tenant_code}"

                # Get collection info to find the vector name
                collection_info = qdrant_client.get_collection(collection_name)
                vector_names = list(
                    collection_info.config.params.vectors.keys())

                if not vector_names:
                    logger.error(
                        f"No vector names found in collection {collection_name}")
                    return []

                vector_name = vector_names[0]  # Use the first vector name
                logger.info(f"Using vector name: {vector_name} for search")

                # Generate embeddings for the query
                embedding_model = TextEmbedding(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                embeddings = list(embedding_model.embed([query]))
                query_vector = embeddings[0].tolist()

                # Search Qdrant directly with vector name specified and return more results
                search_results = qdrant_client.search(
                    collection_name=collection_name,
                    # Specify vector name
                    query_vector=(vector_name, query_vector),
                    limit=2
                )

                # Log the search results
                logger.info(f"Search query: '{query}'")
                logger.info(f"Found {len(search_results)} results in Qdrant")

                # Convert results to the expected format
                formatted_results = []
                for i, result in enumerate(search_results):
                    # Log each result with its score and text
                    logger.info(f"Result {i+1} (score: {result.score:.4f}):")

                    # Log the entire payload for debugging
                    logger.info(
                        f"  Full payload keys: {list(result.payload.keys())}")

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
                                # Log all payload values to see what's available
                                logger.info(
                                    f"  Full payload: {result.payload}")

                    # Log the first 100 characters of the text to avoid huge logs
                    logger.info(f"  Text: {text[:100]}..." if len(
                        text) > 100 else f"  Text: {text}")

                    # If we still don't have text, use a placeholder
                    if not text:
                        text = "No text content found in this result."

                    formatted_results.append({
                        "text": text,
                        "metadata": {
                            "id": result.payload.get("id", ""),
                            "call_id": result.payload.get("call_id", ""),
                            "timestamp": result.payload.get("timestamp", ""),
                            "score": result.score
                        }
                    })

                logger.info(
                    f"Found {len(formatted_results)} results using direct Qdrant search")
                return formatted_results

            except Exception as e:
                logger.error(f"Error in direct Qdrant search: {e}")
                logger.exception("Detailed error:")

                # Return empty results if both methods fail
                return []

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

    def process_query(self, query: str) -> str:
        """Process a user query using hybrid approach with Agno multiagent"""
        try:
            # Check cache first
            cached_response = self._check_query_cache(query)
            if cached_response:
                logger.info("Using cached response")
                return cached_response

            # 1. Search for relevant transcriptions using vector search
            similar_transcripts = self.search_transcriptions(query)

            if not similar_transcripts:
                return "I couldn't find any relevant transcriptions to answer your question. Could you please provide more details or try a different query?"

            # 2. Check for entity corrections (e.g., partial names)
            entity_corrections = self._detect_entity_corrections(
                query, similar_transcripts)

            # 3. Generate SQL query based on the natural language query and similar transcripts
            sql_query = self.generate_sql_query(query, similar_transcripts)

            # Log the generated SQL query
            logger.info(f"Generated SQL query: {sql_query}")

            # 4. Format examples for the agent
            examples_text = self._format_example_transcriptions(
                similar_transcripts)

            # 5. Initialize agents if needed
            self._initialize_agents_if_needed()

            # Get the agent
            agent = self._get_agent()

            if agent is None:
                return "I'm sorry, I couldn't process your query. The agent could not be initialized."

            # 6. Execute SQL query if available
            sql_results = ""
            if sql_query and self.session:
                try:
                    # Check SQL query cache
                    cached_sql_results = self._check_sql_cache(sql_query)
                    if cached_sql_results:
                        logger.info("Using cached SQL results")
                        sql_results = cached_sql_results
                    else:
                        logger.info(f"Executing SQL query: {sql_query}")
                        result = self.session.execute(text(sql_query))
                        rows = result.fetchall()
                        if rows:
                            # Format SQL results
                            columns = result.keys()
                            sql_results = "SQL Query Results:\n"
                            sql_results += "\n".join(
                                [f"{', '.join(columns)}", "-" * 40])
                            for row in rows[:10]:  # Limit to 10 rows
                                sql_results += f"\n{', '.join(str(val) for val in row)}"
                            if len(rows) > 10:
                                sql_results += f"\n... and {len(rows) - 10} more rows"

                            # Cache SQL results
                            self._cache_sql_results(sql_query, sql_results)

                            # Log the SQL results
                            logger.info(f"SQL query results: {sql_results}")
                        else:
                            logger.info("SQL query returned no results")
                except Exception as e:
                    logger.error(f"Error executing SQL query: {e}")
                    logger.exception("Detailed error:")
                    sql_results = f"Error executing SQL query: {str(e)}"

            # 7. Create a prompt with the context, query, entity corrections, and SQL results
            prompt = f"""
            I need to answer this user question:
            "{query}"
            
            {entity_corrections}
            
            Here are some relevant call transcriptions that might help:
            {examples_text}
            
            SQL Query Results:
            {sql_results}
            
            Please provide a comprehensive answer based on the transcriptions and SQL results.
            If the transcriptions don't contain enough information to answer the question fully, 
            acknowledge this limitation in your response.
            
            {'' if not entity_corrections else 'IMPORTANT: Make sure to acknowledge the entity correction at the beginning of your response.'}
            """

            # 8. Use the agent to generate a response
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

                return response

            except Exception as e:
                logger.error(f"Error running agent: {e}")
                logger.exception("Detailed error:")
                return f"I'm sorry, I couldn't process your query. Error: {str(e)}"

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.exception("Detailed error:")
            return f"I'm sorry, I couldn't process your query. Error: {str(e)}"

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
            formatted_examples += f"ID: {metadata.get('id', 'unknown')}\n"
            formatted_examples += f"Call ID: {metadata.get('call_id', 'unknown')}\n"
            formatted_examples += f"Timestamp: {metadata.get('timestamp', 'unknown')}\n"
            formatted_examples += f"Relevance Score: {metadata.get('score', 0):.4f}\n"
            formatted_examples += f"Text:\n{text}\n\n"

        return formatted_examples

    def generate_sql_query(self, query: str, similar_transcripts: List[Dict[str, Any]]) -> str:
        """Generate a SQL query based on the user's natural language query and similar transcripts"""
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
                """

            # Create a prompt with the context and query
            prompt = f"""
            Generate a PostgreSQL query for this user question:
            "{query}"
            
            Use these example transcripts as reference:
            {examples_text}
            
            Database schema information:
            {db_schema}
            
            {entity_filter_hint}
            
            CRITICAL REQUIREMENTS:
            1. ALWAYS include "tenant_code = '{self.tenant_code}'" in the WHERE clause
            2. NEVER use DELETE, UPDATE, INSERT or other data-modifying statements
            3. Use PostgreSQL syntax
            4. If text matching is needed, use ILIKE for case-insensitivity
            5. For people's names, use ILIKE with wildcards
            6. ONLY use tables that exist in the schema information provided
            7. The main table for call data is "transcription" (NOT "transcripts")
            8. LIMIT results to 10-20 rows maximum for readability
            9. SELECT only the most relevant columns for the question
            10. Focus on finding SPECIFIC records that answer the question, not general statistics
            
            Generate ONLY the SQL query, no explanations.
            """

            # Use the agent to generate a SQL query
            try:
                response_obj = sql_agent.run(prompt)

                # Extract the text content from the RunResponse object
                if hasattr(response_obj, 'content'):
                    sql_query = response_obj.content
                elif isinstance(response_obj, str):
                    sql_query = response_obj
                else:
                    # Convert the response object to a string if it's not already a string
                    sql_query = str(response_obj)

                # Clean up the SQL query (remove markdown formatting, etc.)
                sql_query = self._clean_sql_query(sql_query)

                # Validate the SQL query
                sql_query = self._validate_sql_query(sql_query)

                # Add LIMIT if not present
                if "LIMIT" not in sql_query.upper():
                    sql_query += " LIMIT 20;"

                return sql_query

            except Exception as e:
                logger.error(f"Error running SQL agent: {e}")
                logger.exception("Detailed error:")
                return ""

        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            logger.exception("Detailed error:")
            return ""

    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean up the SQL query by removing markdown formatting, etc."""
        # Remove markdown code block formatting
        sql_query = re.sub(r'```sql\s*', '', sql_query)
        sql_query = re.sub(r'```\s*', '', sql_query)

        # Remove leading/trailing whitespace
        sql_query = sql_query.strip()

        return sql_query

    def _validate_sql_query(self, sql_query: str) -> str:
        """Validate the SQL query to ensure it's safe to execute"""
        # Check if the query is empty
        if not sql_query:
            return ""

        # Check if the query contains the tenant code
        if self.tenant_code and f"tenant_code = '{self.tenant_code}'" not in sql_query:
            # Add tenant code filter if not present
            if "WHERE" in sql_query.upper():
                # Add to existing WHERE clause
                sql_query = sql_query.replace(
                    "WHERE", f"WHERE tenant_code = '{self.tenant_code}' AND ", 1)
            else:
                # Add new WHERE clause before ORDER BY, GROUP BY, LIMIT, etc.
                for clause in ["ORDER BY", "GROUP BY", "LIMIT", "HAVING"]:
                    if clause in sql_query.upper():
                        sql_query = sql_query.replace(
                            clause, f"WHERE tenant_code = '{self.tenant_code}' {clause}", 1)
                        break
                else:
                    # If no clause found, add WHERE at the end
                    sql_query += f" WHERE tenant_code = '{self.tenant_code}'"

        # Ensure the query is read-only
        if any(keyword in sql_query.upper() for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]):
            logger.error(f"SQL query contains forbidden keywords: {sql_query}")
            return ""

        return sql_query

    def _get_database_schema(self) -> str:
        """Get database schema information for the SQL agent"""
        try:
            if not self.session:
                return "No database connection available."

            # Get table information from the database
            schema_info = {}

            # Get list of restricted tables
            restricted_tables = self._get_restricted_tables()

            # Query for table names
            result = self.session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))

            tables = [row[0]
                      for row in result if row[0] not in restricted_tables]

            # For each table, get column information
            for table in tables:
                result = self.session.execute(text(f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = '{table}'
                """))

                columns = {row[0]: row[1] for row in result}
                schema_info[table] = columns

            # Format the schema information
            formatted_schema = "Available tables and their columns:\n\n"

            for table, columns in schema_info.items():
                formatted_schema += f"Table: {table}\n"
                formatted_schema += "Columns:\n"
                for column, data_type in columns.items():
                    formatted_schema += f"  - {column} ({data_type})\n"
                formatted_schema += "\n"

            return formatted_schema

        except Exception as e:
            logger.error(f"Error getting database schema: {e}")
            logger.exception("Detailed error:")
            return "Error retrieving database schema."

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
