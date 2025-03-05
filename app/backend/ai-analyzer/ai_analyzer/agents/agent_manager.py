#!/usr/bin/env python3
import os
import logging
from typing import List, Dict, Any, Optional
import uuid
import re
from sqlalchemy import text
import time

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
                    limit=50  # Increased from 5 to 50
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
