#!/usr/bin/env python3
import os
import logging
from typing import List, Dict, Any, Optional
import uuid
from sqlalchemy import text

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.vectordb.qdrant import QdrantDB
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.base import KnowledgeBase

from ai_analyzer.config import config
from ai_analyzer.utils import get_qdrant_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get OpenAI API key from environment or config
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY", config.get("OPENAI_API_KEY", ""))
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment or config")

# Default model for all agents
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"  # or "gpt-3.5-turbo"


class TranscriptionKnowledgeBase(KnowledgeBase):
    """Custom knowledge base for transcription data from Qdrant"""

    def __init__(self, tenant_code: str, vector_db: QdrantDB):
        super().__init__(vector_db=vector_db)
        self.tenant_code = tenant_code
        self.collection_name = f"tenant_{tenant_code}"

    def load(self):
        """No need to load data as it's already in Qdrant"""
        logger.info(
            f"Using existing Qdrant collection: {self.collection_name}")
        return True

    def search_by_id(self, transcription_id: str, limit: int = 1):
        """Search for a specific transcription by ID"""
        # Use Qdrant's filter to find the exact transcription
        filter_condition = {
            "must": [
                {
                    "key": "id",
                    "match": {"value": transcription_id}
                }
            ]
        }

        results = self.vector_db.search(
            query="",  # Empty query since we're filtering by ID
            filter=filter_condition,
            limit=limit
        )

        return results


class AgentFactory:
    """Factory for creating Agno agents"""

    @staticmethod
    def create_knowledge_base(tenant_code: str) -> TranscriptionKnowledgeBase:
        """Create a knowledge base for the specified tenant"""
        # Get Qdrant client
        qdrant_client = get_qdrant_client()

        # Create QdrantDB instance
        vector_db = QdrantDB(
            client=qdrant_client,
            collection_name=f"tenant_{tenant_code}",
            embedder=OpenAIEmbedder(
                api_key=OPENAI_API_KEY, id="text-embedding-3-small"),
        )

        # Create and return knowledge base
        return TranscriptionKnowledgeBase(
            tenant_code=tenant_code,
            vector_db=vector_db
        )

    @staticmethod
    def create_initial_questions_agent(tenant_code: str, model: str = DEFAULT_MODEL) -> Agent:
        """Create an agent that generates initial questions based on transcriptions"""
        knowledge_base = AgentFactory.create_knowledge_base(tenant_code)

        return Agent(
            name="Initial Questions Agent",
            role="Generate initial questions based on call transcriptions",
            model=OpenAIChat(api_key=OPENAI_API_KEY, id=model),
            knowledge=knowledge_base,
            instructions=[
                "You are an expert call analyst who reviews call transcriptions.",
                "Your task is to generate 3-5 insightful questions based on the call transcription.",
                "Focus on questions that would help understand customer needs, pain points, and opportunities.",
                "Questions should be specific to the content of the call, not generic.",
                "Format your response as a numbered list of questions.",
                "Each question should be clear, concise, and directly related to the call content."
            ],
            markdown=True
        )

    @staticmethod
    def create_response_analyzer_agent(tenant_code: str, model: str = DEFAULT_MODEL) -> Agent:
        """Create an agent that analyzes responses to questions"""
        knowledge_base = AgentFactory.create_knowledge_base(tenant_code)

        return Agent(
            name="Response Analyzer",
            role="Analyze responses to questions about call transcriptions",
            model=OpenAIChat(api_key=OPENAI_API_KEY, id=model),
            knowledge=knowledge_base,
            instructions=[
                "You are an expert call analyst who reviews responses to questions about call transcriptions.",
                "Your task is to analyze the response and extract key insights.",
                "Identify patterns, sentiments, and important information from the response.",
                "Provide a brief summary of the response analysis.",
                "Highlight any actionable insights or recommendations."
            ],
            markdown=True
        )

    @staticmethod
    def create_followup_agent(tenant_code: str, model: str = DEFAULT_MODEL) -> Agent:
        """Create an agent that generates follow-up questions based on previous responses"""
        knowledge_base = AgentFactory.create_knowledge_base(tenant_code)

        return Agent(
            name="Follow-up Questions Agent",
            role="Generate follow-up questions based on previous responses",
            model=OpenAIChat(api_key=OPENAI_API_KEY, id=model),
            knowledge=knowledge_base,
            instructions=[
                "You are an expert call analyst who reviews call transcriptions and previous responses.",
                "Your task is to generate 2-3 follow-up questions based on the previous responses.",
                "Questions should dig deeper into areas mentioned in the previous responses.",
                "Focus on areas that need clarification or could reveal more insights.",
                "Format your response as a numbered list of questions.",
                "Each question should be clear, concise, and build upon the previous conversation."
            ],
            markdown=True
        )

    @staticmethod
    def create_agent_team(tenant_code: str, model: str = DEFAULT_MODEL) -> Agent:
        """Create a team of agents for call analysis"""
        initial_questions_agent = AgentFactory.create_initial_questions_agent(
            tenant_code, model)
        response_analyzer_agent = AgentFactory.create_response_analyzer_agent(
            tenant_code, model)
        followup_agent = AgentFactory.create_followup_agent(
            tenant_code, model)

        return Agent(
            name="Call Analysis Team",
            team=[initial_questions_agent,
                  response_analyzer_agent, followup_agent],
            model=OpenAIChat(api_key=OPENAI_API_KEY, id=model),
            instructions=[
                "You are a team of call analysis experts.",
                "Work together to analyze call transcriptions and generate insightful questions.",
                "Use the knowledge base to find relevant information in the transcriptions.",
                "Provide clear, concise, and actionable insights and questions."
            ],
            markdown=True
        )


class AgentManager:
    """Manager for Agno agents"""

    def __init__(self, tenant_code: str, session=None, model: str = DEFAULT_MODEL):
        """Initialize the agent manager"""
        self.tenant_code = tenant_code
        self.session = session
        self.model = model
        logger.info(
            f"Initialized AgentManager for tenant: {tenant_code} with model: {model}")
        self.agent_team = AgentFactory.create_agent_team(tenant_code, model)

    def generate_initial_questions(self, transcription_id: str) -> List[str]:
        """Generate initial questions for a transcription"""
        try:
            # Create knowledge base
            knowledge_base = AgentFactory.create_knowledge_base(
                self.tenant_code)

            # Search for the transcription by ID
            search_results = knowledge_base.search_by_id(transcription_id)

            # Extract transcription text
            transcription_text = ""
            if search_results and len(search_results) > 0:
                transcription_text = search_results[0].document

            # Create agent
            agent = AgentFactory.create_initial_questions_agent(
                self.tenant_code, self.model)

            # Generate questions
            prompt = f"Based on this call transcription:\n\n{transcription_text}\n\nGenerate 3-5 insightful questions:"
            response = agent.get_response(prompt)

            # Parse questions from response
            questions = self._parse_questions(response)

            if not search_results:
                logger.warning(
                    f"Transcription {transcription_id} not found in Qdrant")
                return []

            if not transcription_text:
                logger.warning(
                    f"No text found for transcription {transcription_id}")
                return []

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
            # Create knowledge base
            knowledge_base = AgentFactory.create_knowledge_base(
                self.tenant_code)

            # Search for the transcription by ID
            search_results = knowledge_base.search_by_id(transcription_id)

            # Extract transcription text
            transcription_text = ""
            if search_results and len(search_results) > 0:
                transcription_text = search_results[0].document

            # Get question text
            question_text = self._get_question_from_db(question_id)

            # Create agent
            agent = AgentFactory.create_response_analyzer_agent(
                self.tenant_code, self.model)

            # Generate analysis
            prompt = f"""
            Call Transcription:
            {transcription_text}
            
            Question:
            {question_text}
            
            Response:
            {response_text}
            
            Please analyze this response:
            """

            analysis = agent.get_response(prompt)

            if not search_results:
                logger.warning(
                    f"Transcription {transcription_id} not found in Qdrant")
                return "Error: Transcription not found"

            # Save analysis to database
            analysis_id = self._save_analysis_to_db(
                transcription_id, question_id, analysis)

            return analysis
        except Exception as e:
            logger.error(f"Error analyzing response: {e}")
            logger.exception("Detailed error:")
            return f"Error analyzing response: {str(e)}"

    def generate_followup_questions(self, transcription_id: str, previous_questions: List[str], previous_responses: List[str]) -> List[str]:
        """Generate follow-up questions based on previous responses"""
        try:
            # Create knowledge base
            knowledge_base = AgentFactory.create_knowledge_base(
                self.tenant_code)

            # Search for the transcription by ID
            search_results = knowledge_base.search_by_id(transcription_id)

            # Extract transcription text
            transcription_text = ""
            if search_results and len(search_results) > 0:
                transcription_text = search_results[0].document

            # Create agent
            agent = AgentFactory.create_followup_agent(
                self.tenant_code, self.model)

            # Prepare conversation history
            conversation = ""
            for i in range(len(previous_questions)):
                conversation += f"Q: {previous_questions[i]}\nA: {previous_responses[i]}\n\n"

            # Generate follow-up questions
            prompt = f"""
            Call Transcription:
            {transcription_text}
            
            Previous Conversation:
            {conversation}
            
            Generate 2-3 follow-up questions:
            """

            response = agent.get_response(prompt)

            # Parse questions from response
            questions = self._parse_questions(response)

            # Save questions to database
            self._save_questions_to_db(transcription_id, questions, "followup")

            return questions
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            logger.exception("Detailed error:")
            return []

    def _parse_questions(self, response: str) -> List[str]:
        """Parse questions from agent response"""
        lines = response.strip().split('\n')
        questions = []

        for line in lines:
            line = line.strip()
            # Check if line starts with a number or has a question mark
            if (line and (line[0].isdigit() and '. ' in line[:5]) or '?' in line):
                # Remove the number prefix if it exists
                if line[0].isdigit() and '. ' in line[:5]:
                    question = line[line.find('. ')+2:]
                else:
                    question = line
                questions.append(question)

        return questions

    def _get_question_from_db(self, question_id: str) -> str:
        """Get question text from database"""
        if not self.session:
            return "Unknown question"

        try:
            result = self.session.execute(
                text("""
                    SELECT question FROM questions 
                    WHERE id = :question_id
                """),
                {"question_id": question_id}
            ).fetchone()

            if result:
                return result[0]
            return "Unknown question"
        except Exception as e:
            logger.error(f"Error getting question from database: {e}")
            return "Unknown question"

    def _save_questions_to_db(self, transcription_id: str, questions: List[str], question_type: str) -> List[str]:
        """Save questions to database"""
        logger.info(
            f"Saving {len(questions)} {question_type} questions for transcription {transcription_id}")

        question_ids = []
        if self.session:
            try:
                for question in questions:
                    question_id = str(uuid.uuid4())
                    self.session.execute(
                        text("""
                            INSERT INTO questions (id, transcription_id, question, type, tenant_code)
                            VALUES (:id, :transcription_id, :question, :type, :tenant_code)
                        """),
                        {
                            "id": question_id,
                            "transcription_id": transcription_id,
                            "question": question,
                            "type": question_type,
                            "tenant_code": self.tenant_code
                        }
                    )
                    question_ids.append(question_id)
                self.session.commit()
            except Exception as e:
                logger.error(f"Error saving questions to database: {e}")
                logger.exception("Detailed error:")
                self.session.rollback()

        return question_ids

    def _save_analysis_to_db(self, transcription_id: str, question_id: str, analysis: str) -> str:
        """Save analysis to database"""
        logger.info(
            f"Saving analysis for question {question_id} of transcription {transcription_id}")

        analysis_id = str(uuid.uuid4())
        if self.session:
            try:
                self.session.execute(
                    text("""
                        INSERT INTO analyses (id, transcription_id, question_id, analysis, tenant_code)
                        VALUES (:id, :transcription_id, :question_id, :analysis, :tenant_code)
                    """),
                    {
                        "id": analysis_id,
                        "transcription_id": transcription_id,
                        "question_id": question_id,
                        "analysis": analysis,
                        "tenant_code": self.tenant_code
                    }
                )
                self.session.commit()
            except Exception as e:
                logger.error(f"Error saving analysis to database: {e}")
                logger.exception("Detailed error:")
                self.session.rollback()

        return analysis_id

    def search_transcriptions(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for transcriptions related to the query"""
        try:
            # Create knowledge base
            knowledge_base = AgentFactory.create_knowledge_base(
                self.tenant_code)

            # Search for relevant transcriptions
            search_results = knowledge_base.vector_db.search(
                query=query,
                limit=limit
            )

            # Format the results
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "id": result.id,
                    "text": result.document,
                    "metadata": result.metadata,
                    "score": result.score
                })

            return formatted_results
        except Exception as e:
            logger.error(f"Error searching transcriptions: {e}")
            logger.exception("Detailed error:")
            return []

    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate a response based on search results"""
        try:
            # Create an agent
            agent = AgentFactory.create_initial_questions_agent(
                self.tenant_code, self.model)

            # Prepare context from search results
            context = "Based on the following transcriptions:\n\n"
            for i, result in enumerate(search_results):
                context += f"Transcription {i+1}:\n{result['text']}\n\n"

            # Generate response
            prompt = f"{context}\nQuestion: {query}\n\nPlease provide a comprehensive answer:"
            response = agent.get_response(prompt)

            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.exception("Detailed error:")
            return f"Error generating response: {str(e)}"
