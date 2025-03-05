#!/usr/bin/env python3
import os
import logging
from typing import List, Dict, Any, Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.langchain import LangChainKnowledgeBase

# Import the correct version of langchain components
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

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


class AgentFactory:
    """Factory for creating Agno agents"""

    @staticmethod
    def create_agent_team(tenant_code: str) -> Agent:
        """Create a team of agents for call analysis"""
        try:
            # Create knowledge base first
            knowledge_base = AgentFactory.create_knowledge_base(tenant_code)

            # Create individual agents with the same knowledge base
            initial_questions_agent = Agent(
                name="Initial Questions Generator",
                role="Generate initial questions for call analysis",
                model=OpenAIChat(api_key=OPENAI_API_KEY),
                knowledge=knowledge_base,  # Pass the knowledge base
                instructions=[
                    "You are an expert call analyst who generates initial questions for call analysis.",
                    "Your task is to generate questions that help users explore call data.",
                    "Focus on questions that reveal trends, patterns, and insights.",
                    "Make questions specific and actionable."
                ],
                markdown=True,
                search_knowledge=True
            )

            response_analyzer_agent = Agent(
                name="Response Analyzer",
                role="Analyze responses to questions about call transcriptions",
                model=OpenAIChat(api_key=OPENAI_API_KEY),
                knowledge=knowledge_base,  # Pass the knowledge base
                instructions=[
                    "You are an expert call analyst who reviews responses to questions about call transcriptions.",
                    "Your task is to analyze the response and extract key insights.",
                    "Identify patterns, sentiments, and important information from the response.",
                    "Provide a brief summary of the response analysis.",
                    "Highlight any actionable insights or recommendations."
                ],
                markdown=True,
                search_knowledge=True
            )

            followup_agent = Agent(
                name="Follow-up Questions Generator",
                role="Generate follow-up questions based on conversation",
                model=OpenAIChat(api_key=OPENAI_API_KEY),
                knowledge=knowledge_base,  # Pass the knowledge base
                instructions=[
                    "You are an expert call analyst who generates follow-up questions.",
                    "Your task is to generate questions that help users explore call data further.",
                    "Focus on questions that build on previous responses.",
                    "Make questions specific and actionable."
                ],
                markdown=True,
                search_knowledge=True
            )

            # Create team agent with the same knowledge base
            team_agent = Agent(
                name="Call Analysis Team",
                team=[initial_questions_agent,
                      response_analyzer_agent, followup_agent],
                model=OpenAIChat(api_key=OPENAI_API_KEY),
                knowledge=knowledge_base,  # Pass the knowledge base
                instructions=[
                    "You are a team of expert call analysts who review call transcriptions.",
                    "Your task is to analyze call transcriptions and provide insights.",
                    "Use the specialized agents in your team to handle different aspects of the analysis."
                ],
                markdown=True,
                search_knowledge=True
            )

            return team_agent

        except Exception as e:
            logger.error(f"Error creating agent team: {e}")
            logger.exception("Detailed error:")

            # Return a minimal agent as fallback
            return Agent(
                name="Fallback Agent",
                model=OpenAIChat(api_key=OPENAI_API_KEY),
                instructions=[
                    "You are a helpful assistant for call analysis."],
                markdown=True
            )

    @staticmethod
    def create_followup_agent(tenant_code: str) -> Optional[Agent]:
        """Create a followup agent for a tenant"""
        try:
            # Create agent
            agent = Agent(
                model=OpenAIChat(api_key=OPENAI_API_KEY),
                description="You are an AI assistant that generates follow-up questions based on conversation history.",
                markdown=True
            )

            return agent
        except Exception as e:
            logger.error(f"Error creating followup agent: {e}")
            logger.exception("Detailed error:")
            return None

    @staticmethod
    def create_knowledge_base(tenant_code: str) -> LangChainKnowledgeBase:
        """Create a knowledge base that uses the existing tenant collection in Qdrant"""
        try:
            # Get Qdrant client
            qdrant_client = get_qdrant_client()
            collection_name = f"tenant_{tenant_code}"

            # Check if collection exists
            collections = qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name not in collection_names:
                logger.warning(
                    f"Collection {collection_name} not found in Qdrant")
                # Return empty knowledge base
                return LangChainKnowledgeBase(retriever=None)

            # Get collection info to find the vector name and dimensions
            collection_info = qdrant_client.get_collection(collection_name)
            vector_names = list(collection_info.config.params.vectors.keys())

            if not vector_names:
                logger.error(
                    f"No vector names found in collection {collection_name}")
                return LangChainKnowledgeBase(retriever=None)

            vector_name = vector_names[0]  # Use the first vector name
            logger.info(
                f"Using vector name: {vector_name} for collection {collection_name}")

            # Create embeddings model that matches what's in the collection
            embeddings = FastEmbedEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

            # Connect to the existing vector store
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings,
                vector_name=vector_name
            )

            # Create retriever with more results for better context
            retriever = vector_store.as_retriever(
                # Return top 100 results for comprehensive coverage
                search_kwargs={"k": 100}
            )

            # Create and return knowledge base with the retriever
            knowledge_base = LangChainKnowledgeBase(retriever=retriever)
            logger.info(
                f"Successfully created knowledge base for collection {collection_name}")
            return knowledge_base

        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            logger.exception("Detailed error:")

            # Return a minimal knowledge base as fallback
            return LangChainKnowledgeBase(retriever=None)

    @staticmethod
    def create_initial_questions_agent(tenant_code: str) -> Agent:
        """Create an agent that generates initial questions based on transcriptions"""
        knowledge_base = AgentFactory.create_knowledge_base(tenant_code)

        # Fix: Remove 'model' parameter from OpenAIChat constructor
        return Agent(
            name="Initial Questions Agent",
            role="Generate initial questions based on call transcriptions",
            model=OpenAIChat(api_key=OPENAI_API_KEY),  # Remove model parameter
            knowledge=knowledge_base,
            instructions=[
                "You are an expert call analyst who reviews call transcriptions.",
                "Your task is to generate 3-5 insightful questions based on the call transcription.",
                "Focus on questions that would help understand customer needs, pain points, and opportunities.",
                "Questions should be specific to the content of the call, not generic.",
                "Format your response as a numbered list of questions.",
                "Each question should be clear, concise, and directly related to the call content."
            ],
            markdown=True,
            search_knowledge=True
        )

    @staticmethod
    def create_response_analyzer_agent(tenant_code: str) -> Agent:
        """Create an agent that analyzes responses to questions"""
        knowledge_base = AgentFactory.create_knowledge_base(tenant_code)

        # Fix: Remove 'model' parameter from OpenAIChat constructor
        return Agent(
            name="Response Analyzer",
            role="Analyze responses to questions about call transcriptions",
            model=OpenAIChat(api_key=OPENAI_API_KEY),  # Remove model parameter
            knowledge=knowledge_base,
            instructions=[
                "You are an expert call analyst who reviews responses to questions about call transcriptions.",
                "Your task is to analyze the response and extract key insights.",
                "Identify patterns, sentiments, and important information from the response.",
                "Provide a brief summary of the response analysis.",
                "Highlight any actionable insights or recommendations."
            ],
            markdown=True,
            search_knowledge=True
        )
