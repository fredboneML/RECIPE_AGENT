# File: backend/ai-analyzer/ai_analyzer/agents/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
import logging
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.language_models.llms import LLM
from langchain.chains import LLMChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_groq import ChatGroq
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseContext:
    """Context information about the database structure and restrictions"""
    allowed_tables: List[str]
    restricted_tables: List[str]
    table_schemas: Dict[str, Dict[str, str]]
 #   base_context: Optional[str] = None


@dataclass
class AgentResponse:
    """Structured response from any agent"""
    success: bool
    content: Any
    error_message: Optional[str] = None
    suggested_followup: Optional[List[str]] = None
    reformulated_question: Optional[str] = None


class LangChainModelProvider:
    """Factory class for creating LangChain model instances"""

    @staticmethod
    def create_model(
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> Union[ChatOpenAI, Ollama, LLM]:
        # Log model creation
        logger.info(
            f"Creating model with provider: {provider}, model: {model_name}")

        # Check environment variables for Groq compatibility
        use_openai_compatibility = os.getenv(
            "GROQ_USE_OPENAI_COMPATIBILITY", "false").lower() == "true"
        if provider == "groq" and not kwargs.get('use_openai_compatibility', False):
            kwargs['use_openai_compatibility'] = use_openai_compatibility
            logger.info(
                f"Setting use_openai_compatibility from environment: {use_openai_compatibility}")

        # Handle Groq provider with direct OpenAI interface
        if provider == "groq":
            # Force use of Groq API
            groq_model_name = os.getenv("GROQ_MODEL_NAME", model_name)
            groq_api_key = os.getenv("GROQ_API_KEY", api_key)

            # Create direct OpenAI client for Groq
            openai_client = OpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            # Use ChatOpenAI with client parameter
            temp = kwargs.pop('temperature', 0)
            model = ChatOpenAI(
                model_name=groq_model_name,
                client=openai_client,
                temperature=temp,
                **kwargs
            )
            logger.info(
                f"Created OpenAI-compatible model with Groq backend using model: {groq_model_name}")
            return model
        elif provider == "openai":
            # Remove temperature from kwargs to avoid duplicate parameter
            temp = kwargs.pop('temperature', 0)

            model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                temperature=temp,
                **kwargs
            )
            logger.info("Created OpenAI model")
            return model
        elif provider == "ollama":
            logger.info("Creating Ollama model")
            return Ollama(
                model=model_name,
                **kwargs
            )
        elif provider == "huggingface":
            logger.info("Creating HuggingFace model")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **kwargs
            )
            return HuggingFacePipeline(pipeline=pipe)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")


class BaseAgent(ABC):
    """Abstract base class for all agents using LangChain"""

    def __init__(self,
                 model_provider: str,
                 model_name: str,
                 api_key: Optional[str] = None,
                 **kwargs):
        # Log agent initialization
        logger.info(
            f"Initializing {self.__class__.__name__} with model provider: {model_provider}, model: {model_name}")

        # Add Groq OpenAI compatibility setting from environment if not in kwargs
        if model_provider == "groq" and "use_openai_compatibility" not in kwargs:
            use_openai_compatibility = os.getenv(
                "GROQ_USE_OPENAI_COMPATIBILITY", "false").lower() == "true"
            kwargs["use_openai_compatibility"] = use_openai_compatibility
            logger.info(
                f"Using Groq OpenAI compatibility from environment: {use_openai_compatibility}")

        # Also get the Groq API key from environment if not provided
        if model_provider == "groq" and api_key is None:
            api_key = os.getenv("GROQ_API_KEY", "")
            logger.info("Using Groq API key from environment")

        # Special handling for Groq to ensure direct creation with OpenAI compatibility
        if model_provider == "groq":
            # We'll set a flag for use_openai_compatibility that will be used for logging
            self.use_openai_compatibility = kwargs.get(
                'use_openai_compatibility', False)

        self.llm = LangChainModelProvider.create_model(
            model_provider,
            model_name,
            api_key,
            **kwargs
        )

        # Log model type and configuration
        logger.info(f"Created {type(self.llm).__name__} model instance")
        if hasattr(self.llm, 'model_name'):
            logger.info(f"Model name: {self.llm.model_name}")
        if hasattr(self.llm, 'temperature'):
            logger.info(f"Model temperature: {self.llm.temperature}")
        if model_provider == "groq":
            logger.info(
                f"Using Groq model with OpenAI compatibility: {getattr(self.llm, 'use_openai_compatibility', False)}")

        # Store model info for later use
        self.model_provider = model_provider
        self.model_name = model_name

    def _create_chain(self, prompt_template: str) -> LLMChain:
        """Create a LangChain chain with the given prompt template"""
        prompt = PromptTemplate(
            input_variables=["input"],
            template=prompt_template
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    @abstractmethod
    async def process(self, *args, **kwargs) -> AgentResponse:
        pass
