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

        if provider == "openai":
            model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                temperature=kwargs.get('temperature', 0),
                **kwargs
            )
            logger.info("Created OpenAI model")
            return model
        elif provider == "groq":
            # Support both native Groq and OpenAI compatibility
            if kwargs.get('use_openai_compatibility', False):
                logger.info("Creating Groq model with OpenAI compatibility")
                model = ChatOpenAI(
                    model_name=model_name,
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=kwargs.get('temperature', 0),
                    **kwargs
                )
            else:
                logger.info("Creating native Groq model")
                model = ChatGroq(
                    model_name=model_name,
                    groq_api_key=api_key,
                    temperature=kwargs.get('temperature', 0),
                    **kwargs
                )
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
