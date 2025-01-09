# backend/ai-analyzer/ai_analyzer/agents/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DatabaseContext:
    """Context information about the database structure and restrictions"""
    allowed_tables: List[str]
    restricted_tables: List[str]
    table_schemas: Dict[str, Dict[str, str]]
    base_context: Optional[str] = None


@dataclass
class AgentResponse:
    """Structured response from any agent"""
    success: bool
    content: Any
    error_message: Optional[str] = None
    suggested_followup: Optional[List[str]] = None
    reformulated_question: Optional[str] = None


class ModelProvider(ABC):
    """Abstract base class for different model providers"""

    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        pass


class OpenAIProvider(ModelProvider):
    def __init__(self, model_name: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    async def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content


class OllamaProvider(ModelProvider):
    def __init__(self, model_name: str):
        from langchain_community.llms import Ollama
        self.model = Ollama(model=model_name)

    async def generate_response(self, prompt: str) -> str:
        return await self.model.agenerate([prompt])


class HuggingFaceProvider(ModelProvider):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from transformers import pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            token=api_key
        )

    async def generate_response(self, prompt: str) -> str:
        response = self.pipeline(prompt, max_length=500)
        return response[0]['generated_text']


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self,
                 model_provider: str,
                 model_name: str,
                 api_key: Optional[str] = None):
        self.model_provider = self._initialize_provider(
            model_provider,
            model_name,
            api_key
        )

    def _initialize_provider(self,
                             provider: str,
                             model_name: str,
                             api_key: Optional[str]) -> ModelProvider:
        if provider == "openai":
            return OpenAIProvider(model_name, api_key)
        elif provider == "ollama":
            return OllamaProvider(model_name)
        elif provider == "huggingface":
            return HuggingFaceProvider(model_name, api_key)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    @abstractmethod
    async def process(self, *args, **kwargs) -> AgentResponse:
        pass
