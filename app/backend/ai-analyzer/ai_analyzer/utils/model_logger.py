import os
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLogger:
    """Utility class for logging model usage and configuration"""

    @staticmethod
    def log_model_usage(
        agent_name: str,
        model_provider: str,
        model_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log model usage information"""
        logger.info(f"Model usage by {agent_name}:")
        logger.info(f"Provider: {model_provider}")
        logger.info(f"Model: {model_name}")
        if params:
            logger.info(f"Parameters: {params}")


def get_model_config_from_env() -> Dict[str, Any]:
    """Get model configuration from environment variables"""
    config = {
        "provider": os.getenv("MODEL_PROVIDER", "openai"),
        "model_name": os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        "api_key": os.getenv("API_KEY", ""),
        "groq_model_name": os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768"),
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "groq_use_openai_compatibility": os.getenv("GROQ_USE_OPENAI_COMPATIBILITY", "false").lower() == "true"
    }

    # Log configuration (excluding sensitive data)
    logger.info("Model configuration loaded:")
    logger.info(f"Provider: {config['provider']}")
    logger.info(f"Model name: {config['model_name']}")
    if config['provider'] == 'groq':
        logger.info(f"Groq model: {config['groq_model_name']}")
        logger.info(
            f"Groq OpenAI compatibility: {config['groq_use_openai_compatibility']}")

    return config
