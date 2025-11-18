import os
import logging
from typing import Dict, Any, Optional
from openai import OpenAI

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
        "provider": os.getenv("MODEL_PROVIDER"),
        "model_name": os.getenv("MODEL_NAME"),
        "api_key": os.getenv("API_KEY", ""),
        "groq_model_name": os.getenv("GROQ_MODEL_NAME"),
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "groq_use_openai_compatibility": os.getenv("GROQ_USE_OPENAI_COMPATIBILITY", "false").lower() == "true",
    }

    # If using Groq with OpenAI compatibility, ensure base_url is set
    if config['provider'] == 'groq' and config['groq_use_openai_compatibility']:
        config['base_url'] = "https://api.groq.com/openai/v1"
        logger.info("Setting base_url for Groq OpenAI compatibility mode")

    # Log configuration (excluding sensitive data)
    logger.info("Model configuration loaded:")
    logger.info(f"Provider: {config['provider']}")
    logger.info(f"Model name: {config['model_name']}")
    if config['provider'] == 'groq':
        logger.info(f"Groq model: {config['groq_model_name']}")
        logger.info(
            f"Groq OpenAI compatibility: {config['groq_use_openai_compatibility']}")
        if config['groq_use_openai_compatibility']:
            logger.info(f"Using Groq base_url: {config['base_url']}")

    return config


def query_llm(prompt: str, provider: str = "openai", model: str = "gpt-4o-mini", use_azure: Optional[bool] = None) -> Optional[str]:
    """
    Simple LLM query function using OpenAI or Azure OpenAI client

    Args:
        prompt: The text prompt to send
        provider: The API provider (currently only supports "openai")
        model: The model to use (default: gpt-4o-mini - cheapest OpenAI model)
        use_azure: Whether to use Azure OpenAI instead of OpenAI (default: loaded from USE_AZURE env var, or False)

    Returns:
        The LLM's response or None if there was an error
    """
    try:
        # Load use_azure from environment variable if not explicitly provided
        if use_azure is None:
            use_azure = os.getenv('USE_AZURE', 'false').lower() == 'true'
        if provider != "openai":
            logger.warning(
                f"Provider {provider} not supported, falling back to OpenAI")
            provider = "openai"

        # Create OpenAI or Azure OpenAI client
        if use_azure:
            # Get Azure OpenAI credentials from environment
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            azure_endpoint = os.getenv(
                'AZURE_OPENAI_ENDPOINT', 'https://msopenai.openai.azure.com')

            if not api_key:
                logger.error(
                    "AZURE_OPENAI_API_KEY not found in environment variables")
                return None

            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-08-01-preview",
                azure_endpoint=azure_endpoint
            )

            # Use Azure deployment name if model is default
            if model == "gpt-4o-mini":
                model = os.getenv(
                    'AZURE_OPENAI_MODEL_DEPLOYMENT', 'gpt-4o-mini')
        else:
            # Get OpenAI API key from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error(
                    "OPENAI_API_KEY not found in environment variables")
                return None

            client = OpenAI(api_key=api_key)

        # Log model usage
        ModelLogger.log_model_usage(
            agent_name="recipe_search_agent",
            model_provider="azure_openai" if use_azure else provider,
            model_name=model,
            params={"temperature": 0.7, "use_azure": use_azure}
        )

        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return None
