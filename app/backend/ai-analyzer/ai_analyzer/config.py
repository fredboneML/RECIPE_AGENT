# ai_analyzer/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# LIMIT = 10000
LIMIT = 500
LAST_ID = 0
DATA_DIR = '/usr/src/app/ai-analyzer/data'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_env_once():
    """Load environment variables once and return config dict"""
    try:
        # Check multiple possible locations for .env file
        possible_paths = [
            Path(__file__).resolve().parent.parent /
            '.env',  # /usr/src/app/ai-analyzer/.env
            # /usr/src/app/ai-analyzer/ai_analyzer/.env
            Path(__file__).resolve().parent / '.env',
            # /usr/src/app/.env
            Path('/usr/src/app/.env'),
            Path('/usr/src/app/ai-analyzer/.env')            # Direct path
        ]

        env_file = None
        for path in possible_paths:
            if path.exists():
                env_file = path
                logger.info(f"Found .env file at: {path}")
                break

            # Load environment variables
            load_dotenv(env_file)
        else:
            logger.warning(
                "No .env file found, falling back to environment variables")

        # Create configuration dictionary
        config = {
            'POSTGRES_USER': os.getenv('POSTGRES_USER'),
            'POSTGRES_PASSWORD': os.getenv('POSTGRES_PASSWORD'),
            'DB_HOST': os.getenv('DB_HOST', 'database'),
            'DB_PORT': os.getenv('DB_PORT', '5432'),
            'POSTGRES_DB': os.getenv('POSTGRES_DB'),
            'AI_ANALYZER_OPENAI_API_KEY': os.getenv('AI_ANALYZER_OPENAI_API_KEY'),
            'URL': os.getenv('URL'),
            'API_KEY': os.getenv('API_KEY'),
            'DATA_DIR': os.getenv('DATA_DIR', 'data'),
            'LIMIT': int(os.getenv('LIMIT', 50)),
            'ADMIN_USER': os.getenv('ADMIN_USER'),
            'ADMIN_PASSWORD': os.getenv('ADMIN_PASSWORD'),
            'READ_USER': os.getenv('READ_USER'),
            'READ_USER_PASSWORD': os.getenv('READ_USER_PASSWORD'),
            'TENANT_CODE': os.getenv('TENANT_CODE'),
            'LAST_ID': os.getenv('LAST_ID'),
            'OPENAI_MODEL': os.getenv('OPENAI_MODEL'),
            # Added default value
            'MODEL_NAME': os.getenv('MODEL_NAME', 'gpt-4o-mini-2024-07-18'),
            'JWT_SECRET_KEY': os.getenv('JWT_SECRET_KEY'),
            'JWT_ALGORITHM': os.getenv('JWT_ALGORITHM', "HS256"),
            'ACCESS_TOKEN_EXPIRE_MINUTES': int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 60 * 24)),
            # Azure AD SSO Configuration
            'AZURE_AD_TENANT_ID': os.getenv('AZURE_AD_TENANT_ID'),
            'AZURE_AD_CLIENT_ID': os.getenv('AZURE_AD_CLIENT_ID'),
            'SSO_ENABLED': os.getenv('SSO_ENABLED', 'true').lower() == 'true',
            'LOCAL_AUTH_ENABLED': os.getenv('LOCAL_AUTH_ENABLED', 'true').lower() == 'true',
        }

        # Validate required configuration
        required_keys = ['POSTGRES_USER', 'POSTGRES_PASSWORD',
                         'DB_HOST', 'DB_PORT', 'POSTGRES_DB']
        missing_keys = [key for key in required_keys if not config[key]]

        if missing_keys:
            raise ValueError(f"Missing required configuration: {
                             ', '.join(missing_keys)}")

        return config

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


# Load configuration once when module is imported
try:
    config = load_env_once()
    logger.info(f"Using database host: {
                config['DB_HOST']} and port: {config['DB_PORT']}")

    # Create frequently used configurations
    # f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}'
    DATABASE_URL = f"postgresql://{config['POSTGRES_USER']}:{config['POSTGRES_PASSWORD']}@{
        config['DB_HOST']}:{config['DB_PORT']}/{config['POSTGRES_DB']}"
    DATA_DIR = config['DATA_DIR']
    # Export MODEL_NAME for other modules to use
    MODEL_NAME = config['MODEL_NAME']

    # Export JWT configuration constants
    JWT_SECRET_KEY = config['JWT_SECRET_KEY']
    JWT_ALGORITHM = config['JWT_ALGORITHM']
    ACCESS_TOKEN_EXPIRE_MINUTES = config['ACCESS_TOKEN_EXPIRE_MINUTES']

    # Export Azure AD SSO configuration
    AZURE_AD_TENANT_ID = config['AZURE_AD_TENANT_ID']
    AZURE_AD_CLIENT_ID = config['AZURE_AD_CLIENT_ID']
    SSO_ENABLED = config['SSO_ENABLED']
    LOCAL_AUTH_ENABLED = config['LOCAL_AUTH_ENABLED']

except Exception as e:
    logger.error(f"Failed to initialize configuration: {e}")
    raise
