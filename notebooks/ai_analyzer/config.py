# ai_analyzer/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

LIMIT = 10000
LAST_ID = 0
DATA_DIR = '../app/backend/ai-analyzer/data'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_env_once():
    """Load environment variables once and return config dict"""
    try:
        # Check multiple possible locations for .env file
        possible_paths = [
            #Path(__file__).resolve().parent.parent / '.env',  # /usr/src/app/ai-analyzer/.env
            #Path(__file__).resolve().parent / '.env',         # /usr/src/app/ai-analyzer/ai_analyzer/.env
            Path('./app/.env'),                       # /usr/src/app/.env
            Path('./app/ai-analyzer/.env')            # Direct path
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
            logger.warning("No .env file found, falling back to environment variables")
        
        # Create configuration dictionary
        config = {
            'POSTGRES_USER': os.getenv('POSTGRES_USER'),
            'POSTGRES_PASSWORD': os.getenv('POSTGRES_PASSWORD'),
            'DB_HOST': os.getenv('DB_HOST'),
            'DB_PORT': os.getenv('DB_PORT'),
            'POSTGRES_DB': os.getenv('POSTGRES_DB'),
            'AI_ANALYZER_OPENAI_API_KEY': os.getenv('AI_ANALYZER_OPENAI_API_KEY'),
            'URL': os.getenv('URL'),
            'API_KEY': os.getenv('API_KEY'),
            'DATA_DIR': os.getenv('DATA_DIR', 'data'),
            'LIMIT': int(os.getenv('LIMIT', 1000)),
            'ADMIN_USER': os.getenv('ADMIN_USER'),
            'ADMIN_PASSWORD': os.getenv('ADMIN_PASSWORD'),        
            'READ_USER': os.getenv('READ_USER'),
            'READ_USER_PASSWORD': os.getenv('READ_USER_PASSWORD')
        }
        
        # Validate required configuration
        required_keys = ['POSTGRES_USER', 'POSTGRES_PASSWORD', 'DB_HOST', 'DB_PORT', 'POSTGRES_DB']
        missing_keys = [key for key in required_keys if not config[key]]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration: {', '.join(missing_keys)}")
            
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

# Load configuration once when module is imported
try:
    config = load_env_once()
    
    # Create frequently used configurations
    # f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}'
    DATABASE_URL = f"postgresql://{config['POSTGRES_USER']}:{config['POSTGRES_PASSWORD']}@{config['DB_HOST']}:{config['DB_PORT']}/{config['POSTGRES_DB']}"
    DATA_DIR = config['DATA_DIR']
    
except Exception as e:
    logger.error(f"Failed to initialize configuration: {e}")
    raise

