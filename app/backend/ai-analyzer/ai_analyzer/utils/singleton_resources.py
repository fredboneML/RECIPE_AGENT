import logging
import threading
from typing import Optional, Dict, Any
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe singleton management


class ResourceManager:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    qdrant_client: Optional[QdrantClient] = None
    embedding_models: Dict[str, TextEmbedding] = {}

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        # Only initialize once
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialized = True
                    logger.info("Initializing ResourceManager singleton")

                    # Create persistent cache directory
                    cache_dir = "/tmp/fastembed_cache"
                    os.makedirs(cache_dir, exist_ok=True)

            # IMPORTANT: Pre-warm model OUTSIDE the lock to avoid deadlock
            self._pre_warm_model()

    def _pre_warm_model(self):
        """Pre-warm the embedding model without holding the lock"""
        try:
            logger.info("Pre-warming embedding model cache...")
            model_name = "paraphrase-multilingual-MiniLM-L12-v2"
            cache_dir = "/tmp/fastembed_cache"

            # Only initialize if not already present
            if model_name not in self.embedding_models:
                logger.info(f"Initializing model: {model_name}")
                model = TextEmbedding(
                    f"sentence-transformers/{model_name}",
                    cache_dir=cache_dir
                )

                # Test the model with a simple sentence
                test_input = ["This is a test sentence."]
                _ = list(model.embed(test_input))

                # Store the model in our dictionary
                with self._lock:
                    if model_name not in self.embedding_models:
                        self.embedding_models[model_name] = model

                logger.info("Embedding model cache pre-warmed successfully")
        except Exception as e:
            logger.warning(f"Failed to pre-warm embedding model: {e}")

    def get_qdrant_client(self) -> QdrantClient:
        with self._lock:
            if self.qdrant_client is None:
                logger.info("Starting Qdrant client initialization")
                try:
                    logger.info("Creating Qdrant client instance")
                    self.qdrant_client = QdrantClient(
                        host="qdrant",
                        port=6333,
                        timeout=10.0,
                        prefer_grpc=False  # Force HTTP for better stability
                    )
                    logger.info(
                        "Qdrant client instance created, testing connection")

                    # Test connection with timeout
                    try:
                        collections = self.qdrant_client.get_collections()
                        logger.info(
                            f"Qdrant connected successfully. Found collections: {[c.name for c in collections.collections]}")
                    except Exception as e:
                        logger.error(
                            f"Failed to get collections from Qdrant: {e}")
                        logger.error(
                            "Connection test failed, raising exception")
                        raise
                except Exception as e:
                    logger.error(f"Failed to initialize Qdrant client: {e}")
                    logger.error("Stack trace:", exc_info=True)
                    raise
            return self.qdrant_client

    def get_embedding_model(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> TextEmbedding:
        with self._lock:
            if model_name not in self.embedding_models:
                logger.info(
                    f"Starting embedding model initialization: {model_name}")
                try:
                    logger.info("Creating embedding model instance")
                    # Use a persistent cache directory
                    cache_dir = "/tmp/fastembed_cache"
                    os.makedirs(cache_dir, exist_ok=True)

                    self.embedding_models[model_name] = TextEmbedding(
                        f"sentence-transformers/{model_name}",
                        cache_dir=cache_dir
                    )

                    # Test the model to ensure it's loaded
                    test_input = ["This is a test sentence."]
                    _ = list(
                        self.embedding_models[model_name].embed(test_input))

                    logger.info(
                        f"Model {model_name} instance created and tested successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize embedding model: {e}")
                    logger.error("Stack trace:", exc_info=True)
                    raise
            return self.embedding_models[model_name]


# Create a single global instance
_RESOURCE_MANAGER = ResourceManager()

# Public API functions


def get_qdrant_client() -> QdrantClient:
    """Get the singleton Qdrant client instance"""
    return _RESOURCE_MANAGER.get_qdrant_client()


def get_embedding_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> TextEmbedding:
    """Get the singleton embedding model instance"""
    return _RESOURCE_MANAGER.get_embedding_model(model_name)
