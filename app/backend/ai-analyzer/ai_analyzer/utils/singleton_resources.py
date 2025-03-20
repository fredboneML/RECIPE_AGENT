import logging
import threading
from typing import Optional, Dict, Any
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe singleton management


class ResourceManager:
    _instance = None
    _lock = threading.Lock()

    qdrant_client: Optional[QdrantClient] = None
    embedding_models: Dict[str, TextEmbedding] = {}

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
            return cls._instance

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
                    self.embedding_models[model_name] = TextEmbedding(
                        f"sentence-transformers/{model_name}",
                        cache_dir="/tmp/fastembed_cache"  # Use persistent cache
                    )
                    logger.info(
                        f"Model {model_name} instance created successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize embedding model: {e}")
                    logger.error("Stack trace:", exc_info=True)
                    raise
            return self.embedding_models[model_name]


# Singleton instance
_resource_manager = ResourceManager()

# Public API functions


def get_qdrant_client() -> QdrantClient:
    return _resource_manager.get_qdrant_client()


def get_embedding_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> TextEmbedding:
    return _resource_manager.get_embedding_model(model_name)
