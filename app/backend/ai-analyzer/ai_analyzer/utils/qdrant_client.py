import logging
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global singleton instances with thread safety
_qdrant_client: Optional[QdrantClient] = None
_embedding_model: Optional[TextEmbedding] = None
_lock = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client (thread-safe singleton pattern)"""
    global _qdrant_client
    with _lock:
        try:
            if _qdrant_client is None:
                logger.info("Creating new Qdrant client connection")
                _qdrant_client = QdrantClient(
                    host="qdrant",
                    port=6333,
                    timeout=10.0  # Add explicit timeout
                )
                # Test the connection
                try:
                    collections = _qdrant_client.get_collections()
                    logger.info(
                        f"Successfully connected to Qdrant. Found collections: {[c.name for c in collections.collections]}")
                except Exception as e:
                    logger.error(
                        f"Connected to Qdrant but failed to get collections: {e}")
            return _qdrant_client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            logger.exception("Detailed connection error:")
            raise


def get_embedding_model() -> TextEmbedding:
    """Get embedding model (thread-safe singleton pattern)"""
    global _embedding_model
    with _lock:
        try:
            if _embedding_model is None:
                logger.info("Initializing embedding model")
                _embedding_model = TextEmbedding(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    cache_dir="/app/embedding_model"  # Use a persistent cache directory
                )
                logger.info("Embedding model initialized successfully")
            return _embedding_model
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            logger.exception("Detailed error:")
            raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_qdrant_with_retry(client: QdrantClient, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """Perform Qdrant search with retry logic"""
    try:
        return client.search(
            collection_name=collection_name,
            query_vector=(
                "fast-paraphrase-multilingual-minilm-l12-v2", query_vector),
            limit=limit,
            timeout=5
        )
    except Exception as e:
        logger.error(f"Qdrant search error: {e}")
        raise
