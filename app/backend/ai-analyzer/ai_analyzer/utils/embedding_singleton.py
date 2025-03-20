import os
import logging
from typing import Optional, List
from fastembed import TextEmbedding
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModelSingleton:
    _instance = None
    _lock = threading.Lock()
    _model = None
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(
                    EmbeddingModelSingleton, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    try:
                        # Get the model path from environment variable or use default
                        model_path = os.getenv('EMBEDDING_MODEL_PATH', os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), 'embedding_model'))

                        # Create the model directory if it doesn't exist
                        os.makedirs(model_path, exist_ok=True)

                        # Initialize the model
                        self._model = TextEmbedding(model_path=model_path)
                        self._initialized = True
                        logger.info(
                            f"Embedding model initialized successfully from {model_path}")
                    except Exception as e:
                        logger.error(
                            f"Error initializing embedding model: {e}")
                        logger.exception("Detailed error:")
                        raise

    @property
    def model(self) -> TextEmbedding:
        if not self._model:
            self.__init__()
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            return list(self.model.embed(texts))
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            logger.exception("Detailed error:")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text"""
        try:
            return list(self.model.embed([text]))[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            logger.exception("Detailed error:")
            raise


def get_embedding_model() -> TextEmbedding:
    """Get the singleton instance of the embedding model"""
    try:
        singleton = EmbeddingModelSingleton()
        return singleton.model
    except Exception as e:
        logger.error(f"Error getting embedding model: {e}")
        logger.exception("Detailed error:")
        raise
