# ai_analyzer/utils/singleton_embeddings.py
import logging
from typing import List, Optional
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
import numpy as np

logger = logging.getLogger(__name__)


class SingletonEmbeddings(Embeddings):
    """
    A custom embeddings class that directly wraps our singleton TextEmbedding model
    without any additional initialization or downloading of model files.
    """

    def __init__(self, model: TextEmbedding):
        """
        Initialize with an existing TextEmbedding model instance.

        Args:
            model: The singleton TextEmbedding model instance
        """
        self.model = model
        logger.info(
            "SingletonEmbeddings wrapper initialized with existing model instance")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embeddings, one for each document
        """
        try:
            embeddings = list(self.model.embed(texts))
            # Convert any numpy arrays to list
            return [emb.tolist() if isinstance(emb, np.ndarray) else list(emb) for emb in embeddings]
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return zero vectors in case of error to avoid crashing
            return [[0.0] * 384] * len(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query.

        Args:
            text: Query text to embed

        Returns:
            Embedding for the query
        """
        try:
            embeddings = list(self.model.embed([text]))
            if embeddings:
                # Convert numpy array to list if needed
                emb = embeddings[0]
                return emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
            return [0.0] * 384  # Default embedding size for the model
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * 384  # Return zero vector in case of error
