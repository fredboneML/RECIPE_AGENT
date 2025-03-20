import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_tenant_vector_db(tenant_code: str, documents: List[Dict[str, Any]]) -> None:
    """Update the vector database for a specific tenant with new documents."""
    try:
        # Get singleton instances
        from ai_analyzer.utils.qdrant_client import get_qdrant_client, get_embedding_model
        qdrant_client = get_qdrant_client()
        embedding_model = get_embedding_model()

        collection_name = f"tenant_{tenant_code}"

        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_exists = any(
            c.name == collection_name for c in collections.collections)

        if not collection_exists:
            # Create collection if it doesn't exist
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text": {
                        "size": 384,  # Size for MiniLM-L12-v2 embeddings
                        "distance": "Cosine"
                    }
                }
            )
            logger.info(f"Created new collection: {collection_name}")

        # Prepare documents for upload
        points = []
        for doc in documents:
            # Generate embedding for the text
            text = doc.get("text", "")
            if not text:
                continue

            embeddings = list(embedding_model.embed([text]))
            vector = embeddings[0].tolist()

            # Create point with metadata
            point = {
                "id": doc.get("id", ""),
                "vector": vector,
                "payload": {
                    "text": text,
                    "call_id": doc.get("call_id", ""),
                    "timestamp": doc.get("timestamp", ""),
                    **doc.get("metadata", {})
                }
            }
            points.append(point)

        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )
            logger.info(
                f"Uploaded batch of {len(batch)} documents to {collection_name}")

    except Exception as e:
        logger.error(f"Error updating vector database: {e}")
        logger.exception("Detailed error:")
        raise
