#!/usr/bin/env python3

from qdrant_client import QdrantClient
import logging
import sys
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_qdrant')


def test_qdrant_connection(retries=5, delay=5):
    """Test connection to Qdrant server"""
    for attempt in range(retries):
        try:
            logger.info(f"Attempt {attempt+1}/{retries} connecting to Qdrant")

            # Try to connect to Qdrant
            client = QdrantClient(host="qdrant", port=6333)

            # Test the connection by listing collections
            collections = client.get_collections()

            logger.info(f"Successfully connected to Qdrant!")
            logger.info(
                f"Found collections: {[c.name for c in collections.collections]}")

            # Create a test collection
            test_collection_name = f"test_collection_{int(time.time())}"

            logger.info(f"Creating test collection '{test_collection_name}'")
            client.create_collection(
                collection_name=test_collection_name,
                vectors_config={
                    "size": 384,
                    "distance": "Cosine"
                }
            )

            # Verify collection was created
            collections = client.get_collections()
            logger.info(
                f"Collections after creation: {[c.name for c in collections.collections]}")

            # Clean up test collection
            if test_collection_name in [c.name for c in collections.collections]:
                logger.info(
                    f"Deleting test collection '{test_collection_name}'")
                client.delete_collection(collection_name=test_collection_name)

            logger.info("Qdrant connection test completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(
                    "Failed to connect to Qdrant after multiple attempts")
                return False


if __name__ == "__main__":
    logger.info("Testing Qdrant connection...")
    result = test_qdrant_connection()
    if not result:
        sys.exit(1)
    logger.info("Test completed successfully")
    sys.exit(0)
