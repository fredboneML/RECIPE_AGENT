#!/usr/bin/env python3
"""
Add Full-Text Index to Qdrant Collection

This script adds a full-text index on the 'recipe_name' field to enable
efficient keyword-based searches across 600K+ recipes.

IMPORTANT: This does NOT require reindexing vectors. It only creates an
index on the existing payload field for fast text searching.

Usage:
    python add_text_index.py [--host QDRANT_HOST] [--port QDRANT_PORT] [--collection COLLECTION_NAME]

Example:
    # Local development
    python add_text_index.py --host localhost --port 6333
    
    # Docker environment
    python add_text_index.py --host qdrant --port 6333
"""

import argparse
import logging
import sys
import time
from qdrant_client import QdrantClient
from qdrant_client.models import TextIndexParams, TokenizerType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_text_index(
    host: str = "qdrant",
    port: int = 6333,
    collection_name: str = "food_recipes_two_step"
) -> bool:
    """
    Add full-text index on recipe_name field for efficient keyword search.

    Args:
        host: Qdrant server host
        port: Qdrant server port
        collection_name: Name of the collection

    Returns:
        True if successful, False otherwise
    """
    try:
        # Connect to Qdrant with extended timeout for large collections
        logger.info(f"Connecting to Qdrant at {host}:{port}...")
        client = QdrantClient(host=host, port=port,
                              timeout=300)  # 5 minute timeout

        # Verify collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name not in collection_names:
            logger.error(f"Collection '{collection_name}' not found!")
            logger.error(f"Available collections: {collection_names}")
            return False

        # Get collection info
        collection_info = client.get_collection(collection_name)
        logger.info(
            f"Collection '{collection_name}' has {collection_info.points_count} points")

        # Check existing indexes
        existing_indexes = collection_info.payload_schema
        logger.info(
            f"Existing payload indexes: {list(existing_indexes.keys()) if existing_indexes else 'None'}")

        # Define text index parameters
        # Using 'word' tokenizer for product names like "ZB TYP GYROS OFENK. GTF H.T. GLUFR VEGAN"
        text_index_params = TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,  # Split on whitespace and punctuation
            min_token_len=2,  # Allow short tokens like "ZB"
            max_token_len=40,  # Allow long tokens
            lowercase=True,  # Case-insensitive matching
        )

        # Create index on recipe_name field
        logger.info("Creating full-text index on 'recipe_name' field...")
        logger.info("  Tokenizer: WORD (splits on whitespace/punctuation)")
        logger.info("  Min token length: 2")
        logger.info("  Lowercase: True (case-insensitive)")
        logger.info(
            f"  Collection size: {collection_info.points_count} points")
        logger.info("  This may take several minutes for large collections...")

        # Use wait=False for large collections to avoid timeout, then poll for completion
        client.create_payload_index(
            collection_name=collection_name,
            field_name="recipe_name",
            field_schema=text_index_params,
            wait=False  # Don't wait - index creation happens in background
        )

        # Poll until index is created
        logger.info("  Index creation started, waiting for completion...")
        max_wait_seconds = 600  # 10 minutes max
        poll_interval = 5  # Check every 5 seconds
        elapsed = 0

        while elapsed < max_wait_seconds:
            time.sleep(poll_interval)
            elapsed += poll_interval

            # Check if index exists
            info = client.get_collection(collection_name)
            if info.payload_schema and "recipe_name" in info.payload_schema:
                logger.info(
                    f"✓ Full-text index on 'recipe_name' created successfully! (took {elapsed}s)")
                break

            if elapsed % 30 == 0:  # Log progress every 30 seconds
                logger.info(f"  Still indexing... ({elapsed}s elapsed)")
        else:
            logger.warning(
                f"  Index creation still in progress after {max_wait_seconds}s - continuing anyway")

        # Also create index on description field for broader keyword matching
        logger.info("Creating full-text index on 'description' field...")
        logger.info("  This may take several minutes for large collections...")

        client.create_payload_index(
            collection_name=collection_name,
            field_name="description",
            field_schema=text_index_params,
            wait=False  # Don't wait - index creation happens in background
        )

        # Poll until index is created
        logger.info("  Index creation started, waiting for completion...")
        elapsed = 0

        while elapsed < max_wait_seconds:
            time.sleep(poll_interval)
            elapsed += poll_interval

            # Check if index exists
            info = client.get_collection(collection_name)
            if info.payload_schema and "description" in info.payload_schema:
                logger.info(
                    f"✓ Full-text index on 'description' created successfully! (took {elapsed}s)")
                break

            if elapsed % 30 == 0:  # Log progress every 30 seconds
                logger.info(f"  Still indexing... ({elapsed}s elapsed)")
        else:
            logger.warning(
                f"  Index creation still in progress after {max_wait_seconds}s - continuing anyway")

        # Verify indexes were created
        collection_info = client.get_collection(collection_name)
        new_indexes = collection_info.payload_schema
        logger.info(
            f"Current payload indexes: {list(new_indexes.keys()) if new_indexes else 'None'}")

        # Test the index with a sample search
        logger.info("\nTesting full-text search...")
        from qdrant_client.models import Filter, FieldCondition, MatchText

        test_results, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="recipe_name",
                        match=MatchText(text="gyros")
                    )
                ]
            ),
            limit=5,
            with_payload=True,
            with_vectors=False
        )

        logger.info(
            f"Test search for 'gyros' found {len(test_results)} results:")
        for i, result in enumerate(test_results, 1):
            recipe_name = result.payload.get(
                "recipe_name", "Unknown") if result.payload else "Unknown"
            logger.info(f"  {i}. {recipe_name}")

        if test_results:
            logger.info("\n✅ Full-text index is working correctly!")
        else:
            logger.warning(
                "\n⚠️ No results found - this may be expected if no recipes contain 'gyros'")

        return True

    except Exception as e:
        logger.error(f"Failed to create text index: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add full-text index to Qdrant collection for efficient keyword search"
    )
    parser.add_argument(
        "--host",
        default="qdrant",
        help="Qdrant server host (default: qdrant)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant server port (default: 6333)"
    )
    parser.add_argument(
        "--collection",
        default="food_recipes_two_step",
        help="Collection name (default: food_recipes_two_step)"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Adding Full-Text Index to Qdrant Collection")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Collection: {args.collection}")
    logger.info("=" * 60)

    success = add_text_index(
        host=args.host,
        port=args.port,
        collection_name=args.collection
    )

    if success:
        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS! Full-text indexes have been created.")
        logger.info(
            "The flavor safeguard keyword search will now work efficiently.")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("\n" + "=" * 60)
        logger.error("FAILED to create full-text indexes.")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
