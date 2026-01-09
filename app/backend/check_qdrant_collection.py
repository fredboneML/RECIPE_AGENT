#!/usr/bin/env python3
"""
Diagnostic script to check Qdrant collection configuration
Run this to diagnose timeout and vector name issues
"""
import logging
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_collection_config():
    """Check Qdrant collection configuration and statistics"""
    try:
        # Connect to Qdrant
        client = QdrantClient(host="qdrant", port=6333, timeout=60.0)
        logger.info("Connected to Qdrant successfully")
        
        # Get all collections
        collections = client.get_collections()
        logger.info(f"\n{'='*80}")
        logger.info("AVAILABLE COLLECTIONS:")
        logger.info(f"{'='*80}")
        for c in collections.collections:
            logger.info(f"  - {c.name}")
        
        # Check the main recipe collection
        collection_name = "food_recipes_two_step"
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING COLLECTION: {collection_name}")
        logger.info(f"{'='*80}")
        
        collection_info = client.get_collection(collection_name)
        
        # Vector configuration
        logger.info(f"\nVECTOR CONFIGURATION:")
        logger.info(f"  Vector names: {list(collection_info.config.params.vectors.keys())}")
        for vector_name, vector_config in collection_info.config.params.vectors.items():
            logger.info(f"  - {vector_name}:")
            logger.info(f"      Size: {vector_config.size}")
            logger.info(f"      Distance: {vector_config.distance}")
        
        # Collection statistics
        logger.info(f"\nCOLLECTION STATISTICS:")
        logger.info(f"  Total points: {collection_info.points_count:,}")
        logger.info(f"  Vectors count: {collection_info.vectors_count:,}")
        logger.info(f"  Indexed vectors: {collection_info.indexed_vectors_count:,}")
        logger.info(f"  Segments count: {collection_info.segments_count}")
        logger.info(f"  Status: {collection_info.status}")
        
        # Check payload schema
        if hasattr(collection_info.config, 'payload_schema') and collection_info.config.payload_schema:
            logger.info(f"\nPAYLOAD SCHEMA:")
            for field_name, field_config in collection_info.config.payload_schema.items():
                logger.info(f"  - {field_name}: {field_config}")
        
        # Sample query to test performance
        logger.info(f"\n{'='*80}")
        logger.info("TESTING QUERIES:")
        logger.info(f"{'='*80}")
        
        # Test 1: Count by version
        logger.info("\nTest 1: Counting recipes by version filter...")
        for version in ['P', 'L']:
            try:
                result = client.scroll(
                    collection_name=collection_name,
                    scroll_filter={
                        "must": [{"key": "version", "match": {"value": version}}]
                    },
                    limit=1,
                    with_payload=False,
                    with_vectors=False
                )
                logger.info(f"  Version {version}: Found at least {len(result[0])} recipes")
            except Exception as e:
                logger.error(f"  Version {version}: ERROR - {e}")
        
        # Test 2: Count by country + version
        logger.info("\nTest 2: Counting recipes by country=Austria + version filter...")
        for version in ['P', 'L']:
            try:
                result = client.scroll(
                    collection_name=collection_name,
                    scroll_filter={
                        "must": [
                            {"key": "country", "match": {"value": "Austria"}},
                            {"key": "version", "match": {"value": version}}
                        ]
                    },
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                    timeout=30
                )
                logger.info(f"  Austria + Version {version}: Found at least {len(result[0])} recipes")
            except Exception as e:
                logger.error(f"  Austria + Version {version}: ERROR - {e}")
        
        logger.info(f"\n{'='*80}")
        logger.info("DIAGNOSIS COMPLETE")
        logger.info(f"{'='*80}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking collection: {e}")
        logger.exception("Detailed error:")
        return False


if __name__ == "__main__":
    success = check_collection_config()
    sys.exit(0 if success else 1)
