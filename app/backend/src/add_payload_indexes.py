#!/usr/bin/env python3
"""
Add payload indexes to Qdrant collection for fast filtering.

This script creates indexes on 'country' and 'version' fields to dramatically
speed up filtered queries on large collections (600K+ recipes).

Without indexes: O(n) - scans all records
With indexes: O(log n) - instant lookup

Usage:
    python src/add_payload_indexes.py --host qdrant --port 6333
    
Or from docker:
    docker-compose exec backend_app python src/add_payload_indexes.py --host qdrant
"""
import argparse
import logging
import sys
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_payload_indexes(
    host: str = "qdrant",
    port: int = 6333,
    collection_name: str = "food_recipes_two_step"
):
    """
    Create payload indexes for fast filtering on country and version fields.
    
    Args:
        host: Qdrant server host
        port: Qdrant server port
        collection_name: Name of the collection to index
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Connecting to Qdrant at {host}:{port}...")
        client = QdrantClient(host=host, port=port, timeout=120.0)
        
        # Verify collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection_name not in collection_names:
            logger.error(f"Collection '{collection_name}' not found!")
            logger.info(f"Available collections: {collection_names}")
            return False
        
        logger.info(f"Found collection: {collection_name}")
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        logger.info(f"Collection has {collection_info.points_count:,} points")
        
        # Check existing indexes
        logger.info("\n" + "="*80)
        logger.info("CHECKING EXISTING INDEXES")
        logger.info("="*80)
        
        existing_indexes = []
        if hasattr(collection_info.config, 'params') and hasattr(collection_info.config.params, 'payload_schema'):
            if collection_info.config.params.payload_schema:
                for field_name, field_info in collection_info.config.params.payload_schema.items():
                    existing_indexes.append(field_name)
                    logger.info(f"  ✓ Existing index on field: {field_name}")
        
        if not existing_indexes:
            logger.info("  No existing payload indexes found")
        
        # Create indexes for country and version fields
        logger.info("\n" + "="*80)
        logger.info("CREATING PAYLOAD INDEXES")
        logger.info("="*80)
        
        fields_to_index = [
            ("country", "Country filter for recipe searches"),
            ("version", "Version filter (P, L, Missing)")
        ]
        
        for field_name, description in fields_to_index:
            if field_name in existing_indexes:
                logger.info(f"\n  ⏭  Index '{field_name}' already exists - skipping")
                continue
            
            logger.info(f"\n  Creating index on field: {field_name}")
            logger.info(f"    Purpose: {description}")
            logger.info(f"    Type: keyword (exact match)")
            
            try:
                # Create keyword index for exact matching
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                    wait=True  # Wait for index to be built
                )
                logger.info(f"  ✅ Index created successfully on '{field_name}'")
                
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"  ⏭  Index '{field_name}' already exists")
                else:
                    logger.error(f"  ❌ Failed to create index on '{field_name}': {e}")
                    return False
        
        # Verify indexes were created
        logger.info("\n" + "="*80)
        logger.info("VERIFYING INDEXES")
        logger.info("="*80)
        
        collection_info = client.get_collection(collection_name)
        if hasattr(collection_info.config, 'params') and hasattr(collection_info.config.params, 'payload_schema'):
            if collection_info.config.params.payload_schema:
                logger.info("  Current payload indexes:")
                for field_name, field_info in collection_info.config.params.payload_schema.items():
                    logger.info(f"    ✓ {field_name}: {field_info}")
            else:
                logger.warning("  No payload schema found after index creation")
        
        # Test query performance
        logger.info("\n" + "="*80)
        logger.info("TESTING QUERY PERFORMANCE")
        logger.info("="*80)
        
        import time
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        
        # Create a dummy query vector for testing
        test_vector = [0.1] * 384  # Text vector size
        
        test_filters = [
            ("No filter", None),
            ("Austria only", Filter(must=[FieldCondition(key="country", match=MatchValue(value="Austria"))])),
            ("Version P only", Filter(must=[FieldCondition(key="version", match=MatchValue(value="P"))])),
            ("Austria + Version P", Filter(must=[
                FieldCondition(key="country", match=MatchValue(value="Austria")),
                FieldCondition(key="version", match=MatchValue(value="P"))
            ])),
            ("Austria + Version L", Filter(must=[
                FieldCondition(key="country", match=MatchValue(value="Austria")),
                FieldCondition(key="version", match=MatchValue(value="L"))
            ]))
        ]
        
        for test_name, test_filter in test_filters:
            try:
                start_time = time.time()
                results = client.search(
                    collection_name=collection_name,
                    query_vector=("text", test_vector),
                    query_filter=test_filter,
                    limit=15,
                    with_payload=False,
                    with_vectors=False,
                    timeout=30
                )
                elapsed = time.time() - start_time
                logger.info(f"  ✅ {test_name:30s} | {len(results)} results | {elapsed:.3f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"  ❌ {test_name:30s} | TIMEOUT/ERROR after {elapsed:.3f}s: {e}")
        
        logger.info("\n" + "="*80)
        logger.info("INDEX CREATION COMPLETE!")
        logger.info("="*80)
        logger.info("\n✅ Payload indexes have been created successfully")
        logger.info("   Filtered queries should now be much faster")
        logger.info("   Restart your backend to ensure all changes take effect")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating payload indexes: {e}")
        logger.exception("Detailed error:")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create payload indexes in Qdrant for fast filtering"
    )
    parser.add_argument(
        "--host",
        type=str,
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
        type=str,
        default="food_recipes_two_step",
        help="Collection name (default: food_recipes_two_step)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("QDRANT PAYLOAD INDEX CREATOR")
    logger.info("="*80)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Collection: {args.collection}")
    logger.info("="*80)
    
    success = create_payload_indexes(
        host=args.host,
        port=args.port,
        collection_name=args.collection
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
