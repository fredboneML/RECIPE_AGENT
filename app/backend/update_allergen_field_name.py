#!/usr/bin/env python3
"""
Script to update 'Allergens' to 'Allergen-free' in existing Qdrant payloads
without reindexing. This only updates the text fields, not the vectors.

Usage:
    python update_allergen_field_name.py [--dry-run] [--batch-size 1000]
"""

import argparse
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, SetPayloadOperation, SetPayload
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Qdrant configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "food_recipes_two_step"

# Text replacements to make
REPLACEMENTS = [
    # In description field
    ("Allergens:", "Allergen-free:"),
    ("Allergens :", "Allergen-free:"),
    # In feature_text field (may have different formats)
    ("Allergens,", "Allergen-free,"),
    ("Allergens ", "Allergen-free "),
]


def update_text_fields(client: QdrantClient, batch_size: int = 1000, dry_run: bool = False):
    """
    Update all records in the collection to replace 'Allergens' with 'Allergen-free'
    in the description and feature_text fields.
    """
    
    # Get collection info
    collection_info = client.get_collection(COLLECTION_NAME)
    total_points = collection_info.points_count
    logger.info(f"Collection has {total_points:,} points")
    
    # Track statistics
    stats = {
        'total_processed': 0,
        'total_updated': 0,
        'description_updates': 0,
        'feature_text_updates': 0,
        'errors': 0,
    }
    
    # Scroll through all points
    offset = None
    batch_num = 0
    
    while True:
        batch_num += 1
        logger.info(f"Processing batch {batch_num} (processed so far: {stats['total_processed']:,})")
        
        # Fetch batch of points
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False  # Don't need vectors, saves bandwidth
        )
        
        points, next_offset = scroll_result
        
        if not points:
            logger.info("No more points to process")
            break
        
        # Process each point in the batch
        updates_in_batch = []
        
        for point in points:
            stats['total_processed'] += 1
            point_id = point.id
            payload = point.payload or {}
            
            # Check and update description
            description = payload.get('description', '')
            feature_text = payload.get('feature_text', '')
            
            new_description = description
            new_feature_text = feature_text
            
            # Apply replacements
            for old_text, new_text in REPLACEMENTS:
                if old_text in new_description:
                    new_description = new_description.replace(old_text, new_text)
                if old_text in new_feature_text:
                    new_feature_text = new_feature_text.replace(old_text, new_text)
            
            # Check if any changes were made
            description_changed = new_description != description
            feature_text_changed = new_feature_text != feature_text
            
            if description_changed or feature_text_changed:
                update_payload = {}
                
                if description_changed:
                    update_payload['description'] = new_description
                    stats['description_updates'] += 1
                
                if feature_text_changed:
                    update_payload['feature_text'] = new_feature_text
                    stats['feature_text_updates'] += 1
                
                updates_in_batch.append({
                    'id': point_id,
                    'payload': update_payload
                })
                stats['total_updated'] += 1
        
        # Apply updates for this batch
        if updates_in_batch and not dry_run:
            try:
                # Process updates in mini-batches for better performance
                mini_batch_size = 100
                for i in range(0, len(updates_in_batch), mini_batch_size):
                    mini_batch = updates_in_batch[i:i + mini_batch_size]
                    for update in mini_batch:
                        client.set_payload(
                            collection_name=COLLECTION_NAME,
                            payload=update['payload'],
                            points=[update['id']],
                            wait=False  # Async - don't wait for each update
                        )
                logger.info(f"  Updated {len(updates_in_batch)} points in this batch")
            except Exception as e:
                logger.error(f"  Error updating batch: {e}")
                stats['errors'] += 1
        elif updates_in_batch and dry_run:
            logger.info(f"  [DRY RUN] Would update {len(updates_in_batch)} points in this batch")
            # Show sample of what would be updated
            if batch_num == 1 and updates_in_batch:
                sample = updates_in_batch[0]
                logger.info(f"  Sample update - Point ID: {sample['id']}")
                for key, value in sample['payload'].items():
                    logger.info(f"    {key}: {value[:100]}..." if len(str(value)) > 100 else f"    {key}: {value}")
        
        # Move to next batch
        offset = next_offset
        
        if offset is None:
            logger.info("Reached end of collection")
            break
        
        # Progress update every 10 batches
        if batch_num % 10 == 0:
            progress = (stats['total_processed'] / total_points) * 100
            logger.info(f"Progress: {progress:.1f}% ({stats['total_processed']:,}/{total_points:,})")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Update 'Allergens' to 'Allergen-free' in Qdrant payloads"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of points to process per batch (default: 1000)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=QDRANT_HOST,
        help=f'Qdrant host (default: {QDRANT_HOST})'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=QDRANT_PORT,
        help=f'Qdrant port (default: {QDRANT_PORT})'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Allergens → Allergen-free Field Update Script")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("*** DRY RUN MODE - No changes will be made ***")
    
    # Connect to Qdrant
    logger.info(f"Connecting to Qdrant at {args.host}:{args.port}...")
    client = QdrantClient(host=args.host, port=args.port, timeout=60)
    
    # Verify connection
    try:
        collections = client.get_collections()
        logger.info(f"Connected successfully. Found {len(collections.collections)} collections.")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return 1
    
    # Run update
    start_time = time.time()
    stats = update_text_fields(
        client,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
    elapsed_time = time.time() - start_time
    
    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total points processed: {stats['total_processed']:,}")
    logger.info(f"Total points updated: {stats['total_updated']:,}")
    logger.info(f"  - Description field updates: {stats['description_updates']:,}")
    logger.info(f"  - Feature text updates: {stats['feature_text_updates']:,}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    if stats['total_processed'] > 0:
        rate = stats['total_processed'] / elapsed_time
        logger.info(f"Processing rate: {rate:.0f} points/second")
    
    if args.dry_run:
        logger.info("\n*** This was a DRY RUN - no changes were made ***")
        logger.info("Run without --dry-run to apply the changes")
    
    return 0


if __name__ == "__main__":
    exit(main())
