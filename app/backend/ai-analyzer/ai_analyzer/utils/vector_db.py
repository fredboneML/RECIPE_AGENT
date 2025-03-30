import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sqlalchemy import create_engine
import pandas as pd
from ai_analyzer.utils.qdrant_client import get_qdrant_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection(db_url: str):
    """Create a database connection"""
    return create_engine(db_url)


def fetch_recent_data(engine, tenant_code: str, months: int) -> pd.DataFrame:
    """Fetch recent data from the database"""
    query = f"""
        SELECT 
            id, 
            transcription_id, 
            topic, 
            summary, 
            processing_date, 
            sentiment, 
            call_duration_secs, 
            telephone_number, 
            call_direction,
            tenant_code,
            transcription
        FROM transcription
        WHERE processing_date >= CURRENT_DATE - INTERVAL '{months} months'
        AND tenant_code = '{tenant_code}'
        LIMIT 1000
    """

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        logger.info(f"Fetched {len(df)} records for tenant {tenant_code}")
        return df


def update_tenant_vector_db(db_url: str, tenant_code: str, months: int = 3) -> None:
    """Update the vector database for a specific tenant with data from the last X months"""
    try:
        # Get database connection
        engine = get_db_connection(db_url)

        # Get Qdrant client
        client = get_qdrant_client()

        # Collection name based on tenant code
        collection_name = f"tenant_{tenant_code}"

        # Fetch recent data
        df = fetch_recent_data(engine, tenant_code, months)

        if len(df) > 0:
            logger.info(f"Found {len(df)} records for tenant {tenant_code}")

            # Check if collection exists
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            collection_exists = collection_name in collection_names

            # Get existing record IDs if collection exists
            existing_ids = set()
            if collection_exists:
                try:
                    # Query for all existing points to get their IDs and metadata
                    existing_points = client.scroll(
                        collection_name=collection_name,
                        limit=10000,  # Adjust based on expected volume
                        with_payload=True
                    )[0]

                    # Extract IDs and processing dates
                    for point in existing_points:
                        point_id = point.id
                        existing_ids.add(point_id)

                    logger.info(
                        f"Found {len(existing_ids)} existing records in collection {collection_name}")
                except Exception as e:
                    logger.warning(f"Error fetching existing records: {e}")
                    # If we can't get existing records, we'll recreate the collection
                    client.delete_collection(collection_name=collection_name)
                    collection_exists = False
                    logger.info(
                        f"Recreated collection {collection_name} due to error")

            # If collection doesn't exist, create it
            if not collection_exists:
                logger.info(f"Creating new collection {collection_name}")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "fast-paraphrase-multilingual-minilm-l12-v2": {
                            "size": 384,  # Size for MiniLM-L12-v2 embeddings
                            "distance": "Cosine"
                        }
                    }
                )
                logger.info(f"Created new collection: {collection_name}")

            # Prepare documents and metadata
            documents = []
            metadata = []
            ids = []

            # Track new record IDs to avoid duplicates
            new_record_ids = set()

            for _, row in df.iterrows():
                record_id = str(row.id)

                # Skip if this record is already in the collection
                if record_id in existing_ids and record_id in new_record_ids:
                    continue

                # Add to tracking set
                new_record_ids.add(record_id)

                # Combine text fields for embedding
                text_parts = []
                if 'transcription' in df.columns and pd.notna(row.transcription):
                    text_parts.append(row.transcription)
                if 'summary' in df.columns and pd.notna(row.summary):
                    text_parts.append(f"Summary: {row.summary}")
                if 'topic' in df.columns and pd.notna(row.topic):
                    text_parts.append(f"Topic: {row.topic}")

                document = " ".join(text_parts)
                documents.append(document)

                # Prepare metadata
                record_metadata = {
                    "id": record_id,
                    "tenant_code": tenant_code,
                    "processing_date": str(row.processing_date) if 'processing_date' in df.columns and pd.notna(row.processing_date) else None
                }

                # Add optional fields to metadata
                if 'topic' in df.columns and pd.notna(row.topic):
                    record_metadata["topic"] = row.topic
                if 'summary' in df.columns and pd.notna(row.summary):
                    record_metadata["summary"] = row.summary
                if 'sentiment' in df.columns and pd.notna(row.sentiment):
                    record_metadata["sentiment"] = row.sentiment
                if 'call_duration_secs' in df.columns and pd.notna(row.call_duration_secs):
                    record_metadata["call_duration_secs"] = str(
                        int(row.call_duration_secs))
                if 'call_direction' in df.columns and pd.notna(row.call_direction):
                    record_metadata["call_direction"] = row.call_direction

                metadata.append(record_metadata)
                ids.append(record_id)

            # Find records to delete (existing records not in the new dataset)
            ids_to_delete = existing_ids - new_record_ids
            if ids_to_delete:
                logger.info(
                    f"Deleting {len(ids_to_delete)} outdated records from collection {collection_name}")
                try:
                    client.delete(
                        collection_name=collection_name,
                        points_selector=models.PointIdsList(
                            points=list(ids_to_delete)
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error deleting outdated records: {e}")

            # Only add new records if there are any
            if documents:
                logger.info(
                    f"Adding {len(documents)} new documents to collection {collection_name}")

                # Set the embedding model for the client
                client.set_model(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

                # Add documents with metadata
                client.add(
                    collection_name=collection_name,
                    documents=documents,
                    metadata=metadata,
                    ids=ids
                )
            else:
                logger.info(
                    f"No new documents to add to collection {collection_name}")

            # Verify collection has data
            try:
                # Use the count method directly to get the number of vectors
                count_result = client.count(collection_name=collection_name)
                vector_count = count_result.count

                logger.info(
                    f"Vector database for tenant {tenant_code} now contains {vector_count} records")
            except Exception as e:
                logger.error(f"Error getting collection info: {e}")
                logger.exception("Detailed collection info error:")

            logger.info(
                f"Successfully updated vector database for tenant {tenant_code}")
        else:
            logger.warning(
                f"No data found for tenant {tenant_code} in the last {months} months")
    except Exception as e:
        logger.error(
            f"Error updating vector database for tenant {tenant_code}: {e}")
        logger.exception("Detailed error:")
        raise
