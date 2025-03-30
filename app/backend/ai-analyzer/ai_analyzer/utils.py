import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
import numpy as np
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the embedding model
embedding_model = TextEmbedding(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def get_db_connection(db_url: str):
    """Create a database connection"""
    try:
        engine = create_engine(db_url)
        return engine
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise


def get_qdrant_client():
    """Get a Qdrant client"""
    try:
        logger.info("Attempting to connect to Qdrant at qdrant:6333")
        client = QdrantClient(host="qdrant", port=6333)
        # Test the connection
        try:
            collections = client.get_collections()
            logger.info(
                f"Successfully connected to Qdrant. Found collections: {[c.name for c in collections.collections]}")
        except Exception as e:
            logger.error(
                f"Connected to Qdrant but failed to get collections: {e}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        logger.exception("Detailed connection error:")
        raise


def create_collection_if_not_exists(client: QdrantClient, collection_name: str):
    """Create a collection if it doesn't exist"""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name not in collection_names:
            logger.info(f"Creating new collection: {collection_name}")
            # Create the collection with the embedding dimension
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "fast-paraphrase-multilingual-minilm-l12-v2": {
                        "size": 384,  # Size for MiniLM-L12-v2 embeddings
                        "distance": "Cosine"
                    }
                }
            )
            logger.info(f"Collection {collection_name} created successfully")
        else:
            logger.info(f"Collection {collection_name} already exists")
    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {e}")
        logger.exception("Detailed collection creation error:")
        raise


def fetch_recent_data(engine, tenant_code: str, months: int = 3):
    """Fetch data from the last X months for a specific tenant"""
    try:
        query = text(f"""
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
                tenant_code
            FROM transcription
            WHERE processing_date >= CURRENT_DATE - INTERVAL '{months} months'
            AND tenant_code = :tenant_code
            LIMIT 1000
        """)

        with engine.connect() as conn:
            result = conn.execute(query, {"tenant_code": tenant_code})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            logger.info(f"Fetched {len(df)} records for tenant {tenant_code}")
            return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        logger.exception("Details:")
        return pd.DataFrame()  # Return empty DataFrame instead of raising


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts"""
    try:
        embeddings = list(embedding_model.embed(texts))
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


def prepare_data_for_qdrant(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepare data for Qdrant by generating embeddings"""
    try:
        logger.info(f"Preparing {len(df)} records for Qdrant")

        # Log the available columns for debugging
        logger.info(f"Available columns in DataFrame: {df.columns.tolist()}")

        # Extract text for embedding
        texts = []
        for _, row in df.iterrows():
            # Use only the columns that exist in the DataFrame
            text_parts = []

            if 'topic' in df.columns and pd.notna(row.topic):
                text_parts.append(f"Topic: {row.topic}")

            if 'summary' in df.columns and pd.notna(row.summary):
                text_parts.append(f"Summary: {row.summary}")

            # Combine the available parts
            text = "\n".join(text_parts)
            texts.append(text)

        logger.info(f"Generating embeddings for {len(texts)} texts")

        # Generate embeddings
        try:
            # Convert generator to list
            embeddings = list(embedding_model.embed(texts))
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            logger.exception("Detailed embedding error:")
            raise

        # Prepare records
        records = []
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                # Convert embedding to list and handle numpy arrays
                if hasattr(embeddings[i], 'tolist'):
                    vector = embeddings[i].tolist()
                else:
                    vector = list(embeddings[i])

                # Create payload with only the columns that exist
                payload = {
                    "db_id": str(row.id),
                    "transcription_id": str(row.transcription_id)
                }

                # Add optional fields if they exist
                if 'topic' in df.columns and pd.notna(row.topic):
                    payload["topic"] = row.topic

                if 'summary' in df.columns and pd.notna(row.summary):
                    payload["summary"] = row.summary

                if 'processing_date' in df.columns and pd.notna(row.processing_date):
                    payload["processing_date"] = str(row.processing_date)

                if 'sentiment' in df.columns and pd.notna(row.sentiment):
                    payload["sentiment"] = row.sentiment

                if 'call_duration_secs' in df.columns and pd.notna(row.call_duration_secs):
                    payload["call_duration_secs"] = int(row.call_duration_secs)

                if 'call_direction' in df.columns and pd.notna(row.call_direction):
                    payload["call_direction"] = row.call_direction

                record = {
                    "id": str(uuid.uuid4()),
                    "vector": vector,
                    "payload": payload
                }
                records.append(record)
            except Exception as e:
                logger.error(f"Error creating record for row {i}: {e}")
                logger.exception(f"Row data: {row.to_dict()}")

        logger.info(f"Created {len(records)} records for Qdrant")
        return records
    except Exception as e:
        logger.error(f"Error preparing data for Qdrant: {e}")
        logger.exception("Detailed preparation error:")
        raise


def update_qdrant_collection(client: QdrantClient, collection_name: str, records: List[Dict[str, Any]]):
    """Update a Qdrant collection with new records"""
    try:
        # Create collection if it doesn't exist
        create_collection_if_not_exists(client, collection_name)

        logger.info(
            f"Preparing to insert {len(records)} records into {collection_name}")

        # Insert records in batches
        batch_size = 100
        logger.info(
            f"Inserting {len(records)} records in batches of {batch_size}")

        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]

            points = [
                models.PointStruct(
                    id=record["id"],
                    vector=record["vector"],
                    payload=record["payload"]
                )
                for record in batch
            ]

            logger.info(
                f"Upserting batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1} into {collection_name}")

            # Add more detailed logging
            try:
                upsert_result = client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                logger.info(f"Upsert result: {upsert_result}")
            except Exception as e:
                logger.error(f"Error upserting batch: {e}")
                logger.exception("Detailed upsert error:")
                raise

            logger.info(
                f"Inserted batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1} into {collection_name}")

        # Verify collection was created and has data
        try:
            # Use the count method directly to get the number of vectors
            count_result = client.count(collection_name=collection_name)
            vector_count = count_result.count

            logger.info(
                f"Collection {collection_name} created with {vector_count} vectors")
        except Exception as e:
            logger.error(f"Error verifying collection {collection_name}: {e}")
            logger.exception("Detailed verification error:")

        logger.info(
            f"Updated collection {collection_name} with {len(records)} records")
    except Exception as e:
        logger.error(f"Error updating Qdrant collection: {e}")
        logger.exception("Detailed error:")
        raise


def update_tenant_vector_db(db_url: str, tenant_code: str, months: int = 3):
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
                    record_metadata["call_duration_secs"] = int(
                        row.call_duration_secs)
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
                    ids=ids,
                    vector_name="fast-paraphrase-multilingual-minilm-l12-v2"
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
