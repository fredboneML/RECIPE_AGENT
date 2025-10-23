#!/usr/bin/env python3
"""
Vector Index Initialization with Qdrant Storage

This version actually stores vectors in Qdrant instead of just in-memory.
"""
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
import sys
import time
import logging
import json
import pandas as pd
from pathlib import Path
import uuid

# Add the current directory to Python path to import local modules
sys.path.insert(0, '/usr/src/app')
sys.path.insert(0, '/usr/src/app/src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vector_index_init')


def wait_for_qdrant(qdrant_host="qdrant", qdrant_port=6333, max_retries=30, delay=2):
    """Wait for Qdrant to be ready"""
    for attempt in range(max_retries):
        try:
            client = QdrantClient(host=qdrant_host, port=qdrant_port)
            collections = client.get_collections()
            logger.info("Qdrant is ready!")
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                logger.info(
                    f"Qdrant not ready (attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(
                    f"Failed to connect to Qdrant after {max_retries} attempts")
                raise


def read_recipe_json(recipe_json_path):
    """Read recipe JSON file and extract features/values."""
    result = None
    try:
        with open(recipe_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Reading: {os.path.basename(recipe_json_path)}")

        if not data:
            logger.warning(f"The JSON file {recipe_json_path} is empty")
        else:
            # Handle both array and single object formats
            if isinstance(data, list) and len(data) > 0:
                # Array format - look for Classification in each item
                for item in data:
                    if isinstance(item, dict) and 'Classification' in item and item['Classification']:
                        if 'valueschar' in item['Classification'] and item['Classification']['valueschar']:
                            result = pd.DataFrame(
                                item['Classification']['valueschar'])
                            logger.info(
                                f"Found Classification data in array format")
                            break
            elif isinstance(data, dict) and 'Classification' in data:
                # Single object format
                if data['Classification'] and 'valueschar' in data['Classification'] and data['Classification']['valueschar']:
                    result = pd.DataFrame(data['Classification']['valueschar'])
                    logger.info(f"Found Classification data in object format")

            if result is None:
                logger.warning(
                    f"Classification/valueschar not found in {recipe_json_path}")
    except Exception as e:
        logger.error(f"Error reading {recipe_json_path}: {e}")

    return result


def clean_tdline(tdline: str) -> str:
    """If 'Colour', 'Flavour', or 'Stabilizer' exist in tdline, keep it as is."""
    keywords = ["Colour", "Flavour", "Stabilizer"]
    if any(word in tdline for word in keywords):
        return tdline
    else:
        return ""


def extract_recipe_name(filename: str) -> str:
    """Extracts the recipe name from a given filename."""
    filename = os.path.basename(filename)
    parts = filename.split("_")

    if len(parts) > 2:
        rest = "_".join(parts[2:])
    else:
        rest = ""

    rest = rest.replace(".json", "")
    alpha_count = sum(1 for c in rest if c.isalpha())

    if alpha_count >= 3:
        rest = rest.replace("_", " ")
        return f"Recipe Name: {rest.strip()}"
    else:
        return "Recipe Name: "


def extract_recipe_description(recipe_json_path):
    """Extract comprehensive recipe description."""
    import re

    description_parts = []
    replacements = {
        r"(?i)\bfl_": "Flavour ",
        r"(?i)\bco_": "Colour ",
        r"(?i)\bstab_": "Stabilizer "
    }

    try:
        recipe_name = extract_recipe_name(recipe_json_path)
        if recipe_name.strip() != "Recipe Name:":
            description_parts.append(recipe_name)

        with open(recipe_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both array and single object formats
        if isinstance(data, list) and len(data) > 0:
            # Array format - use the first item that has the data we need
            data = data[0]
        elif not isinstance(data, dict):
            logger.warning(f"Unexpected data format in {recipe_json_path}")
            return f"Recipe {os.path.basename(recipe_json_path).replace('.json', '')}"

        if 'MaterialMasterShorttext' in data.keys() and data['MaterialMasterShorttext']:
            try:
                material_master_short_text = data['MaterialMasterShorttext'][0]['maktx']
                if material_master_short_text and material_master_short_text.strip():
                    description_parts.append(
                        f"MaterialMasterShorttext: {material_master_short_text}")
            except (IndexError, KeyError, TypeError) as e:
                logger.warning(
                    f"Error extracting MaterialMasterShorttext: {e}")

        if 'Texts' in data.keys() and data['Texts']:
            try:
                texts_df = pd.DataFrame(data['Texts'])
                if 'lines' in texts_df.columns and not texts_df.empty:
                    lines_data = texts_df['lines'].iloc[0]
                    if lines_data and len(lines_data) > 0 and 'tdline' in lines_data[0]:
                        tdline = lines_data[0]['tdline']
                        if tdline and tdline.strip():
                            for pattern, repl in replacements.items():
                                tdline = re.sub(pattern, repl, tdline)
                            cleaned_tdline = clean_tdline(tdline)
                            if cleaned_tdline:
                                description_parts.append(cleaned_tdline)
            except (IndexError, KeyError, TypeError) as e:
                logger.warning(f"Error extracting Texts: {e}")

        if len(description_parts) <= 1:
            if 'Classification' in data and 'valueschar' in data['Classification']:
                try:
                    classification_df = pd.DataFrame(
                        data['Classification']['valueschar'])

                    if 'charactDescr' in classification_df.columns and 'valueCharLong' in classification_df.columns:
                        key_features = [
                            'Product Line', 'Customer Brand', 'Project title',
                            'Color', 'Flavor', 'Flavour', 'Fruit content', 'Brix',
                            'Produktsegment (SD Reporting)', 'Industry (SD Reporting)'
                        ]

                        classification_parts = []
                        for _, row in classification_df.iterrows():
                            feature = row['charactDescr']
                            value = row['valueCharLong']

                            if (feature in key_features and
                                pd.notna(value) and
                                str(value).strip() and
                                    str(value).strip().lower() not in ['none', '', 'null']):
                                classification_parts.append(
                                    f"{feature}: {value}")

                        if classification_parts:
                            description_parts.extend(classification_parts[:3])

                except Exception as e:
                    logger.warning(f"Error extracting Classification: {e}")

        if description_parts:
            final_description = ", ".join(description_parts)
            return final_description
        else:
            base_name = os.path.basename(recipe_json_path).replace(
                '.json', '').replace('_', ' ').replace('-', ' ')
            return f"Recipe {base_name}"

    except Exception as e:
        logger.error(f"Error extracting description: {e}")
        base_name = os.path.basename(recipe_json_path).replace(
            '.json', '').replace('_', ' ').replace('-', ' ')
        return f"Recipe {base_name}"


def load_recipes_from_data_dir(data_dir):
    """Load all recipe JSON files from data directory"""
    recipe_json_list = []

    if not os.path.exists(data_dir):
        logger.warning(f"Data directory does not exist: {data_dir}")
        return recipe_json_list

    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            recipe_json_list.append(os.path.join(data_dir, file))

    logger.info(
        f"Found {len(recipe_json_list)} recipe JSON files in {data_dir}")
    return sorted(recipe_json_list)


def create_qdrant_collection(qdrant_client, collection_name, vector_size=384):
    """Create Qdrant collection if it doesn't exist"""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name in collection_names:
            logger.info(
                f"Collection '{collection_name}' already exists.")
            return True

        # Create collection with proper vector configuration
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        logger.info(
            f"Created collection '{collection_name}' with vector size {vector_size}")
        return True

    except Exception as e:
        logger.error(f"Error creating Qdrant collection: {e}")
        return False


def index_recipes_to_qdrant(qdrant_client, embedding_model, collection_name, data_dir):
    """Index recipes directly into Qdrant"""
    try:
        # Load recipe files
        recipe_json_list = load_recipes_from_data_dir(data_dir)

        if not recipe_json_list:
            logger.info("No recipes found to index. Skipping indexing step.")
            return True

        logger.info(
            f"Processing {len(recipe_json_list)} recipes for Qdrant indexing...")

        # Initialize sentence transformer for embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        model = SentenceTransformer(embedding_model)

        # Process recipes in batches
        batch_size = 10
        points = []

        for i, recipe_path in enumerate(recipe_json_list):
            try:
                # Extract features and values
                recipe_data = read_recipe_json(recipe_path)

                # Extract description
                description = extract_recipe_description(recipe_path)

                if recipe_data is not None and isinstance(recipe_data, pd.DataFrame):
                    if 'charactDescr' in recipe_data.columns and 'valueCharLong' in recipe_data.columns:
                        recipe_temp = recipe_data[[
                            'charactDescr', 'valueCharLong']]
                        features = recipe_temp['charactDescr'].tolist()
                        values = recipe_temp['valueCharLong'].tolist()

                        recipe_filename = os.path.basename(
                            recipe_path).split('.')[0]

                        # Create embedding from description
                        embedding = model.encode(description)

                        # Create point for Qdrant
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding.tolist(),
                            payload={
                                "recipe_name": recipe_filename,
                                "description": description,
                                # Limit for payload size
                                "features": features[:50],
                                "values": [str(v) for v in values[:50]],
                                "num_features": len(features)
                            }
                        )
                        points.append(point)

                        logger.info(
                            f"Prepared recipe {i+1}/{len(recipe_json_list)}: {recipe_filename}")

                        # Upload in batches
                        if len(points) >= batch_size:
                            qdrant_client.upsert(
                                collection_name=collection_name,
                                points=points
                            )
                            logger.info(
                                f"Uploaded batch of {len(points)} recipes to Qdrant")
                            points = []

                    else:
                        logger.warning(
                            f"Missing required columns in {recipe_path}")
                else:
                    logger.warning(
                        f"Skipping invalid recipe: {recipe_path}")
            except Exception as e:
                logger.error(f"Error processing recipe {recipe_path}: {e}")
                continue

        # Upload remaining points
        if points:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(
                f"Uploaded final batch of {len(points)} recipes to Qdrant")

        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        logger.info(
            f"Successfully indexed {collection_info.points_count} recipes in Qdrant")

        return True

    except Exception as e:
        logger.error(f"Error indexing recipes to Qdrant: {e}")
        return False


def main():
    """Main function to initialize vector index and index recipes in Qdrant"""
    try:
        # Get configuration from environment
        qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
        qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        collection_name = os.getenv(
            'RECIPE_COLLECTION_NAME', 'food_recipes_two_step')
        embedding_model = os.getenv(
            'EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
        data_dir = os.getenv(
            'RECIPE_DATA_DIR', '/usr/src/app/data')

        logger.info("=" * 60)
        logger.info("Starting Qdrant Vector Index Initialization")
        logger.info("=" * 60)
        logger.info(f"Qdrant Host: {qdrant_host}")
        logger.info(f"Qdrant Port: {qdrant_port}")
        logger.info(f"Collection Name: {collection_name}")
        logger.info(f"Embedding Model: {embedding_model}")
        logger.info(f"Data Directory: {data_dir}")
        logger.info("=" * 60)

        # Wait for Qdrant to be ready
        logger.info("Waiting for Qdrant to be ready...")
        qdrant_client = wait_for_qdrant(qdrant_host, qdrant_port)

        # Create Qdrant collection
        logger.info("Creating Qdrant collection...")
        create_qdrant_collection(
            qdrant_client, collection_name, vector_size=384)

        # Index recipes directly into Qdrant
        logger.info("Indexing recipes to Qdrant...")
        success = index_recipes_to_qdrant(
            qdrant_client, embedding_model, collection_name, data_dir)

        if success:
            logger.info("=" * 60)
            logger.info(
                "Qdrant Vector Index Initialization Completed Successfully!")
            logger.info("=" * 60)
        else:
            logger.warning("=" * 60)
            logger.warning(
                "Qdrant Vector Index Initialization Completed with Warnings")
            logger.warning("=" * 60)

    except Exception as e:
        logger.error(f"Error during vector index initialization: {e}")
        logger.exception("Detailed error:")
        raise


if __name__ == "__main__":
    main()
