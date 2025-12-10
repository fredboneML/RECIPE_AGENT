#!/usr/bin/env python3
"""
Vector Index Initialization with Qdrant Storage - Enhanced Version

This version uses EnhancedTwoStepRecipeManager logic for sophisticated feature encoding
with pre-analyzed feature types from charactDescr_valueCharLong_map.json.
"""
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
import time
import logging
import json
import pandas as pd
from pathlib import Path
import uuid
import re
from feature_analyzer import FeatureAnalyzer
from src.two_step_recipe_search import EnhancedTwoStepRecipeManager
import sys

# Country code to country name mapping
COUNTRY_CODE_MAP = {
    "AT": "Austria",
    "DZ": "Algeria",
    "AR": "Argentina",
    "AU": "Australia",
    "BE": "Belgium",
    "BA": "Bosnia and Herzegovina",
    "BR": "Brazil",
    "BG": "Bulgaria",
    "CN": "China",
    "CZ": "Czech Republic",
    "EG": "Egypt",
    "FR": "France",
    "DE": "Germany",
    "HU": "Hungary",
    "JP": "Japan",
    "MX": "Mexico",
    "MA": "Morocco",
    "PL": "Poland",
    "RO": "Romania",
    "RU": "Russia",
    "SK": "Slovakia",
    "ZA": "South Africa",
    "KR": "South Korea",
    "TR": "Turkey",
    "UA": "Ukraine",
    "US": "United States"
}

# Add the current directory to Python path to import local modules
# Backend root (for feature_analyzer.py)
sys.path.insert(0, '/usr/src/app')
sys.path.insert(0, '/usr/src/app/src')  # For two_step_recipe_search.py
sys.path.insert(0, '/usr/src/app/data')  # For feature_mapping_generator.py

# Now import everything else (AFTER sys.path setup)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vector_index_init')


def get_country_code(filename: str) -> str:
    """
    Extracts the 2-letter country code from filenames like:
    '000000000000375392_AT10_02_P.json'
    '000000000000096221_DE10_01_P.json'
    '000000000000382151_US60_08_P.json'

    If no valid pattern is found, returns 'Other'.
    """
    # Look for: underscore + 2 uppercase letters + 2 digits + underscore
    # e.g. '_AT10_', '_FR20_', '_US60_', '_CN10_', etc.
    match = re.search(r'_([A-Z]{2})\d{2}_', filename)
    if match:
        return match.group(1)
    return "Other"


def get_country_name(filename: str) -> str:
    """
    Extracts the country name from the filename using the country code.
    Returns 'Other' if the country code is not found in the mapping.
    """
    country_code = get_country_code(filename)
    return COUNTRY_CODE_MAP.get(country_code, "Other")


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
        elif 'Classification' in data.keys() and data['Classification'] and 'valueschar' in data['Classification'] and data['Classification']['valueschar']:
            result = pd.DataFrame(data['Classification']['valueschar'])
        else:
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


def create_qdrant_collection(qdrant_client, collection_name, text_vector_size=384, feature_vector_size=484):
    """Create Qdrant collection with named vectors for text and features"""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name in collection_names:
            logger.info(
                f"Collection '{collection_name}' already exists.")
            return True

        # Create collection with named vectors configuration
        # text_vector: for Step 1 text search (384 dim)
        # feature_vector: for Step 2 feature refinement (484 dim = 384 text + 100 categorical)
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text": VectorParams(
                    size=text_vector_size,
                    distance=Distance.COSINE
                ),
                "features": VectorParams(
                    size=feature_vector_size,
                    distance=Distance.COSINE
                )
            }
        )
        logger.info(
            f"Created collection '{collection_name}' with named vectors:")
        logger.info(f"  - text: {text_vector_size} dimensions")
        logger.info(f"  - features: {feature_vector_size} dimensions")
        return True

    except Exception as e:
        logger.error(f"Error creating Qdrant collection: {e}")
        return False


def pre_analyze_features(feature_map_path: str) -> dict:
    """
    Pre-analyze features using the charactDescr_valueCharLong_map.json

    Returns:
        Dictionary with feature analysis results
    """
    if not os.path.exists(feature_map_path):
        logger.warning(f"Feature map not found at {feature_map_path}")
        logger.warning(
            "Proceeding without pre-analysis (will analyze during indexing)")
        return None

    try:
        logger.info("=" * 60)
        logger.info("PRE-ANALYZING FEATURES")
        logger.info("=" * 60)
        logger.info(f"Loading feature map from: {feature_map_path}")

        # Create analyzer
        analyzer = FeatureAnalyzer(feature_map_path)

        # Run analysis
        analysis = analyzer.analyze_all_features()

        # Print summary
        analyzer.print_summary()

        # Get simplified config for indexing
        feature_config = analyzer.get_feature_config_for_indexing()

        logger.info("=" * 60)
        logger.info(f"✅ Pre-analyzed {len(feature_config)} features")
        logger.info(f"   Binary: {len(analyzer.binary_features)}")
        logger.info(f"   Numerical: {len(analyzer.numerical_features)}")
        logger.info(f"   Range: {len(analyzer.range_features)}")
        logger.info(f"   Categorical: {len(analyzer.categorical_features)}")
        logger.info("=" * 60)

        return {
            'feature_config': feature_config,
            'binary_features': analyzer.binary_features,
            'numerical_features': analyzer.numerical_features,
            'range_features': analyzer.range_features,
            'categorical_features': analyzer.categorical_features
        }

    except Exception as e:
        logger.error(f"Error in feature pre-analysis: {e}")
        logger.warning("Proceeding without pre-analysis")
        return None


def index_recipes_to_qdrant(qdrant_client, embedding_model, collection_name, data_dir, feature_map_path=None):
    """Index recipes to Qdrant using EnhancedTwoStepRecipeManager logic with pre-analyzed features"""
    try:
        # Pre-analyze features if map is available
        feature_analysis = None
        if feature_map_path:
            feature_analysis = pre_analyze_features(feature_map_path)

        # Load recipe files
        recipe_json_list = load_recipes_from_data_dir(data_dir)

        if not recipe_json_list:
            logger.info("No recipes found to index. Skipping indexing step.")
            return True

        logger.info(
            f"Processing {len(recipe_json_list)} recipes with enhanced encoding...")

        # Initialize EnhancedTwoStepRecipeManager for sophisticated feature encoding
        logger.info("Initializing EnhancedTwoStepRecipeManager...")
        manager = EnhancedTwoStepRecipeManager(
            collection_name=collection_name,
            embedding_model=embedding_model,
            max_features=200
        )

        # If we have pre-analyzed features, inject them into the manager
        if feature_analysis and 'feature_config' in feature_analysis:
            logger.info("Injecting pre-analyzed feature types into manager...")
            feature_config = feature_analysis['feature_config']

            # Inject binary features
            for feature_name, feature_type in feature_config.items():
                if feature_type == 'binary':
                    manager.feature_types[feature_name] = 'binary'
                elif feature_type == 'numerical':
                    manager.feature_types[feature_name] = 'numerical'
                elif feature_type == 'range':
                    # Treat ranges as numerical
                    manager.feature_types[feature_name] = 'numerical'

            logger.info(
                f"✅ Injected {len(feature_config)} pre-analyzed feature types")
            logger.info(
                f"   Binary features: {sum(1 for t in feature_config.values() if t == 'binary')}")
            logger.info(
                f"   Numerical features: {sum(1 for t in feature_config.values() if t in ['numerical', 'range'])}")
            logger.info(
                f"   Categorical features: {sum(1 for t in feature_config.values() if t == 'categorical')}")

        # Collect all recipes first
        features_list = []
        values_list = []
        descriptions_list = []
        recipe_ids = []
        countries_list = []

        logger.info("Step 1: Loading recipes from JSON files...")
        for i, recipe_path in enumerate(recipe_json_list):
            try:
                # Extract features and values
                recipe_data = read_recipe_json(recipe_path)

                # Extract description
                description = extract_recipe_description(recipe_path)

                # Extract country name from filename
                recipe_filename_full = os.path.basename(recipe_path)
                country_name = get_country_name(recipe_filename_full)

                if recipe_data is not None and isinstance(recipe_data, pd.DataFrame):
                    if 'charactDescr' in recipe_data.columns and 'valueCharLong' in recipe_data.columns:
                        recipe_temp = recipe_data[[
                            'charactDescr', 'valueCharLong']]
                        features = recipe_temp['charactDescr'].tolist()
                        values = recipe_temp['valueCharLong'].tolist()

                        recipe_filename = os.path.basename(
                            recipe_path).split('.')[0]

                        features_list.append(features)
                        values_list.append(values)
                        descriptions_list.append(description)
                        recipe_ids.append(recipe_filename)
                        countries_list.append(country_name)

                        if (i + 1) % 1000 == 0:
                            logger.info(
                                f"Loaded {i + 1}/{len(recipe_json_list)} recipes")
                    else:
                        logger.warning(
                            f"Missing required columns in {recipe_path}")
                else:
                    logger.warning(f"Skipping invalid recipe: {recipe_path}")
            except Exception as e:
                logger.error(f"Error loading recipe {recipe_path}: {e}")
                continue

        logger.info(f"Loaded {len(features_list)} valid recipes")

        # Step 2: Use EnhancedTwoStepRecipeManager to process recipes
        # This will analyze features, detect binary patterns, and create proper encodings
        logger.info(
            "Step 2: Processing recipes with EnhancedTwoStepRecipeManager...")
        success = manager.update_recipes(
            features_list=features_list,
            values_list=values_list,
            descriptions_list=descriptions_list,
            recipe_ids=recipe_ids
        )

        if not success:
            logger.error(
                "Failed to process recipes with EnhancedTwoStepRecipeManager")
            return False

        # Create a mapping of recipe_id to country for quick lookup
        recipe_country_map = {recipe_id: country for recipe_id,
                              country in zip(recipe_ids, countries_list)}

        # Log country distribution
        country_counts = {}
        for country in countries_list:
            country_counts[country] = country_counts.get(country, 0) + 1
        logger.info(f"Country distribution: {country_counts}")

        # Step 3: Upload to Qdrant with named vectors
        logger.info("Step 3: Uploading recipes to Qdrant with named vectors...")
        batch_size = 10
        points = []

        for i, recipe_data in enumerate(manager.recipes):
            try:
                recipe_id = recipe_data["id"]
                text_vector = recipe_data["text_vector"]
                feature_vector = recipe_data["feature_vector"]
                payload = recipe_data["payload"]

                # Get country for this recipe
                country = recipe_country_map.get(recipe_id, "Other")

                # Create point with named vectors
                # Use deterministic UUID to avoid duplicates on re-indexing
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, recipe_id))

                point = PointStruct(
                    id=point_id,
                    vector={
                        "text": text_vector.tolist(),
                        "features": feature_vector.tolist()
                    },
                    payload={
                        "recipe_name": payload.get("recipe_id", recipe_id),
                        "description": payload.get("description", ""),
                        # Limit for payload size
                        "features": payload.get("features", [])[:50],
                        "values": payload.get("values", [])[:50],
                        "num_features": len(payload.get("features", [])),
                        "feature_text": payload.get("feature_text", ""),
                        "country": country
                    }
                )
                points.append(point)

                # Upload in batches
                if len(points) >= batch_size:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    logger.info(
                        f"Uploaded batch: {i - len(points) + 2} to {i + 1}/{len(manager.recipes)}")
                    points = []

            except Exception as e:
                logger.error(
                    f"Error creating point for recipe {recipe_id}: {e}")
                continue

        # Upload remaining points
        if points:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Uploaded final batch of {len(points)} recipes")

        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name)
        logger.info(
            f"Successfully indexed {collection_info.points_count} recipes in Qdrant")

        # Log feature analysis
        stats = manager.get_stats()
        logger.info("=" * 60)
        logger.info("FEATURE ANALYSIS:")
        logger.info(f"  Total unique features: {stats['total_features']}")
        logger.info(
            f"  Binary features detected: {len(stats['feature_analysis']['binary_feature_names'])}")
        logger.info(
            f"  Numerical features: {len(stats['feature_analysis']['numerical_features'])}")
        logger.info(
            f"  Categorical features: {len(stats['feature_analysis']['categorical_features'])}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Error indexing recipes to Qdrant: {e}")
        logger.exception("Detailed error:")
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
            'RECIPE_DATA_DIR', '/usr/src/app/ai-analyzer/data')
        feature_map_path = os.getenv(
            'FEATURE_MAP_PATH', '/usr/src/app/Test_Input/charactDescr_valueCharLong_map.json')

        logger.info("=" * 60)
        logger.info("Starting Qdrant Vector Index Initialization (Enhanced)")
        logger.info("=" * 60)
        logger.info(f"Qdrant Host: {qdrant_host}")
        logger.info(f"Qdrant Port: {qdrant_port}")
        logger.info(f"Collection Name: {collection_name}")
        logger.info(f"Embedding Model: {embedding_model}")
        logger.info(f"Data Directory: {data_dir}")
        logger.info(f"Feature Map: {feature_map_path}")
        logger.info("=" * 60)

        # Wait for Qdrant to be ready
        logger.info("Waiting for Qdrant to be ready...")
        qdrant_client = wait_for_qdrant(qdrant_host, qdrant_port)

        # Create Qdrant collection with named vectors
        logger.info("Creating Qdrant collection with named vectors...")
        create_qdrant_collection(
            qdrant_client,
            collection_name,
            text_vector_size=384,      # Text embedding dimension
            feature_vector_size=484    # Feature vector dimension (384 + 100)
        )

        # Index recipes directly into Qdrant with pre-analyzed features
        logger.info(
            "Indexing recipes to Qdrant with enhanced feature encoding...")
        success = index_recipes_to_qdrant(
            qdrant_client, embedding_model, collection_name, data_dir, feature_map_path)

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
