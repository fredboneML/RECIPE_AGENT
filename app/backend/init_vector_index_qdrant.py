#!/usr/bin/env python3
"""
Vector Index Initialization with Qdrant Storage - Enhanced Version with TRUE BATCH PROCESSING

This version uses EnhancedTwoStepRecipeManager logic for sophisticated feature encoding
with pre-analyzed feature types from charactDescr_valueCharLong_map.json.

KEY IMPROVEMENT: Processes files in batches instead of loading all into memory first.
This allows indexing of 600K+ files without memory issues and with visible progress.

IMPORTANT: Uses EnhancedTwoStepRecipeManager to preserve binary opposition mapping
and all sophisticated feature detection for proper search functionality.
"""
from src.two_step_recipe_search import EnhancedTwoStepRecipeManager
from feature_analyzer import FeatureAnalyzer
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient
import os
import time
import logging
import json
import pandas as pd
import uuid
import re
from typing import List, Optional
import numpy as np
import sys

# Add the current directory to Python path to import local modules
# Backend root (for feature_analyzer.py)
sys.path.insert(0, '/usr/src/app')
sys.path.insert(0, '/usr/src/app/src')  # For two_step_recipe_search.py
sys.path.insert(0, '/usr/src/app/data')  # For feature_mapping_generator.py


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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vector_index_init')

# =============================================================================
# BATCH PROCESSING CONFIGURATION
# =============================================================================
FILE_BATCH_SIZE = 5000          # Number of files to process per batch
QDRANT_UPSERT_BATCH_SIZE = 100  # Batch size for Qdrant upserts
# Additional file sample for feature analysis (supplements the feature map)
# Set to 0 to only use the feature map, or a small number for extra coverage
FEATURE_ANALYSIS_FILE_SAMPLE = 10000


def get_country_code(filename: str) -> str:
    """
    Extracts the 2-letter country code from filenames like:
    '000000000000375392_AT10_02_P.json'
    """
    match = re.search(r'_([A-Z]{2})\d{2}_', filename)
    if match:
        return match.group(1)
    return "Other"


def get_country_name(filename: str) -> str:
    """Extracts the country name from the filename using the country code."""
    country_code = get_country_code(filename)
    return COUNTRY_CODE_MAP.get(country_code, "Other")


def wait_for_qdrant(qdrant_host="qdrant", qdrant_port=6333, max_retries=30, delay=2):
    """Wait for Qdrant to be ready"""
    for attempt in range(max_retries):
        try:
            client = QdrantClient(host=qdrant_host, port=qdrant_port)
            client.get_collections()  # Test connection
            logger.info("Qdrant is ready!")
            return client
        except Exception:
            if attempt < max_retries - 1:
                logger.info(
                    f"Qdrant not ready (attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(
                    f"Failed to connect to Qdrant after {max_retries} attempts")
                raise


def read_recipe_json(recipe_json_path: str) -> Optional[pd.DataFrame]:
    """Read recipe JSON file and extract features/values."""
    result = None
    try:
        with open(recipe_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            pass  # Empty file
        elif 'Classification' in data.keys() and data['Classification'] and 'valueschar' in data['Classification'] and data['Classification']['valueschar']:
            result = pd.DataFrame(data['Classification']['valueschar'])
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


def extract_recipe_description(recipe_json_path: str) -> str:
    """Extract comprehensive recipe description."""
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
            except (IndexError, KeyError, TypeError):
                pass

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
            except (IndexError, KeyError, TypeError):
                pass

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

                except Exception:
                    pass

        if description_parts:
            return ", ".join(description_parts)
        else:
            base_name = os.path.basename(recipe_json_path).replace(
                '.json', '').replace('_', ' ').replace('-', ' ')
            return f"Recipe {base_name}"

    except Exception as e:
        logger.error(f"Error extracting description: {e}")
        base_name = os.path.basename(recipe_json_path).replace(
            '.json', '').replace('_', ' ').replace('-', ' ')
        return f"Recipe {base_name}"


def load_recipes_from_data_dir(data_dir: str) -> List[str]:
    """Load all recipe JSON file paths from data directory"""
    recipe_json_list: List[str] = []

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


def pre_analyze_features(feature_map_path: str) -> Optional[dict]:
    """Pre-analyze features using the charactDescr_valueCharLong_map.json"""
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

        analyzer = FeatureAnalyzer(feature_map_path)
        analyzer.analyze_all_features()
        analyzer.print_summary()
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


def load_feature_map_for_analysis(feature_map_path: str) -> tuple:
    """
    Load the charactDescr_valueCharLong_map.json and convert it to features/values lists
    for binary opposition analysis.

    This is much faster than reading individual files since the map already contains
    all unique feature-value combinations from the entire dataset.

    Returns:
        Tuple of (features_list, values_list) in the format expected by _analyze_feature_values
    """
    features_list: List[List[str]] = []
    values_list: List[List[str]] = []

    if not os.path.exists(feature_map_path):
        logger.warning(f"Feature map not found at {feature_map_path}")
        return features_list, values_list

    try:
        logger.info("Loading feature map for binary opposition analysis...")
        with open(feature_map_path, 'r', encoding='utf-8') as f:
            feature_map = json.load(f)

        # Convert map to features/values format
        # Each feature with its values becomes a "virtual recipe"
        for feature_name, values in feature_map.items():
            if values:
                # Create a virtual recipe with this feature and all its possible values
                for value in values:
                    features_list.append([feature_name])
                    values_list.append([value])

        logger.info(
            f"  Loaded {len(feature_map)} features with {len(features_list)} feature-value pairs from map")
        return features_list, values_list

    except Exception as e:
        logger.error(f"Error loading feature map: {e}")
        return [], []


def analyze_features_for_binary_patterns(
    recipe_json_list: List[str],
    manager: EnhancedTwoStepRecipeManager,
    feature_map_path: Optional[str] = None,
    file_sample_size: int = 0
) -> None:
    """
    Analyze feature patterns to build binary opposition mappings.

    Uses a combination of:
    1. The charactDescr_valueCharLong_map.json (contains ALL unique feature-value pairs)
    2. Optional: Sample of actual recipe files for additional coverage

    This approach captures more patterns without reading all 600K files.

    Args:
        recipe_json_list: List of recipe file paths
        manager: The EnhancedTwoStepRecipeManager instance
        feature_map_path: Path to charactDescr_valueCharLong_map.json
        file_sample_size: Additional files to sample (0 = no additional files)
    """
    features_list = []
    values_list = []

    # Step 1: Load from feature map (comprehensive, contains all unique values)
    if feature_map_path:
        map_features, map_values = load_feature_map_for_analysis(
            feature_map_path)
        features_list.extend(map_features)
        values_list.extend(map_values)
        logger.info(f"  Added {len(map_features)} entries from feature map")

    # Step 2: Optionally sample additional files (for extra context/combinations)
    if file_sample_size > 0 and recipe_json_list:
        import random

        # Take random sample (better distribution than evenly spaced)
        sample_files = recipe_json_list
        if len(recipe_json_list) > file_sample_size:
            sample_files = random.sample(recipe_json_list, file_sample_size)

        logger.info(f"  Sampling {len(sample_files):,} additional files...")

        file_entries = 0
        for i, recipe_path in enumerate(sample_files):
            try:
                recipe_data = read_recipe_json(recipe_path)

                if recipe_data is not None and isinstance(recipe_data, pd.DataFrame):
                    if 'charactDescr' in recipe_data.columns and 'valueCharLong' in recipe_data.columns:
                        features = recipe_data['charactDescr'].tolist()
                        values = recipe_data['valueCharLong'].tolist()
                        features_list.append(features)
                        values_list.append(values)
                        file_entries += 1

                if (i + 1) % 5000 == 0:
                    logger.info(
                        f"    Processed {i + 1:,}/{len(sample_files):,} sample files...")

            except Exception:
                continue

        logger.info(f"  Added {file_entries} entries from file samples")

    # Step 3: Run the manager's feature analysis on combined data
    total_entries = len(features_list)
    logger.info(
        f"Running binary opposition detection on {total_entries:,} total entries...")
    manager._analyze_feature_values(features_list, values_list)

    logger.info("✅ Feature analysis complete:")
    logger.info(f"   Detected {len(manager.feature_types)} feature types")
    logger.info(
        f"   Binary features with oppositions: {len(manager.binary_features)}")


def index_recipes_to_qdrant_batched(
    qdrant_client: QdrantClient,
    embedding_model_name: str,
    collection_name: str,
    data_dir: str,
    feature_map_path: Optional[str] = None
) -> bool:
    """
    Index recipes to Qdrant using TRUE BATCH PROCESSING with EnhancedTwoStepRecipeManager.

    This preserves all sophisticated feature encoding including:
    - Binary opposition mapping
    - Numerical feature detection
    - Categorical encoding

    Processes files in chunks to:
    1. Avoid loading all files into memory
    2. Show progress as recipes are indexed
    3. Use EnhancedTwoStepRecipeManager for proper encoding
    """
    try:
        # Pre-analyze features from feature map
        feature_analysis = None
        if feature_map_path:
            feature_analysis = pre_analyze_features(feature_map_path)

        # Get list of all recipe files
        recipe_json_list = load_recipes_from_data_dir(data_dir)

        if not recipe_json_list:
            logger.info("No recipes found to index. Skipping indexing step.")
            return True

        total_files = len(recipe_json_list)
        total_batches = (total_files + FILE_BATCH_SIZE - 1) // FILE_BATCH_SIZE

        # Initialize EnhancedTwoStepRecipeManager
        logger.info("Initializing EnhancedTwoStepRecipeManager...")
        manager = EnhancedTwoStepRecipeManager(
            collection_name=collection_name,
            embedding_model=embedding_model_name,
            max_features=200
        )

        # Inject pre-analyzed feature types
        if feature_analysis and 'feature_config' in feature_analysis:
            logger.info("Injecting pre-analyzed feature types into manager...")
            feature_config = feature_analysis['feature_config']

            for feature_name, feature_type in feature_config.items():
                if feature_type == 'binary':
                    manager.feature_types[feature_name] = 'binary'
                elif feature_type == 'numerical':
                    manager.feature_types[feature_name] = 'numerical'
                elif feature_type == 'range':
                    manager.feature_types[feature_name] = 'numerical'

            logger.info(
                f"✅ Injected {len(feature_config)} pre-analyzed feature types")
            logger.info(
                f"   Binary features: {sum(1 for t in feature_config.values() if t == 'binary')}")
            logger.info(
                f"   Numerical features: {sum(1 for t in feature_config.values() if t in ['numerical', 'range'])}")
            logger.info(
                f"   Categorical features: {sum(1 for t in feature_config.values() if t == 'categorical')}")

        # Analyze features to build binary opposition mappings
        # Uses feature map + optional file samples for comprehensive coverage
        logger.info("=" * 60)
        logger.info("ANALYZING FEATURE PATTERNS FOR BINARY OPPOSITION MAPPING")
        logger.info("=" * 60)
        analyze_features_for_binary_patterns(
            recipe_json_list,
            manager,
            feature_map_path=feature_map_path,
            file_sample_size=FEATURE_ANALYSIS_FILE_SAMPLE
        )

        logger.info("\n" + "=" * 60)
        logger.info("STARTING BATCH INDEXING")
        logger.info("=" * 60)
        logger.info(f"Total files to process: {total_files:,}")
        logger.info(f"Batch size: {FILE_BATCH_SIZE:,} files")
        logger.info(f"Total batches: {total_batches}")
        logger.info("=" * 60)

        total_indexed = 0
        total_skipped = 0
        start_time = time.time()

        # Process in batches
        for batch_idx in range(total_batches):
            batch_start = batch_idx * FILE_BATCH_SIZE
            batch_end = min(batch_start + FILE_BATCH_SIZE, total_files)
            batch_files = recipe_json_list[batch_start:batch_end]

            batch_start_time = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {batch_idx + 1}/{total_batches}")
            logger.info(
                f"Processing files {batch_start + 1:,} to {batch_end:,}")
            logger.info(f"{'='*60}")

            # Step 1: Read batch of files and extract data
            logger.info("Step 1: Reading JSON files...")
            batch_data = []

            for recipe_path in batch_files:
                try:
                    recipe_data = read_recipe_json(recipe_path)
                    description = extract_recipe_description(recipe_path)
                    filename = os.path.basename(recipe_path)
                    country_name = get_country_name(filename)
                    recipe_id = filename.split('.')[0]

                    if recipe_data is not None and isinstance(recipe_data, pd.DataFrame):
                        if 'charactDescr' in recipe_data.columns and 'valueCharLong' in recipe_data.columns:
                            features = recipe_data['charactDescr'].tolist()
                            values = recipe_data['valueCharLong'].tolist()

                            batch_data.append({
                                'recipe_id': recipe_id,
                                'features': features,
                                'values': values,
                                'description': description,
                                'country': country_name
                            })
                except Exception as e:
                    logger.warning(f"Error reading {recipe_path}: {e}")
                    continue

            if not batch_data:
                logger.warning(f"No valid recipes in batch {batch_idx + 1}")
                continue

            logger.info(f"  Loaded {len(batch_data)} valid recipes from batch")

            # Step 2: Create embeddings using EnhancedTwoStepRecipeManager
            logger.info(
                "Step 2: Creating embeddings with EnhancedTwoStepRecipeManager...")

            # Extract lists for batch processing
            descriptions = [r['description'] for r in batch_data]
            features_list = [r['features'] for r in batch_data]
            values_list = [r['values'] for r in batch_data]

            # Create text embeddings in batch (more efficient)
            logger.info("  Creating text embeddings (batched)...")
            text_embeddings = manager.embedding_model.encode(
                descriptions,
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            # Create feature vectors (uses manager's sophisticated encoding)
            logger.info("  Creating feature vectors...")
            feature_vector_list = []
            for features, values in zip(features_list, values_list):
                # Truncate to same length
                min_length = min(len(features), len(values))
                features = features[:min_length]
                values = values[:min_length]

                # Use manager's feature vector creation (preserves binary opposition logic)
                feature_vector = manager._create_feature_vector(
                    features, values, fit=True)
                feature_vector_list.append(feature_vector)

            feature_vectors = np.array(feature_vector_list)

            # Step 3: Upload to Qdrant
            logger.info("Step 3: Uploading to Qdrant...")
            points = []

            for i, recipe in enumerate(batch_data):
                try:
                    recipe_id = recipe['recipe_id']
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, recipe_id))

                    # Create feature text for payload
                    feature_text = manager._create_feature_text(
                        recipe['features'], recipe['values'])

                    # Clean values for payload
                    clean_values = [manager._clean_value(
                        v) for v in recipe['values'][:50]]

                    point = PointStruct(
                        id=point_id,
                        vector={
                            "text": text_embeddings[i].tolist(),
                            "features": feature_vectors[i].tolist()
                        },
                        payload={
                            "recipe_name": recipe_id,
                            "description": recipe['description'],
                            # Limit for payload size
                            "features": recipe['features'][:50],
                            "values": clean_values,
                            "num_features": len(recipe['features']),
                            # Limit text length
                            "feature_text": feature_text[:1000],
                            "country": recipe['country']
                        }
                    )
                    points.append(point)

                    # Upload in sub-batches
                    if len(points) >= QDRANT_UPSERT_BATCH_SIZE:
                        qdrant_client.upsert(
                            collection_name=collection_name,
                            points=points
                        )
                        total_indexed += len(points)
                        points = []

                except Exception as e:
                    logger.warning(
                        f"Error creating point for {recipe_id}: {e}")
                    total_skipped += 1
                    continue

            # Upload remaining points
            if points:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                total_indexed += len(points)

            # Batch statistics
            batch_time = time.time() - batch_start_time
            elapsed_total = time.time() - start_time

            # Get current collection count
            collection_info = qdrant_client.get_collection(collection_name)

            logger.info(f"\n  ✅ Batch {batch_idx + 1} complete!")
            logger.info(f"  Batch time: {batch_time:.1f}s")
            logger.info(f"  Total indexed so far: {total_indexed:,}")
            logger.info(
                f"  Qdrant collection size: {collection_info.points_count:,}")
            logger.info(f"  Total elapsed time: {elapsed_total:.1f}s")

            # Estimate remaining time
            if batch_idx + 1 < total_batches:
                avg_batch_time = elapsed_total / (batch_idx + 1)
                remaining_batches = total_batches - (batch_idx + 1)
                estimated_remaining = avg_batch_time * remaining_batches
                logger.info(
                    f"  Estimated time remaining: {estimated_remaining/60:.1f} minutes")

        # Final summary
        total_time = time.time() - start_time
        collection_info = qdrant_client.get_collection(collection_name)

        # Log feature analysis from manager
        stats = manager.get_stats()

        logger.info("\n" + "=" * 60)
        logger.info("INDEXING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Total recipes indexed: {total_indexed:,}")
        logger.info(f"Total skipped: {total_skipped:,}")
        logger.info(
            f"Qdrant collection size: {collection_info.points_count:,}")
        logger.info(
            f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        if total_time > 0:
            logger.info(
                f"Average speed: {total_indexed/total_time:.1f} recipes/second")

        logger.info("\n" + "=" * 60)
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
        logger.error(f"Error in batch indexing: {e}")
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
        logger.info("Starting Qdrant Vector Index Initialization")
        logger.info("TRUE BATCH PROCESSING with EnhancedTwoStepRecipeManager")
        logger.info("=" * 60)
        logger.info(f"Qdrant Host: {qdrant_host}")
        logger.info(f"Qdrant Port: {qdrant_port}")
        logger.info(f"Collection Name: {collection_name}")
        logger.info(f"Embedding Model: {embedding_model}")
        logger.info(f"Data Directory: {data_dir}")
        logger.info(f"Feature Map: {feature_map_path}")
        logger.info(f"File Batch Size: {FILE_BATCH_SIZE:,}")
        logger.info(
            f"Additional File Sample for Analysis: {FEATURE_ANALYSIS_FILE_SAMPLE:,}")
        logger.info(f"Qdrant Upsert Batch Size: {QDRANT_UPSERT_BATCH_SIZE}")
        logger.info("=" * 60)

        # Wait for Qdrant to be ready
        logger.info("Waiting for Qdrant to be ready...")
        qdrant_client = wait_for_qdrant(qdrant_host, qdrant_port)

        # Create Qdrant collection with named vectors
        logger.info("Creating Qdrant collection with named vectors...")
        create_qdrant_collection(
            qdrant_client,
            collection_name,
            text_vector_size=384,
            feature_vector_size=484
        )

        # Index recipes using TRUE BATCH PROCESSING with EnhancedTwoStepRecipeManager
        logger.info(
            "Starting batch indexing with EnhancedTwoStepRecipeManager...")
        success = index_recipes_to_qdrant_batched(
            qdrant_client,
            embedding_model,
            collection_name,
            data_dir,
            feature_map_path
        )

        if success:
            logger.info("=" * 60)
            logger.info(
                "✅ Qdrant Vector Index Initialization Completed Successfully!")
            logger.info("=" * 60)
        else:
            logger.warning("=" * 60)
            logger.warning(
                "⚠️ Qdrant Vector Index Initialization Completed with Warnings")
            logger.warning("=" * 60)

    except Exception as e:
        logger.error(f"Error during vector index initialization: {e}")
        logger.exception("Detailed error:")
        raise


if __name__ == "__main__":
    main()
