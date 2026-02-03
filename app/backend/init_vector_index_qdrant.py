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
from feature_normalizer import FeatureNormalizer
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient
import os
import time
import logging
import json
import pandas as pd
import uuid
import re
from typing import List, Optional, Union, Literal
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

# =============================================================================
# 60 SPECIFIED FIELDS FOR INDEXING
# =============================================================================
# These 60 fields are the standardized fields for recipe indexing.
# Field names are always consistent (Z_xxx), but values can be in multiple languages.
# Each field has: code, sort_order, description_en, description_de, field_type
SPECIFIED_FIELDS = {
    'Z_MAKTX':       {'order': 1,  'en': 'Material short text',           'de': 'Materialkurztext',               'type': 'text'},
    'Z_INH01':       {'order': 2,  'en': 'Standard product',              'de': 'Standardprodukt',                'type': 'binary'},
    'Z_WEIM':        {'order': 3,  'en': 'Produktsegment (SD Reporting)', 'de': 'Produktsegment (SD Reporting)',  'type': 'categorical'},
    'Z_KUNPROGRU':   {'order': 4,  'en': 'Customer product group',        'de': 'Kundenproduktgruppe',            'type': 'categorical'},
    'Z_PRODK':       {'order': 5,  'en': 'Market segments',               'de': 'Produktkategorien',              'type': 'categorical'},
    'Z_INH07':       {'order': 6,  'en': 'Extreme recipe',                'de': 'Extremrezeptur',                 'type': 'binary'},
    'Z_KOCHART':     {'order': 7,  'en': 'Pasteurization type',           'de': 'Kochart',                        'type': 'categorical'},
    'Z_KNOGM':       {'order': 8,  'en': 'GMO presence',                  'de': 'GMO enthalten',                  'type': 'binary'},
    'Z_INH08':       {'order': 9,  'en': 'Contains GMO',                  'de': 'Nicht Genfrei',                  'type': 'binary'},
    'Z_INH12':       {'order': 10, 'en': 'Allergen-free',                  'de': 'Allergenfrei',                   'type': 'binary'},
    'ZMX_TIPOALERG': {'order': 11, 'en': 'Alergenic type',                'de': 'Allergentyp',                    'type': 'categorical'},
    'Z_INH02':       {'order': 12, 'en': 'Sweetener',                     'de': 'Süßstoff',                       'type': 'binary'},
    'Z_INH03':       {'order': 13, 'en': 'Saccharose',                    'de': 'Saccharose',                     'type': 'binary'},
    'Z_INH19':       {'order': 14, 'en': 'Aspartame',                     'de': 'Aspartam',                       'type': 'binary'},
    'Z_INH04':       {'order': 15, 'en': 'Preserved',                     'de': 'Konservierung',                  'type': 'binary'},
    'Z_INH18':       {'order': 16, 'en': 'Color',                         'de': 'Farbe',                          'type': 'binary'},
    'Z_INH05':       {'order': 17, 'en': 'Artificial colors',             'de': 'Künstliche Farben',              'type': 'binary'},
    'Z_INH09':       {'order': 18, 'en': 'Flavour',                       'de': 'Aroma',                          'type': 'binary'},
    'Z_INH06':       {'order': 19, 'en': 'Nature identical flavor',       'de': 'naturident/künstliches Aroma',   'type': 'binary'},
    'Z_INH06Z':      {'order': 20, 'en': 'Natural flavor',                'de': 'Natürliche Aromen',              'type': 'binary'},
    'Z_FSTAT':       {'order': 21, 'en': 'Flavor status',                 'de': 'Flavor status',                  'type': 'categorical'},
    'Z_INH21':       {'order': 22, 'en': 'Vitamins',                      'de': 'Vitamine',                       'type': 'binary'},
    'Z_INH13':       {'order': 23, 'en': 'Starch',                        'de': 'Stärke',                         'type': 'binary'},
    'Z_INH14':       {'order': 24, 'en': 'Pectin',                        'de': 'Pektin',                         'type': 'binary'},
    'Z_INH15':       {'order': 25, 'en': 'LBG',                           'de': 'IBKM',                           'type': 'binary'},
    'Z_INH16':       {'order': 26, 'en': 'Blend',                         'de': 'Mischung',                       'type': 'binary'},
    'Z_INH20':       {'order': 27, 'en': 'Xanthan',                       'de': 'Xanthan',                        'type': 'binary'},
    'Z_STABGU':      {'order': 28, 'en': 'Stabilizing System - Guar',     'de': 'Stabilizing System - Guar',      'type': 'binary'},
    'Z_STABCAR':     {'order': 29, 'en': 'Stabilizing System - Carrageen','de': 'Stabilizing System - Carrageen', 'type': 'binary'},
    'Z_STAGEL':      {'order': 30, 'en': 'Stabilizing System - Gellan',   'de': 'Stabilizing System - Gellan',    'type': 'binary'},
    'Z_STANO':       {'order': 31, 'en': 'Stabilizing System - No stabil','de': 'Stabilizing System - No stabil', 'type': 'binary'},
    'Z_INH17':       {'order': 32, 'en': 'Other stabilizer',              'de': 'Andere Stabilisatoren',          'type': 'binary'},
    'Z_BRIX':        {'order': 33, 'en': 'Brix',                          'de': 'Brix',                           'type': 'numerical'},
    'Z_PH':          {'order': 34, 'en': 'pH',                            'de': 'PH',                             'type': 'numerical'},
    'ZM_PH':         {'order': 35, 'en': 'PH AFM',                        'de': 'PH AFM',                         'type': 'numerical'},
    'Z_VISK20S':     {'order': 36, 'en': 'Viscosity 20s (20°C)',          'de': 'Viskosität 20s (20°C)',          'type': 'numerical'},
    'Z_VISK20S_7C':  {'order': 37, 'en': 'Viscosity 20s (7°C)',           'de': 'Viskosität 20s (7°C)',           'type': 'numerical'},
    'Z_VISK30S':     {'order': 38, 'en': 'Viscosity 30s',                 'de': 'Viskosität 30s',                 'type': 'numerical'},
    'Z_VISK60S':     {'order': 39, 'en': 'Viscosity 60s',                 'de': 'Viskosität 60s',                 'type': 'numerical'},
    'Z_VISKHAAKE':   {'order': 40, 'en': 'Viscosity HAAKE',               'de': 'Viskosität HAAKE',               'type': 'numerical'},
    'ZMX_DD103':     {'order': 41, 'en': 'Haake Viscosity',               'de': 'Haake Viskositaet',              'type': 'numerical'},
    'ZMX_DD102':     {'order': 42, 'en': 'Brookfield Viscosity',          'de': 'Brookfield Viskositaet',         'type': 'numerical'},
    'ZM_AW':         {'order': 43, 'en': 'Water Activity AFM',            'de': 'Wasseraktivität AFM',            'type': 'numerical'},
    'Z_FGAW':        {'order': 44, 'en': 'Water activity (FruitPrep)[aW]','de': 'Water activity (FruitPrep)[aW]', 'type': 'numerical'},
    'Z_FRUCHTG':     {'order': 45, 'en': 'Fruit content',                 'de': 'Fruchtgehalt',                   'type': 'numerical'},
    'ZMX_DD108':     {'order': 46, 'en': 'Fruit Content',                 'de': 'Fruchtgehalt',                   'type': 'numerical'},
    'Z_AW':          {'order': 47, 'en': 'Fruit retention in %',          'de': 'Auswaschung %',                  'type': 'numerical'},
    'Z_FLST':        {'order': 48, 'en': 'Puree/with pieces',             'de': 'Flüssig/Stückig',                'type': 'categorical'},
    'Z_PP':          {'order': 49, 'en': 'Puree/Pieces',                  'de': 'Puree/Pieces',                   'type': 'categorical'},
    'ZMX_DD109':     {'order': 50, 'en': '% Fruit Identity',              'de': '% Dosierung',                    'type': 'numerical'},
    'Z_DOSIER':      {'order': 51, 'en': 'Dosage',                        'de': 'Dosierung',                      'type': 'numerical'},
    'Z_ZUCKER':      {'order': 52, 'en': 'Sugar',                         'de': 'Zucker',                         'type': 'numerical'},
    'Z_FETTST':      {'order': 53, 'en': 'Fat level',                     'de': 'Fettstufe',                      'type': 'numerical'},
    'ZMX_DD104':     {'order': 54, 'en': 'White Mass type',               'de': 'Weisse Masse typ',               'type': 'categorical'},
    'Z_PROT':        {'order': 55, 'en': 'Protein content(white mass)[%]','de': 'Protein content(white mass)[%]', 'type': 'numerical'},
    'Z_SALZ':        {'order': 56, 'en': 'Salt',                          'de': 'Salz',                           'type': 'numerical'},
    'Z_INH01K':      {'order': 57, 'en': 'Kosher',                        'de': 'Kosher',                         'type': 'binary'},
    'Z_INH01H':      {'order': 58, 'en': 'Halal',                         'de': 'Halal',                          'type': 'binary'},
    'Z_DAIRY':       {'order': 59, 'en': 'Non-Dairy Product',             'de': 'Non-Dairy Product',              'type': 'binary'},
    'Z_BFS':         {'order': 60, 'en': 'Bake/Freeze Stability',         'de': 'Bake/Freeze Stability',          'type': 'binary'},
}

# Ordered list of field codes for consistent processing
SPECIFIED_FIELDS_ORDERED = [
    'Z_MAKTX', 'Z_INH01', 'Z_WEIM', 'Z_KUNPROGRU', 'Z_PRODK', 'Z_INH07', 'Z_KOCHART', 'Z_KNOGM',
    'Z_INH08', 'Z_INH12', 'ZMX_TIPOALERG', 'Z_INH02', 'Z_INH03', 'Z_INH19', 'Z_INH04', 'Z_INH18',
    'Z_INH05', 'Z_INH09', 'Z_INH06', 'Z_INH06Z', 'Z_FSTAT', 'Z_INH21', 'Z_INH13', 'Z_INH14',
    'Z_INH15', 'Z_INH16', 'Z_INH20', 'Z_STABGU', 'Z_STABCAR', 'Z_STAGEL', 'Z_STANO', 'Z_INH17',
    'Z_BRIX', 'Z_PH', 'ZM_PH', 'Z_VISK20S', 'Z_VISK20S_7C', 'Z_VISK30S', 'Z_VISK60S', 'Z_VISKHAAKE',
    'ZMX_DD103', 'ZMX_DD102', 'ZM_AW', 'Z_FGAW', 'Z_FRUCHTG', 'ZMX_DD108', 'Z_AW', 'Z_FLST',
    'Z_PP', 'ZMX_DD109', 'Z_DOSIER', 'Z_ZUCKER', 'Z_FETTST', 'ZMX_DD104', 'Z_PROT', 'Z_SALZ',
    'Z_INH01K', 'Z_INH01H', 'Z_DAIRY', 'Z_BFS'
]

# Field types for encoding
BINARY_FIELDS = [f for f, info in SPECIFIED_FIELDS.items() if info['type'] == 'binary']
NUMERICAL_FIELDS = [f for f, info in SPECIFIED_FIELDS.items() if info['type'] == 'numerical']
CATEGORICAL_FIELDS = [f for f, info in SPECIFIED_FIELDS.items() if info['type'] == 'categorical']
TEXT_FIELDS = [f for f, info in SPECIFIED_FIELDS.items() if info['type'] == 'text']

# Binary value normalization patterns (multilingual)
BINARY_POSITIVE_PATTERNS = [
    # German positive
    'enthalten', 'mit ', 'vorhanden', 'ja', 'erlaubt', 'aktiv',
    # English positive  
    'yes', 'true', 'with', 'present', 'allowed', 'active', 'contains',
    # French positive
    'oui', 'avec', 'présent', 'autorisé',
    # Italian positive
    'si', 'con', 'presente',
    # Specific positive values (from sample data)
    'standardprodukt', 'standard product',
    'stärke enthalten', 'starch', 'pectin', 'pektin',
    'mit aroma', 'natürliches aroma', 'natural flavor', 'natural flavour',
    'allergenfrei',  # Note: This means "allergen-free" = No allergens = actually indicates YES for allergen-free status
    'halal', 'kosher', 'koscher',
]

BINARY_NEGATIVE_PATTERNS = [
    # German negative
    'kein', 'keine', 'nicht', 'ohne', 'verboten', 'nein',
    # English negative
    'no', 'not', 'without', 'absent', 'forbidden', 'false', 'none',
    # French negative
    'non', 'sans', 'pas', 'aucun', 'interdit',
    # Italian negative
    'senza', 'nessuno',
    # Specific negative values (from sample data)
    'keine süsstoffe', 'no sweetener',
    'keine künstl. farbe', 'no artificial colors',
    'kein aspartam', 'no aspartame',
    'nicht konserviert', 'no preservative',
    'kein naturidentes aroma', 'no nature identical',
    'kein pektin', 'no pectin',
    'kein ibkm', 'no lbg',
    'kein xanthan', 'no xanthan',
    'keine mischung', 'no blend',
    'keine anderen stabil', 'no other stabilizer',
    'keine farbe enthalten', 'no color',
    'keine extremrezeptur', 'no extreme recipe',
    'nicht genfrei',  # Contains GMO (double negative)
]

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


def extract_stlan(recipe_data: Union[dict, str]) -> Literal['L', 'P', 'Missing']:
    """
    Extract stlan value from a recipe JSON.

    The stlan field indicates the recipe version:
    - 'P': Production version (has all maintained data, prioritized when both P and L exist)
    - 'L': Lab/development version
    - 'Missing': stlan field not found (BillOfMaterialSTB may be missing)

    Business Logic:
    - When both P and L versions exist, P is returned (P has all data maintained)
    - When only L exists, L is returned
    - When stlan is missing or BillOfMaterialSTB is missing, 'Missing' is returned

    Args:
        recipe_data: Either a dictionary (parsed JSON) or a file path to a JSON file

    Returns:
        'P' if stlan field exists with value 'P' (prioritized even if L also exists)
        'L' if stlan field exists with value 'L' (and no P found)
        'Missing' if stlan doesn't exist or BillOfMaterialSTB is missing
    """
    # Load JSON if file path is provided
    if isinstance(recipe_data, str):
        try:
            with open(recipe_data, 'r', encoding='utf-8') as f:
                recipe_data = json.load(f)
        except Exception as e:
            logger.warning(
                f"Error loading file for stlan extraction {recipe_data}: {e}")
            return 'Missing'

    # Recursively search for stlan field
    # We'll collect all stlan values and prioritize: P > L > Missing
    # P is prioritized because it contains all maintained data
    found_values = set()

    def find_stlan(obj):
        """Recursively search for stlan field in nested structures."""
        if isinstance(obj, dict):
            # Check if stlan key exists
            if 'stlan' in obj:
                value = obj['stlan']
                if value == 'L' or value == 'P':
                    found_values.add(value)

            # Recursively search in all values
            for value in obj.values():
                find_stlan(value)

        elif isinstance(obj, list):
            # Recursively search in all list items
            for item in obj:
                find_stlan(item)

    find_stlan(recipe_data)

    # Return priority: P > L > Missing
    # P is prioritized because production versions have all data maintained
    if 'P' in found_values:
        return 'P'
    elif 'L' in found_values:
        return 'L'
    else:
        return 'Missing'


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


# =============================================================================
# 60 SPECIFIED FIELDS EXTRACTION
# =============================================================================

def extract_specified_fields(recipe_json_path: str) -> dict:
    """
    Extract the 60 specified fields from a recipe JSON file.
    
    Returns a dictionary with:
    - 'fields': Dict of all 60 fields with their values (None for missing)
    - 'available': List of field codes that have values
    - 'missing': List of field codes that are missing or have no value
    - 'field_types': Dict mapping field codes to their data types
    - 'features_for_embedding': List of (feature_name, value) tuples for embedding
    - 'numerical_values': Dict of numerical field values (for range queries)
    
    Args:
        recipe_json_path: Path to the recipe JSON file
        
    Returns:
        Dictionary with extracted field data
    """
    result = {
        'fields': {},           # All 60 fields with values (None for missing)
        'available': [],        # Field codes with values
        'missing': [],          # Field codes without values
        'field_types': {},      # Field code -> type mapping
        'features_for_embedding': [],  # For text embedding
        'numerical_values': {}, # Numerical values for range queries
        'original_values': {},  # Original values before normalization
    }
    
    try:
        with open(recipe_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Error loading JSON {recipe_json_path}: {e}")
        # Return empty result with all fields as missing
        for field_code in SPECIFIED_FIELDS_ORDERED:
            result['fields'][field_code] = None
            result['missing'].append(field_code)
            result['field_types'][field_code] = SPECIFIED_FIELDS[field_code]['type']
        return result
    
    # Build lookup dictionaries from Classification
    char_lookup = {}  # charact code -> item
    num_lookup = {}   # charact code -> item
    
    if 'Classification' in data and data['Classification']:
        classification = data['Classification']
        
        # Character fields
        if 'valueschar' in classification and classification['valueschar']:
            for item in classification['valueschar']:
                charact = item.get('charact', '')
                if charact:
                    char_lookup[charact] = item
        
        # Numeric fields
        if 'valuesnum' in classification and classification['valuesnum']:
            for item in classification['valuesnum']:
                charact = item.get('charact', '')
                if charact:
                    num_lookup[charact] = item
    
    # Extract Z_MAKTX from MaterialMasterShorttext
    maktx_value = None
    if 'MaterialMasterShorttext' in data and data['MaterialMasterShorttext']:
        # Try English first, then any available
        mms_list = data['MaterialMasterShorttext']
        maktx_entry = next((item for item in mms_list if item.get('spras') == 'E'), None)
        if not maktx_entry and mms_list:
            maktx_entry = mms_list[0]
        if maktx_entry:
            maktx_value = maktx_entry.get('maktx', '')
    
    # Process each of the 60 specified fields
    for field_code in SPECIFIED_FIELDS_ORDERED:
        field_info = SPECIFIED_FIELDS[field_code]
        field_type = field_info['type']
        result['field_types'][field_code] = field_type
        
        value = None
        original_value = None
        normalized_value = None
        
        # Special handling for Z_MAKTX
        if field_code == 'Z_MAKTX':
            if maktx_value and str(maktx_value).strip():
                value = str(maktx_value).strip()
                original_value = value
                normalized_value = value
        
        # Check character fields
        elif field_code in char_lookup:
            item = char_lookup[field_code]
            raw_value = item.get('valueCharLong', item.get('valueChar', ''))
            if raw_value and str(raw_value).strip():
                original_value = str(raw_value).strip()
                # Normalize for binary fields
                if field_type == 'binary':
                    normalized_value = normalize_binary_value(original_value)
                else:
                    normalized_value = original_value
                value = normalized_value
        
        # Check numeric fields
        elif field_code in num_lookup:
            item = num_lookup[field_code]
            value_from = item.get('valueFrom')
            unit = item.get('unitFrom', '')
            
            if value_from is not None:
                try:
                    numeric_val = float(value_from)
                    result['numerical_values'][field_code] = numeric_val
                    
                    # Format with unit for display
                    if unit:
                        value = f"{value_from} {unit}".strip()
                    else:
                        value = str(value_from)
                    original_value = value
                    normalized_value = value
                except (ValueError, TypeError):
                    pass
        
        # Store the value
        result['fields'][field_code] = value
        result['original_values'][field_code] = original_value
        
        if value is not None:
            result['available'].append(field_code)
            # Add to features for embedding
            feature_name_en = field_info['en']
            result['features_for_embedding'].append((feature_name_en, value))
        else:
            result['missing'].append(field_code)
    
    return result


def normalize_binary_value(value: str) -> str:
    """
    Normalize a binary field value to 'Yes' or 'No'.
    
    Handles multilingual values (German, English, French, Italian, etc.)
    
    Args:
        value: Original value string
        
    Returns:
        'Yes', 'No', or the original value if uncertain
    """
    if not value:
        return value
    
    lower_value = value.lower().strip()
    
    # Check negative patterns first (more specific)
    for pattern in BINARY_NEGATIVE_PATTERNS:
        if pattern in lower_value:
            return 'No'
    
    # Check positive patterns
    for pattern in BINARY_POSITIVE_PATTERNS:
        if pattern in lower_value:
            return 'Yes'
    
    # Return original if no match
    return value


def create_searchable_text_from_fields(fields_data: dict) -> str:
    """
    Create searchable text from extracted fields for text embedding.
    
    Combines field names (in English) with their values to create
    a rich text representation for vector search.
    
    Args:
        fields_data: Result from extract_specified_fields()
        
    Returns:
        Searchable text string
    """
    parts = []
    
    for field_code, value in fields_data['fields'].items():
        if value is not None:
            field_info = SPECIFIED_FIELDS[field_code]
            # Use English description for consistent search
            feature_name = field_info['en']
            parts.append(f"{feature_name}: {value}")
    
    return ", ".join(parts)


def create_normalized_features_values(fields_data: dict) -> tuple:
    """
    Create normalized feature names and values lists for vector encoding.
    
    Uses English feature names for consistent multilingual search.
    
    Args:
        fields_data: Result from extract_specified_fields()
        
    Returns:
        Tuple of (features_list, values_list)
    """
    features = []
    values = []
    
    for field_code in SPECIFIED_FIELDS_ORDERED:
        value = fields_data['fields'].get(field_code)
        if value is not None:
            field_info = SPECIFIED_FIELDS[field_code]
            # Use English description for consistent encoding
            features.append(field_info['en'])
            values.append(value)
    
    return features, values


def create_payload_indexes_for_60_fields(qdrant_client: QdrantClient, collection_name: str) -> bool:
    """
    Create payload indexes for efficient filtering on the 60 specified fields.
    
    IMPORTANT: Run this AFTER indexing is complete for 600K+ recipes.
    Creating indexes during indexing can slow down the process significantly.
    
    This creates indexes for:
    1. All 60 spec_fields (for exact match filtering)
    2. Numerical fields (for range queries)
    3. Metadata fields (country, version)
    
    Args:
        qdrant_client: Qdrant client instance
        collection_name: Name of the collection
        
    Returns:
        True if successful, False otherwise
    """
    from qdrant_client.http.models import PayloadSchemaType
    
    try:
        logger.info("=" * 60)
        logger.info("CREATING PAYLOAD INDEXES FOR 60 SPECIFIED FIELDS")
        logger.info("=" * 60)
        
        # Index each of the 60 specified fields
        logger.info("Creating indexes for spec_fields (60 fields)...")
        for field_code in SPECIFIED_FIELDS_ORDERED:
            field_info = SPECIFIED_FIELDS[field_code]
            field_type = field_info['type']
            
            # Determine the index type based on field type
            if field_type == 'numerical':
                # Float index for numerical fields (supports range queries)
                schema_type = PayloadSchemaType.FLOAT
            else:
                # Keyword index for text/binary/categorical fields (supports exact match)
                schema_type = PayloadSchemaType.KEYWORD
            
            try:
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=f"spec_fields.{field_code}",
                    field_schema=schema_type
                )
                logger.debug(f"  Created index for spec_fields.{field_code} ({schema_type})")
            except Exception as e:
                # Index might already exist
                logger.debug(f"  Index for spec_fields.{field_code} may already exist: {e}")
        
        logger.info("✅ Created indexes for 60 spec_fields")
        
        # Create indexes for numerical values (for range queries)
        logger.info("Creating indexes for numerical values...")
        for field_code in NUMERICAL_FIELDS:
            try:
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=f"numerical.{field_code}",
                    field_schema=PayloadSchemaType.FLOAT
                )
            except Exception:
                pass  # May already exist
        logger.info(f"✅ Created {len(NUMERICAL_FIELDS)} numerical field indexes")
        
        # Create indexes for metadata fields
        logger.info("Creating indexes for metadata fields...")
        metadata_indexes = [
            ("country", PayloadSchemaType.KEYWORD),
            ("version", PayloadSchemaType.KEYWORD),
            ("num_available", PayloadSchemaType.INTEGER),
            ("num_missing", PayloadSchemaType.INTEGER),
        ]
        
        for field_name, schema_type in metadata_indexes:
            try:
                qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type
                )
            except Exception:
                pass  # May already exist
        logger.info("✅ Created metadata indexes")
        
        # Create text index for feature_text (full-text search)
        try:
            from qdrant_client.http.models import TextIndexParams, TokenizerType
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="feature_text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                )
            )
            logger.info("✅ Created full-text index for feature_text")
        except Exception as e:
            logger.debug(f"Text index for feature_text may already exist: {e}")
        
        logger.info("=" * 60)
        logger.info("PAYLOAD INDEX CREATION COMPLETE")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating payload indexes: {e}")
        return False


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
                # Skip empty files
                if os.path.getsize(recipe_path) == 0:
                    continue

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
    feature_map_path: Optional[str] = None,
    feature_mappings_path: Optional[str] = None
) -> bool:
    """
    Index recipes to Qdrant using TRUE BATCH PROCESSING with EnhancedTwoStepRecipeManager.

    This preserves all sophisticated feature encoding including:
    - Binary opposition mapping
    - Numerical feature detection
    - Categorical encoding
    - MULTILINGUAL FEATURE NORMALIZATION (German/French/etc → English)

    Processes files in chunks to:
    1. Avoid loading all files into memory
    2. Show progress as recipes are indexed
    3. Use EnhancedTwoStepRecipeManager for proper encoding
    4. Normalize multilingual features to English for consistent search
    """
    try:
        # Initialize feature normalizer for multilingual support
        logger.info(
            "Initializing feature normalizer for multilingual support...")
        normalizer = FeatureNormalizer(feature_mappings_path)
        logger.info("✅ Feature normalizer initialized")

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

            # Step 1: Read batch of files and extract data using 60 SPECIFIED FIELDS
            logger.info("Step 1: Reading JSON files and extracting 60 specified fields...")
            batch_data = []

            for recipe_path in batch_files:
                try:
                    # Skip empty files (safe fallback)
                    if os.path.getsize(recipe_path) == 0:
                        logger.warning(
                            f"Skipping empty JSON file: {recipe_path}")
                        total_skipped += 1
                        continue

                    # Extract the 60 specified fields
                    fields_data = extract_specified_fields(recipe_path)
                    
                    # Get basic recipe info
                    description = extract_recipe_description(recipe_path)
                    filename = os.path.basename(recipe_path)
                    country_name = get_country_name(filename)
                    recipe_id = filename.split('.')[0]

                    # Extract version (stlan) from recipe JSON
                    version = extract_stlan(recipe_path)
                    logger.debug(
                        f"Recipe {recipe_id}: extracted version={version}, "
                        f"available fields: {len(fields_data['available'])}/60")

                    # Create normalized features and values from the 60 fields
                    # Uses English feature names for consistent multilingual search
                    normalized_features, normalized_values = create_normalized_features_values(fields_data)
                    
                    # Only process if we have at least some fields
                    if normalized_features:
                        # Enhance description with key searchable terms from 60 fields
                        enhanced_description = normalizer.enhance_description(
                            description, normalized_features, normalized_values
                        )

                        batch_data.append({
                            'recipe_id': recipe_id,
                            'features': normalized_features,  # English feature names
                            'values': normalized_values,      # Normalized values
                            'fields_data': fields_data,       # Full 60 fields structure
                            'description': enhanced_description,
                            'country': country_name,
                            'version': version
                        })
                    else:
                        # Recipe has no matching fields, skip but log
                        logger.debug(f"Recipe {recipe_id} has no matching fields from 60 specified")
                        total_skipped += 1
                        
                except Exception as e:
                    logger.warning(f"Error reading {recipe_path}: {e}")
                    continue

            if not batch_data:
                logger.warning(f"No valid recipes in batch {batch_idx + 1}")
                continue

            logger.info(f"  Loaded {len(batch_data)} valid recipes from batch")

            # Log version distribution for this batch
            version_counts = {}
            for recipe in batch_data:
                version = recipe.get('version', 'Missing')
                version_counts[version] = version_counts.get(version, 0) + 1
            logger.info(f"  Version distribution in batch: {version_counts}")

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

            # Step 3: Upload to Qdrant with STRUCTURED 60 FIELDS PAYLOAD
            logger.info("Step 3: Uploading to Qdrant with 60 specified fields...")
            points = []

            for i, recipe in enumerate(batch_data):
                try:
                    recipe_id = recipe['recipe_id']
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, recipe_id))

                    # Create feature text for payload using NORMALIZED features
                    # This enables consistent search matching
                    feature_text = manager._create_feature_text(
                        recipe['features'], recipe['values'])

                    # Get the 60 fields data
                    fields_data = recipe['fields_data']
                    
                    # Build the structured payload for 60 specified fields
                    # Each field is stored individually for efficient filtering
                    spec_fields_payload = {}
                    for field_code in SPECIFIED_FIELDS_ORDERED:
                        value = fields_data['fields'].get(field_code)
                        # Store as None (null) if missing, otherwise store the value
                        spec_fields_payload[field_code] = value
                    
                    # Build numerical values for range queries
                    numerical_payload = {}
                    for field_code, num_val in fields_data.get('numerical_values', {}).items():
                        numerical_payload[field_code] = num_val

                    point = PointStruct(
                        id=point_id,
                        vector={
                            "text": text_embeddings[i].tolist(),
                            "features": feature_vectors[i].tolist()
                        },
                        payload={
                            # Basic recipe info
                            "recipe_name": recipe_id,
                            "description": recipe['description'],
                            "country": recipe['country'],
                            "version": recipe['version'],
                            
                            # 60 SPECIFIED FIELDS - structured for filtering
                            # Each field stored individually: spec_fields.Z_MAKTX, spec_fields.Z_BRIX, etc.
                            "spec_fields": spec_fields_payload,
                            
                            # Numerical values for range queries (e.g., Z_BRIX > 40)
                            "numerical": numerical_payload,
                            
                            # Field availability metadata
                            "available_fields": fields_data['available'],
                            "missing_fields": fields_data['missing'],
                            "num_available": len(fields_data['available']),
                            "num_missing": len(fields_data['missing']),
                            
                            # Legacy format for backward compatibility
                            "features": recipe['features'][:50],
                            "values": [manager._clean_value(v) for v in recipe['values'][:50]],
                            "num_features": len(recipe['features']),
                            
                            # Normalized feature text for text-based search
                            "feature_text": feature_text[:1000],
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

        # Calculate final version distribution (sample from collection)
        logger.info("\n" + "=" * 60)
        logger.info("CALCULATING VERSION DISTRIBUTION")
        logger.info("=" * 60)
        try:
            # Sample points to get version distribution
            sample_size = min(10000, collection_info.points_count)
            sample_results = qdrant_client.scroll(
                collection_name=collection_name,
                limit=sample_size,
                with_payload=True,
                with_vectors=False
            )

            version_distribution = {'P': 0, 'L': 0, 'Missing': 0}
            for point in sample_results[0]:
                version = point.payload.get('version', 'Missing')
                if version in version_distribution:
                    version_distribution[version] += 1

            if sample_size > 0:
                logger.info(
                    f"Version distribution (sample of {sample_size:,} recipes):")
                logger.info(
                    f"  P: {version_distribution['P']:,} ({version_distribution['P']/sample_size*100:.1f}%)")
                logger.info(
                    f"  L: {version_distribution['L']:,} ({version_distribution['L']/sample_size*100:.1f}%)")
                logger.info(
                    f"  Missing: {version_distribution['Missing']:,} ({version_distribution['Missing']/sample_size*100:.1f}%)")
        except Exception as e:
            logger.warning(f"Could not calculate version distribution: {e}")

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

        # Create payload indexes for efficient filtering (run AFTER indexing)
        logger.info("\n" + "=" * 60)
        logger.info("CREATING PAYLOAD INDEXES FOR 60 SPECIFIED FIELDS")
        logger.info("(This improves query performance for 600K+ recipes)")
        logger.info("=" * 60)
        create_payload_indexes_for_60_fields(qdrant_client, collection_name)

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
        feature_mappings_path = os.getenv(
            'FEATURE_MAPPINGS_PATH', '/usr/src/app/data/feature_extraction_mappings.json')

        logger.info("=" * 60)
        logger.info("Starting Qdrant Vector Index Initialization")
        logger.info("TRUE BATCH PROCESSING with EnhancedTwoStepRecipeManager")
        logger.info("+ MULTILINGUAL FEATURE NORMALIZATION (DE/FR/etc → EN)")
        logger.info("+ 60 SPECIFIED FIELDS STRUCTURED INDEXING")
        logger.info("=" * 60)
        logger.info(f"60 Specified Fields: {len(SPECIFIED_FIELDS_ORDERED)} fields")
        logger.info(f"  - Binary: {len(BINARY_FIELDS)}, Numerical: {len(NUMERICAL_FIELDS)}")
        logger.info(f"  - Categorical: {len(CATEGORICAL_FIELDS)}, Text: {len(TEXT_FIELDS)}")
        logger.info(f"Qdrant Host: {qdrant_host}")
        logger.info(f"Qdrant Port: {qdrant_port}")
        logger.info(f"Collection Name: {collection_name}")
        logger.info(f"Embedding Model: {embedding_model}")
        logger.info(f"Data Directory: {data_dir}")
        logger.info(f"Feature Map: {feature_map_path}")
        logger.info(f"Feature Mappings: {feature_mappings_path}")
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
        # + MULTILINGUAL FEATURE NORMALIZATION
        logger.info(
            "Starting batch indexing with EnhancedTwoStepRecipeManager + Feature Normalization...")
        success = index_recipes_to_qdrant_batched(
            qdrant_client,
            embedding_model,
            collection_name,
            data_dir,
            feature_map_path,
            feature_mappings_path
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
