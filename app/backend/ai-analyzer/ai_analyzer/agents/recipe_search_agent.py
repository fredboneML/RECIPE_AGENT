#!/usr/bin/env python3
from ai_analyzer.utils.model_logger import query_llm
from qdrant_recipe_manager import QdrantRecipeManager
import os
import logging
import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import sys
from sqlalchemy import create_engine, text

# Add the src directory to the path to import recipe search modules
sys.path.append(os.path.join(os.path.dirname(
    __file__), '..', '..', '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Database connection for translation cache
def get_db_engine():
    """Get database engine for translation cache operations"""
    try:
        db_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('DB_HOST', 'database')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"
        return create_engine(db_url)
    except Exception as e:
        logger.error(f"Error creating database engine: {e}")
        return None


def get_cached_translation(recipe_name: str, target_language: str) -> Optional[Dict[str, Any]]:
    """
    Check if a translation exists in the cache for a specific recipe and language.

    Args:
        recipe_name: The recipe identifier (e.g., "000000000000442937_AT10_01_L")
        target_language: The target language code (e.g., "fr", "de", "en")

    Returns:
        Cached translation dictionary or None if not found
    """
    try:
        engine = get_db_engine()
        if not engine:
            return None

        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT translated_characteristics 
                FROM recipe_translation_cache 
                WHERE recipe_name = :recipe_name 
                AND target_language = :target_language
            """), {"recipe_name": recipe_name, "target_language": target_language})

            row = result.fetchone()
            if row:
                logger.info(
                    f"Found cached translation for {recipe_name} in {target_language}")
                return row[0]  # JSONB is automatically parsed

        return None

    except Exception as e:
        logger.error(f"Error retrieving cached translation: {e}")
        return None


def save_translation_to_cache(recipe_name: str, target_language: str, translated_data: Dict[str, Any]) -> bool:
    """
    Save a translation to the cache.

    Args:
        recipe_name: The recipe identifier
        target_language: The target language code
        translated_data: Dictionary containing translated characteristics

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        engine = get_db_engine()
        if not engine:
            return False

        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO recipe_translation_cache (recipe_name, target_language, translated_characteristics)
                VALUES (:recipe_name, :target_language, CAST(:translated_characteristics AS jsonb))
                ON CONFLICT (recipe_name, target_language) 
                DO UPDATE SET 
                    translated_characteristics = EXCLUDED.translated_characteristics,
                    updated_at = CURRENT_TIMESTAMP
            """), {
                "recipe_name": recipe_name,
                "target_language": target_language,
                "translated_characteristics": json.dumps(translated_data)
            })

        logger.info(
            f"Saved translation for {recipe_name} in {target_language} to cache")
        return True

    except Exception as e:
        logger.error(f"Error saving translation to cache: {e}")
        return False


def detect_text_language(text: str) -> str:
    """
    Detect the language of a given text using AI.
    Returns language code (en, fr, de, it, es, pt, nl, da)
    """
    if not text or not text.strip():
        return "en"

    try:
        prompt = f"""Detect the language of this text and respond with ONLY the language code.

Text: "{text[:500]}"

Valid codes: en, fr, de, it, es, pt, nl, da
Default to "en" if unclear.

Respond with only the language code."""

        response = query_llm(prompt, provider="openai")
        if response:
            lang_code = response.strip().lower()
            valid_codes = ["en", "it", "fr", "de", "es", "pt", "nl", "da"]
            if lang_code in valid_codes:
                return lang_code
        return "en"

    except Exception as e:
        logger.error(f"Error detecting text language: {e}")
        return "en"


def detect_recipe_language(features: List[str], values: List[str], num_samples: int = 10) -> str:
    """
    Detect the dominant language of recipe characteristics by sampling both features and values.

    Args:
        features: List of characteristic names (e.g., ["Allergene", "Color", "Flavour"])
        values: List of corresponding values
        num_samples: Number of samples to take from each list (default 10)

    Returns:
        Detected language code (en, fr, de, it, es, pt, nl, da)
    """
    if not features and not values:
        return "en"

    # Sample up to num_samples from features and values
    # Spread samples across the list to get better coverage
    sample_texts = []

    # Sample features (characteristic names)
    if features:
        step = max(1, len(features) // num_samples)
        for i in range(0, len(features), step):
            if len(sample_texts) < num_samples and features[i]:
                sample_texts.append(features[i])

    # Sample values
    if values:
        step = max(1, len(values) // num_samples)
        for i in range(0, len(values), step):
            if len(sample_texts) < num_samples * 2 and values[i]:
                # Convert to string and skip very short values or numbers
                val_str = str(values[i]).strip()
                if len(val_str) > 2 and not val_str.replace('.', '').replace('-', '').replace(',', '').isdigit():
                    sample_texts.append(val_str)

    if not sample_texts:
        return "en"

    # Create combined sample text for language detection
    combined_sample = " | ".join(sample_texts[:20])  # Max 20 samples

    logger.info(
        f"Language detection sampling {len(sample_texts)} items: {combined_sample[:200]}...")

    try:
        prompt = f"""Analyze the following text samples from a recipe database and determine the DOMINANT language.
These are characteristic names and values from a food product recipe.

Text samples: "{combined_sample}"

The text may contain:
- Characteristic names (e.g., "Allergene", "Color", "Flavour", "Farbe")
- Values (e.g., "Mit Allergenen", "Allergen-free", "keine künstl. Farbe")
- Technical terms that may be in any language

Identify the dominant language by counting which language appears most frequently.
Respond with ONLY the language code from this list:
- "en" for English
- "de" for German
- "fr" for French
- "it" for Italian
- "es" for Spanish
- "pt" for Portuguese
- "nl" for Dutch
- "da" for Danish

If the text is mixed or unclear, determine which language is MORE prevalent.
Respond with only the language code, nothing else."""

        response = query_llm(prompt, provider="openai")
        if response:
            lang_code = response.strip().lower()
            valid_codes = ["en", "it", "fr", "de", "es", "pt", "nl", "da"]
            if lang_code in valid_codes:
                logger.info(f"Detected dominant language: {lang_code}")
                return lang_code
            else:
                logger.warning(
                    f"Invalid language code returned: {lang_code}, defaulting to en")
        return "en"

    except Exception as e:
        logger.error(f"Error detecting recipe language: {e}")
        return "en"


def translate_characteristics_with_llm(features: List[str], values: List[str], target_language: str) -> Tuple[List[str], List[str]]:
    """
    Translate recipe characteristics and values to the target language using LLM.

    Args:
        features: List of characteristic names (e.g., ["Color", "Flavour", "Industry"])
        values: List of corresponding values
        target_language: Target language code (e.g., "fr", "de")

    Returns:
        Tuple of (translated_features, translated_values)
    """
    if not features or not values:
        return features, values

    # Language mapping for prompt
    language_names = {
        "en": "English",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "es": "Spanish",
        "pt": "Portuguese",
        "nl": "Dutch",
        "da": "Danish"
    }

    target_lang_name = language_names.get(target_language, "English")

    try:
        # Create a structured prompt for translation
        characteristics_json = json.dumps([
            {"characteristic": f, "value": v}
            for f, v in zip(features, values)
        ], ensure_ascii=False)

        prompt = f"""Translate the following recipe characteristics and their values to {target_lang_name}.
These are food/recipe industry terms that should be translated appropriately.

Input (JSON array):
{characteristics_json}

Rules:
1. Translate both the "characteristic" name and the "value" to {target_lang_name}
2. Keep technical terms that are internationally used unchanged (e.g., "Halal", "Kosher", brand names)
3. Maintain the same JSON structure in your response
4. If a term is already in {target_lang_name}, keep it unchanged

Respond with ONLY the translated JSON array, no explanations:"""

        response = query_llm(prompt, provider="openai")

        if response:
            # Clean the response and parse JSON
            response_clean = response.strip()
            # Remove markdown code blocks if present
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                response_clean = "\n".join(
                    lines[1:-1] if lines[-1] == "```" else lines[1:])

            translated = json.loads(response_clean)

            translated_features = [
                item.get("characteristic", f) for item, f in zip(translated, features)]
            translated_values = [item.get("value", v)
                                 for item, v in zip(translated, values)]

            logger.info(
                f"Successfully translated {len(features)} characteristics to {target_lang_name}")
            return translated_features, translated_values

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing translation response: {e}")
    except Exception as e:
        logger.error(f"Error translating characteristics: {e}")

    return features, values  # Return original on error


def translate_recipe_characteristics(
    recipe_name: str,
    features: List[str],
    values: List[str],
    target_language: str
) -> Tuple[List[str], List[str]]:
    """
    Translate recipe characteristics to target language, using cache when available.

    Args:
        recipe_name: The recipe identifier for caching
        features: List of characteristic names
        values: List of corresponding values
        target_language: Target language code

    Returns:
        Tuple of (translated_features, translated_values)
    """
    if not features or not values:
        return features, values

    # Check if we need translation by detecting the predominant language
    # Sample both features (characteristic names) AND values for better accuracy
    detected_lang = detect_recipe_language(features, values, num_samples=10)

    logger.info(
        f"Recipe {recipe_name}: detected language={detected_lang}, target={target_language}")

    # If already in target language, no translation needed
    if detected_lang == target_language:
        logger.info(
            f"Recipe {recipe_name} is already in {target_language}, skipping translation")
        return features, values

    # Check cache first
    cached = get_cached_translation(recipe_name, target_language)
    if cached:
        cached_features = cached.get("features", features)
        cached_values = cached.get("values", values)
        logger.info(
            f"Using cached translation for {recipe_name} in {target_language}")
        return cached_features, cached_values

    # Translate using LLM
    translated_features, translated_values = translate_characteristics_with_llm(
        features, values, target_language
    )

    # Save to cache for future use
    if translated_features != features or translated_values != values:
        save_translation_to_cache(recipe_name, target_language, {
            "features": translated_features,
            "values": translated_values
        })

    return translated_features, translated_values


def detect_language_with_ai(text: str) -> str:
    """Use AI to detect the language of the input text"""
    if not text or not text.strip():
        return "en"

    try:
        # Create a prompt for language detection focused on recipe context
        prompt = f"""Analyze the following text and determine its language. This text is a recipe description or search query for finding similar recipes in a food product database.

Text: "{text}"

The text may contain:
- Recipe names and descriptions
- Food ingredients and characteristics
- Product specifications (Color, Flavour, Stabilizer, etc.)
- Industry terms (Dairy, Halal, Kosher, etc.)
- Technical specifications (Brix, Pasteurization, etc.)

Please identify the language and respond with ONLY the language code from this list:
- "en" for English
- "it" for Italian  
- "fr" for French
- "de" for German
- "es" for Spanish
- "pt" for Portuguese
- "nl" for Dutch
- "da" for Danish

If the text contains multiple languages or is unclear, default to "en" (English).

Respond with only the language code, nothing else."""

        # Make AI call for language detection
        response = query_llm(prompt, provider="openai")

        if response:
            # Clean the response and extract language code
            language_code = response.strip().lower()

            # Validate the response
            valid_codes = ["en", "it", "fr", "de", "es", "pt", "nl", "da"]
            if language_code in valid_codes:
                logger.info(
                    f"AI detected language: {language_code} for query: '{text[:100]}...'")
                return language_code
            else:
                logger.warning(
                    f"AI returned invalid language code: {language_code}, defaulting to English")
                return "en"
        else:
            logger.warning(
                "AI language detection failed, defaulting to English")
            return "en"

    except Exception as e:
        logger.error(f"Error in AI language detection: {e}")
        logger.exception("Detailed error:")
        return "en"


# =============================================================================
# FIELD NAME TRANSLATION CACHE (for the 60 specified fields)
# =============================================================================

def get_cached_field_translations(target_language: str) -> Optional[Dict[str, str]]:
    """
    Get cached field name translations for the 60 specified fields.
    
    Args:
        target_language: The target language code (e.g., "fr", "it", "es")
    
    Returns:
        Dictionary mapping field codes to translated names, or None if not cached
    """
    try:
        engine = get_db_engine()
        if not engine:
            return None
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT translated_fields 
                FROM field_name_translation_cache 
                WHERE target_language = :target_language
            """), {"target_language": target_language})
            
            row = result.fetchone()
            if row:
                logger.info(f"Found cached field translations for {target_language}")
                return row[0]  # JSONB is automatically parsed
        
        return None
    
    except Exception as e:
        logger.error(f"Error retrieving cached field translations: {e}")
        return None


def save_field_translations_to_cache(target_language: str, translations: Dict[str, str]) -> bool:
    """
    Save field name translations to cache.
    
    Args:
        target_language: The target language code
        translations: Dictionary mapping field codes to translated names
    
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        engine = get_db_engine()
        if not engine:
            return False
        
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO field_name_translation_cache (target_language, translated_fields)
                VALUES (:target_language, CAST(:translated_fields AS jsonb))
                ON CONFLICT (target_language) 
                DO UPDATE SET 
                    translated_fields = EXCLUDED.translated_fields,
                    updated_at = CURRENT_TIMESTAMP
            """), {
                "target_language": target_language,
                "translated_fields": json.dumps(translations)
            })
        
        logger.info(f"Saved field translations for {target_language} to cache")
        return True
    
    except Exception as e:
        logger.error(f"Error saving field translations to cache: {e}")
        return False


def _load_official_field_translations() -> Dict[str, Dict[str, str]]:
    """
    Load official translations for field names from code_translation.csv.
    This file contains authoritative translations in EN, DE, ES, PL, FR.
    
    Returns:
        Dictionary mapping Z-code to {en: str, de: str, es: str, pl: str, fr: str}
    """
    translations = {}
    # Relative path from this file: agents/ -> ai_analyzer/ -> ai-analyzer/ -> backend/
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'code_translation.csv')
    csv_path = os.path.normpath(csv_path)
    # Fallback to Docker container absolute path
    if not os.path.exists(csv_path):
        csv_path = '/usr/src/app/code_translation.csv'
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                z_code = row.get('Merkmalname', '').strip()
                if z_code:
                    translations[z_code] = {
                        'en': row.get('Description (EN)', '').strip(),
                        'de': row.get('Description (DE)', '').strip(),
                        'es': row.get('Description (ES)', '').strip(),
                        'pl': row.get('Description (PL)', '').strip(),
                        'fr': row.get('Description (FR)', '').strip(),
                    }
        logger.info(f"Loaded official field translations for {len(translations)} Z-codes from {csv_path}")
    except Exception as e:
        logger.warning(f"Could not load official field translations from {csv_path}: {e}")
    
    return translations


# Official translations loaded at module level from code_translation.csv
# These are used for the "Field Name" column in the comparison table
# for languages: EN, DE, ES, PL, FR. Other languages still use LLM.
OFFICIAL_FIELD_TRANSLATIONS = _load_official_field_translations()

# Languages that have official translations in code_translation.csv
OFFICIAL_TRANSLATION_LANGUAGES = {"en", "de", "es", "pl", "fr"}


def translate_field_names_with_llm(field_names: List[Tuple[str, str, str]], target_language: str) -> Dict[str, str]:
    """
    Translate the 60 field names to the target language using LLM.
    
    Args:
        field_names: List of tuples (code, english_name, german_name)
        target_language: Target language code (e.g., "fr", "it", "es")
    
    Returns:
        Dictionary mapping field codes to translated names
    """
    language_names = {
        "en": "English",
        "de": "German",
        "fr": "French",
        "it": "Italian",
        "es": "Spanish",
        "pt": "Portuguese",
        "nl": "Dutch",
        "da": "Danish"
    }
    
    target_lang_name = language_names.get(target_language, "English")
    
    # Build a list of fields to translate
    fields_to_translate = [
        {"code": code, "english": en, "german": de}
        for code, en, de in field_names
    ]
    
    try:
        prompt = f"""Translate the following food/recipe industry field names to {target_lang_name}.

These are technical terms used in food product databases for characteristics like:
- Product properties (Standard product, Product segment, etc.)
- Ingredients (Sweetener, Preservatives, Colors, Flavors, etc.)
- Technical parameters (Brix, pH, Viscosity, etc.)
- Certifications (Kosher, Halal, etc.)

Input fields (JSON array with code, English name, German name):
{json.dumps(fields_to_translate, ensure_ascii=False, indent=2)}

Rules:
1. Translate to {target_lang_name}
2. Keep technical terms that are internationally recognized unchanged (e.g., "Brix", "pH", "Halal", "Kosher")
3. Use industry-standard terminology in {target_lang_name}
4. Maintain consistency with food industry standards

Respond with ONLY a JSON object mapping the code to the translated name. Example format:
{{"Z_MAKTX": "Translated name", "Z_INH01": "Another translation", ...}}

No explanations, just the JSON object:"""

        response = query_llm(prompt, provider="openai")
        
        if response:
            # Clean response
            response_clean = response.strip()
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                response_clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            translations = json.loads(response_clean)
            logger.info(f"Successfully translated {len(translations)} field names to {target_lang_name}")
            return translations
    
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing field name translation response: {e}")
    except Exception as e:
        logger.error(f"Error translating field names: {e}")
    
    # Return English names as fallback
    return {code: en for code, en, de in field_names}


def get_translated_field_names(field_definitions: List[Tuple[str, str, str]], target_language: str) -> Dict[str, str]:
    """
    Get translated field names for the target language.
    
    For languages with official translations in code_translation.csv (EN, DE, ES, PL, FR),
    uses those authoritative translations directly. No LLM or cache needed.
    
    For other languages (e.g., PT, IT, NL, DA), checks cache first, then falls back to LLM.
    
    Args:
        field_definitions: List of (code, english_name, german_name) tuples
        target_language: Target language code
    
    Returns:
        Dictionary mapping field codes to display names in target language
    """
    # For languages with official translations in code_translation.csv, use them directly
    if target_language in OFFICIAL_TRANSLATION_LANGUAGES and OFFICIAL_FIELD_TRANSLATIONS:
        translations = {}
        for code, en, de in field_definitions:
            official = OFFICIAL_FIELD_TRANSLATIONS.get(code, {})
            translated = official.get(target_language, '').strip()
            # Fallback: if official translation is empty or '#N/A', use English
            if not translated or translated == '#N/A':
                translated = en
            translations[code] = translated
        logger.info(f"Using official translations from code_translation.csv for {target_language}")
        return translations
    
    # For other languages, check cache first
    cached = get_cached_field_translations(target_language)
    if cached:
        logger.info(f"Using cached field translations for {target_language}")
        return cached
    
    # Translate using LLM
    translations = translate_field_names_with_llm(field_definitions, target_language)
    
    # Save to cache for future use
    if translations:
        save_field_translations_to_cache(target_language, translations)
    
    return translations


def translate_comparison_values(values: List[str], target_language: str) -> List[str]:
    """
    Translate recipe values to the target language if needed.
    
    Args:
        values: List of recipe field values
        target_language: Target language code
    
    Returns:
        List of translated values
    """
    # Skip translation for English (most values are already in English)
    if target_language == "en":
        return values
    
    # Filter non-empty, non-numeric values that might need translation
    values_to_translate = []
    indices_to_translate = []
    
    for i, val in enumerate(values):
        if val and not val.replace('.', '').replace(',', '').replace('%', '').replace(' ', '').isdigit():
            # Check if it's a common Yes/No or technical term
            lower_val = val.lower().strip()
            # Skip values that don't need translation
            skip_values = ['yes', 'no', 'ja', 'nein', 'oui', 'non', '-', '', 
                          'halal', 'kosher', 'pur', 'puree', 'pieces']
            if lower_val not in skip_values and not any(c.isdigit() for c in val[:3]):
                values_to_translate.append(val)
                indices_to_translate.append(i)
    
    # If nothing to translate, return original
    if not values_to_translate:
        return values
    
    try:
        language_names = {
            "de": "German", "fr": "French", "it": "Italian",
            "es": "Spanish", "pt": "Portuguese", "nl": "Dutch", "da": "Danish"
        }
        target_lang_name = language_names.get(target_language, "English")
        
        prompt = f"""Translate these food/recipe product values to {target_lang_name}:

Values: {json.dumps(values_to_translate, ensure_ascii=False)}

Rules:
1. Keep technical terms unchanged (Brix, pH, brand names)
2. Translate common food industry terms
3. Return a JSON array with translations in the same order

Respond with ONLY a JSON array:"""

        response = query_llm(prompt, provider="openai")
        
        if response:
            response_clean = response.strip()
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                response_clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            translated = json.loads(response_clean)
            
            # Merge translations back into original list
            result = values.copy()
            for idx, trans_idx in enumerate(indices_to_translate):
                if idx < len(translated):
                    result[trans_idx] = translated[idx]
            
            return result
    
    except Exception as e:
        logger.error(f"Error translating values: {e}")
    
    return values


# The 60 specified fields in the exact order for display
SPECIFIED_FIELDS_60 = [
    ('Z_MAKTX',       'Material short text',             'Materialkurztext'),
    ('Z_INH01',       'Standard product',                'Standardprodukt'),
    ('Z_WEIM',        'Product segment',                 'Produktsegment'),
    ('Z_KUNPROGRU',   'Customer product group',          'Kundenproduktgruppe'),
    ('Z_PRODK',       'Market segments',                 'Produktkategorien'),
    ('Z_INH07',       'Extreme recipe',                  'Extremrezeptur'),
    ('Z_KOCHART',     'Pasteurization type',             'Kochart'),
    ('Z_KNOGM',       'GMO presence',                    'GMO enthalten'),
    ('Z_INH08',       'Contains GMO',                    'Nicht Genfrei'),
    ('Z_INH12',       'Allergen-free',                   'Allergenfrei'),
    ('ZMX_TIPOALERG', 'Allergenic type',                 'Allergentyp'),
    ('Z_INH02',       'Sweetener',                       'Süßstoff'),
    ('Z_INH03',       'Saccharose',                      'Saccharose'),
    ('Z_INH19',       'Aspartame',                       'Aspartam'),
    ('Z_INH04',       'Preserved',                       'Konservierung'),
    ('Z_INH18',       'Color',                           'Farbe'),
    ('Z_INH05',       'Artificial colors',               'Künstliche Farben'),
    ('Z_INH09',       'Flavour',                         'Aroma'),
    ('Z_INH06',       'Nature identical flavor',         'naturident/künstliches Aroma'),
    ('Z_INH06Z',      'Natural flavor',                  'Natürliche Aromen'),
    ('Z_FSTAT',       'Flavor status',                   'Flavor status'),
    ('Z_INH21',       'Vitamins',                        'Vitamine'),
    ('Z_INH13',       'Starch',                          'Stärke'),
    ('Z_INH14',       'Pectin',                          'Pektin'),
    ('Z_INH15',       'LBG',                             'IBKM'),
    ('Z_INH16',       'Blend',                           'Mischung'),
    ('Z_INH20',       'Xanthan',                         'Xanthan'),
    ('Z_STABGU',      'Stabilizing System - Guar',       'Stabilizing System - Guar'),
    ('Z_STABCAR',     'Stabilizing System - Carrageen', 'Stabilizing System - Carrageen'),
    ('Z_STAGEL',      'Stabilizing System - Gellan',    'Stabilizing System - Gellan'),
    ('Z_STANO',       'Stabilizing System - No stabil', 'Stabilizing System - No stabil'),
    ('Z_INH17',       'Other stabilizer',               'Andere Stabilisatoren'),
    ('Z_BRIX',        'Brix',                            'Brix'),
    ('Z_PH',          'pH',                              'PH'),
    ('ZM_PH',         'PH AFM',                          'PH AFM'),
    ('Z_VISK20S',     'Viscosity 20s (20°C)',           'Viskosität 20s (20°C)'),
    ('Z_VISK20S_7C',  'Viscosity 20s (7°C)',            'Viskosität 20s (7°C)'),
    ('Z_VISK30S',     'Viscosity 30s',                   'Viskosität 30s'),
    ('Z_VISK60S',     'Viscosity 60s',                   'Viskosität 60s'),
    ('Z_VISKHAAKE',   'Viscosity HAAKE',                'Viskosität HAAKE'),
    ('ZMX_DD103',     'Haake Viscosity',                'Haake Viskosität'),
    ('ZMX_DD102',     'Brookfield Viscosity',           'Brookfield Viskosität'),
    ('ZM_AW',         'Water Activity AFM',              'Wasseraktivität AFM'),
    ('Z_FGAW',        'Water activity (FruitPrep)',     'Water activity (FruitPrep)'),
    ('Z_FRUCHTG',     'Fruit content',                   'Fruchtgehalt'),
    ('ZMX_DD108',     'Fruit Content',                   'Fruchtgehalt'),
    ('Z_AW',          'Fruit retention %',               'Auswaschung %'),
    ('Z_FLST',        'Puree/with pieces',               'Flüssig/Stückig'),
    ('Z_PP',          'Puree/Pieces',                    'Puree/Pieces'),
    ('ZMX_DD109',     '% Fruit Identity',                '% Dosierung'),
    ('Z_DOSIER',      'Dosage',                          'Dosierung'),
    ('Z_ZUCKER',      'Sugar',                           'Zucker'),
    ('Z_FETTST',      'Fat level',                       'Fettstufe'),
    ('ZMX_DD104',     'White Mass type',                 'Weisse Masse typ'),
    ('Z_PROT',        'Protein content',                 'Protein content'),
    ('Z_SALZ',        'Salt',                            'Salz'),
    ('Z_INH01K',      'Kosher',                          'Kosher'),
    ('Z_INH01H',      'Halal',                           'Halal'),
    ('Z_DAIRY',       'Non-Dairy Product',               'Non-Dairy Product'),
    ('Z_BFS',         'Bake/Freeze Stability',           'Bake/Freeze Stability'),
]


def _compute_recipe_differences(
    recipe_raw_values: Dict[str, str],
    numerical_filters: Optional[Dict[str, Dict[str, Any]]],
    categorical_filters: Optional[Dict[str, Dict[str, Any]]],
    field_display_names: Dict[str, str],
    field_index_map: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Compute the differences between extracted constraints and a recipe's actual values.
    
    Args:
        recipe_raw_values: Dict mapping field code to the raw (untranslated) value string
        numerical_filters: Extracted numerical constraints (e.g. {"Z_BRIX": {"gte": 31.99, "lte": 32.01}})
        categorical_filters: Extracted categorical constraints (e.g. {"Z_INH04": {"value": "No"}})
        field_display_names: Dict mapping field code to translated display name
        field_index_map: Dict mapping field code to index position in field_definitions
    
    Returns:
        List of diffs: [{code, field_name, expected, actual, field_index}]
    """
    differences = []
    
    # Check numerical constraints
    if numerical_filters:
        for code, range_spec in numerical_filters.items():
            raw_val = recipe_raw_values.get(code, "").strip()
            if not raw_val:
                # No value in recipe for this field - that's a difference
                differences.append({
                    "code": code,
                    "field_name": field_display_names.get(code, code),
                    "expected": _format_numerical_expected(range_spec),
                    "actual": "-",
                    "field_index": field_index_map.get(code, -1),
                    "type": "numerical"
                })
                continue
            try:
                actual_num = float(raw_val.replace(",", "."))
                gte = range_spec.get("gte")
                lte = range_spec.get("lte")
                gt = range_spec.get("gt")
                lt = range_spec.get("lt")
                
                is_match = True
                if gte is not None and actual_num < gte:
                    is_match = False
                if lte is not None and actual_num > lte:
                    is_match = False
                if gt is not None and actual_num <= gt:
                    is_match = False
                if lt is not None and actual_num >= lt:
                    is_match = False
                
                if not is_match:
                    differences.append({
                        "code": code,
                        "field_name": field_display_names.get(code, code),
                        "expected": _format_numerical_expected(range_spec),
                        "actual": raw_val,
                        "field_index": field_index_map.get(code, -1),
                        "type": "numerical"
                    })
            except (ValueError, TypeError):
                # Can't parse as number, treat as difference
                differences.append({
                    "code": code,
                    "field_name": field_display_names.get(code, code),
                    "expected": _format_numerical_expected(range_spec),
                    "actual": raw_val,
                    "field_index": field_index_map.get(code, -1),
                    "type": "numerical"
                })
    
    # Check categorical constraints
    if categorical_filters:
        for code, match_spec in categorical_filters.items():
            expected_value = match_spec.get("value", "")
            raw_val = recipe_raw_values.get(code, "").strip()
            
            if not raw_val:
                differences.append({
                    "code": code,
                    "field_name": field_display_names.get(code, code),
                    "expected": expected_value,
                    "actual": "-",
                    "field_index": field_index_map.get(code, -1),
                    "type": "categorical"
                })
                continue
            
            # Case-insensitive comparison
            if raw_val.lower() != expected_value.lower():
                differences.append({
                    "code": code,
                    "field_name": field_display_names.get(code, code),
                    "expected": expected_value,
                    "actual": raw_val,
                    "field_index": field_index_map.get(code, -1),
                    "type": "categorical"
                })
    
    return differences


def _format_numerical_expected(range_spec: Dict[str, Any]) -> str:
    """Format a numerical range spec into a human-readable string."""
    parts = []
    gte = range_spec.get("gte")
    lte = range_spec.get("lte")
    gt = range_spec.get("gt")
    lt = range_spec.get("lt")
    
    # If gte and lte are very close, show as single value
    if gte is not None and lte is not None and abs(lte - gte) < 0.1:
        mid = (gte + lte) / 2
        # Format nicely: remove unnecessary decimals
        if mid == int(mid):
            return str(int(mid))
        return f"{mid:.1f}"
    
    if gte is not None:
        parts.append(f"≥{gte}")
    if gt is not None:
        parts.append(f">{gt}")
    if lte is not None:
        parts.append(f"≤{lte}")
    if lt is not None:
        parts.append(f"<{lt}")
    
    return ", ".join(parts) if parts else "?"


def create_comparison_table(
    results: List[Dict[str, Any]],
    detected_language: str = "en",
    numerical_filters: Optional[Dict[str, Dict[str, Any]]] = None,
    categorical_filters: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create a comparison table structure for the top 3 recipes.
    Uses the 60 specified fields in the correct order.
    Translates field names and values to match the detected language of the user's query.
    Translations are cached in the database for performance.

    Also computes per-recipe differences between the extracted constraints and the actual
    recipe field values, so the UI can highlight mismatches.

    Args:
        results: List of recipe search results
        detected_language: The language detected from the user's query (e.g., "fr", "de", "en")
        numerical_filters: Extracted numerical constraints (e.g. {"Z_BRIX": {"gte": 31.99, "lte": 32.01}})
        categorical_filters: Extracted categorical constraints (e.g. {"Z_INH04": {"value": "No"}})

    Returns:
        Dictionary with table structure containing:
        - field_definitions: List of {code, name_en, name_de, display_name} for the 60 fields in order
        - recipes: List of recipe data with values for each field and differences from constraints
        - has_data: Boolean
        - extracted_constraints: Summary of constraints used for comparison
    """
    try:
        # Take only top 3 recipes
        top_recipes = results[:3]

        if not top_recipes:
            return None

        # Get translated field names
        # For EN, DE, ES, PL, FR: uses official translations from code_translation.csv
        # For other languages: uses cached LLM translations
        translated_field_names = get_translated_field_names(SPECIFIED_FIELDS_60, detected_language)
        
        # Build field definitions list (60 fields in order) with translations
        field_definitions = []
        for code, name_en, name_de in SPECIFIED_FIELDS_60:
            # Use translated name from official CSV or cache/LLM, fallback to English
            display_name = translated_field_names.get(code, name_en)
            
            field_definitions.append({
                "code": code,
                "name_en": name_en,
                "name_de": name_de,
                "display_name": display_name
            })

        # Build field_index_map and field_display_names for difference computation
        field_index_map = {}
        field_display_names = {}
        for idx, (code, name_en, name_de) in enumerate(SPECIFIED_FIELDS_60):
            field_index_map[code] = idx
            field_display_names[code] = translated_field_names.get(code, name_en)

        # Check if we have any constraints to compare against
        has_constraints = bool(numerical_filters) or bool(categorical_filters)

        # Initialize table structure with new format
        table_data = {
            "field_definitions": field_definitions,
            "recipes": [],
            "has_data": len(top_recipes) > 0,
            "detected_language": detected_language,
            "has_constraints": has_constraints
        }

        # If we have constraints, also include a summary of the extracted constraints
        # so the frontend knows what was searched for
        if has_constraints:
            constraint_codes = set()
            if numerical_filters:
                constraint_codes.update(numerical_filters.keys())
            if categorical_filters:
                constraint_codes.update(categorical_filters.keys())
            table_data["constrained_field_codes"] = list(constraint_codes)

        # Process each recipe
        for recipe in top_recipes:
            features = recipe.get("features", [])
            values = recipe.get("values", [])
            recipe_name = recipe.get("recipe_name", recipe.get("id", "Unknown"))
            recipe_id = recipe.get("id", "")

            # Create feature-value mapping from recipe data
            feature_map = {}
            for i, feature in enumerate(features):
                if i < len(values):
                    feature_map[feature] = values[i]

            # Also check spec_fields from payload if available
            payload = recipe.get("payload", {})
            spec_fields = payload.get("spec_fields", {})
            numerical = payload.get("numerical", {})

            # Build values list for this recipe following the 60-field order
            # Also build raw values map (untranslated) for difference computation
            recipe_values = []
            raw_values_map = {}
            for code, name_en, name_de in SPECIFIED_FIELDS_60:
                value = ""
                
                # Try to get value from various sources
                # 1. From spec_fields in payload
                if code in spec_fields and spec_fields[code]:
                    value = str(spec_fields[code])
                # 2. From numerical fields in payload
                elif code in numerical and numerical[code] is not None:
                    value = str(numerical[code])
                # 3. From feature map using English name
                elif name_en in feature_map:
                    value = str(feature_map[name_en])
                # 4. From feature map using German name
                elif name_de in feature_map:
                    value = str(feature_map[name_de])
                # 5. From feature map using code directly
                elif code in feature_map:
                    value = str(feature_map[code])

                recipe_values.append(value)
                raw_values_map[code] = value

            # Compute differences BEFORE translating values (use raw EN/DE values)
            differences = []
            if has_constraints:
                differences = _compute_recipe_differences(
                    raw_values_map,
                    numerical_filters,
                    categorical_filters,
                    field_display_names,
                    field_index_map
                )

            # Translate values if target language is not English or German
            # (Values in DB are typically in EN or DE)
            if detected_language not in ("en", "de"):
                # Check if we have cached translation for this recipe's values
                cached_values = get_cached_translation(recipe_name, detected_language)
                if cached_values and "values_60" in cached_values:
                    recipe_values = cached_values["values_60"]
                    logger.info(f"Using cached value translations for {recipe_name} in {detected_language}")
                else:
                    # Translate values and cache them
                    translated_values = translate_comparison_values(recipe_values, detected_language)
                    if translated_values != recipe_values:
                        # Save to cache with the 60-field values
                        save_translation_to_cache(recipe_name, detected_language, {
                            "values_60": translated_values,
                            "features": [],  # Not used in new format
                            "values": []  # Not used in new format
                        })
                        recipe_values = translated_values

            recipe_table_data = {
                "recipe_name": recipe_name,
                "recipe_id": recipe_id,
                "values": recipe_values,  # Values in same order as field_definitions (translated if needed)
                "differences": differences,  # Differences from extracted constraints
                # Keep characteristics for backward compatibility
                "characteristics": [
                    {
                        "charactDescr": fd["display_name"],
                        "code": fd["code"],
                        "valueCharLong": recipe_values[i]
                    }
                    for i, fd in enumerate(field_definitions)
                ]
            }

            if differences:
                logger.info(f"Recipe {recipe_name}: {len(differences)} field(s) differ from extracted constraints")

            table_data["recipes"].append(recipe_table_data)

        logger.info(
            f"Created comparison table for {len(top_recipes)} recipes in {detected_language} with 60 specified fields")
        return table_data

    except Exception as e:
        logger.error(f"Error creating comparison table: {e}")
        logger.exception("Detailed error:")
        return None


def format_response_in_language(results: List[Dict[str, Any]], language: str) -> str:
    """Format the response in the specified language"""
    if not results:
        if language == "nl":
            return "Geen recepten gevonden die overeenkomen met uw beschrijving. Probeer een andere zoekterm of geef meer details over het recept dat u zoekt."
        elif language == "fr":
            return "Aucune recette trouvée correspondant à votre description. Essayez un terme de recherche différent ou fournissez plus de détails sur la recette que vous recherchez."
        elif language == "de":
            return "Keine Rezepte gefunden, die Ihrer Beschreibung entsprechen. Versuchen Sie einen anderen Suchbegriff oder geben Sie mehr Details über das Rezept an, das Sie suchen."
        else:
            return "No recipes found matching your description. Please try a different search term or provide more details about the recipe you're looking for."

    # Create language-specific response parts
    if language == "nl":
        response_parts = [
            f"Gevonden {len(results)} vergelijkbare recepten:\n\n"]
        desc_prefix = "   Beschrijving: "
        score_prefix = "   Overeenkomst Score: "
    elif language == "fr":
        response_parts = [f"Trouvé {len(results)} recettes similaires:\n\n"]
        desc_prefix = "   Description: "
        score_prefix = "   Score de Correspondance: "
    elif language == "de":
        response_parts = [f"Gefunden {len(results)} ähnliche Rezepte:\n\n"]
        desc_prefix = "   Beschreibung: "
        score_prefix = "   Ähnlichkeits-Score: "
    else:  # English
        response_parts = [f"Found {len(results)} similar recipes:\n\n"]
        desc_prefix = "   Description: "
        score_prefix = "   Similarity Score: "

    for i, result in enumerate(results, 1):
        recipe_id = result.get("id", f"recipe_{i}")
        description = result.get("description", "")
        text_score = result.get("text_score", 0)
        combined_score = result.get("combined_score", text_score)

        response_parts.append(f"{i}. Recipe ID: {recipe_id}")
        response_parts.append(
            f"{desc_prefix}{description[:200]}{'...' if len(description) > 200 else ''}")
        score_pct = f"{combined_score * 100:.1f}".replace(".", ",")
        response_parts.append(f"{score_prefix}{score_pct}%")

        response_parts.append("")  # Empty line for readability

    return "\n".join(response_parts)


def format_response_in_language_with_ai(results: List[Dict[str, Any]], language: str, original_query: str) -> str:
    """Use AI to format the response in the specified language"""
    try:
        # Prepare the results data for AI formatting (scores as percentages)
        results_data = []
        for i, result in enumerate(results, 1):
            combined = result.get("combined_score", result.get("text_score", 0))
            results_data.append({
                "rank": i,
                "id": result.get("id", f"recipe_{i}"),
                "description": result.get("description", ""),
                "similarity_score": f"{combined * 100:.1f}".replace(".", ",") + "%"
            })

        # Create language-specific instructions
        language_instructions = {
            "en": "in English",
            "nl": "in Dutch (Nederlands)",
            "fr": "in French (Français)",
            "de": "in German (Deutsch)",
            "it": "in Italian (Italiano)",
            "es": "in Spanish (Español)",
            "pt": "in Portuguese (Português)",
            "da": "in Danish (Dansk)"
        }

        lang_instruction = language_instructions.get(language, "in English")

        # Create AI prompt for response formatting focused on recipe similarity
        prompt = f"""You are a helpful recipe search assistant specializing in finding similar food products and recipes. Format the following recipe search results {lang_instruction}.

Original user query: "{original_query}"
Number of similar recipes found: {len(results)}

Recipe Results:
{json.dumps(results_data, indent=2)}

Please format this as a natural, helpful response that:
1. Announces how many similar recipes were found
2. Lists each recipe with its description and similarity score only (do NOT include any feature score)
3. Uses appropriate language and tone for {lang_instruction}
4. Keeps descriptions concise but informative
5. Shows similarity scores as percentages (e.g. 57,2%) with a comma as decimal separator — they are already provided in that format
6. Focuses on recipe similarity and food product characteristics
7. Mentions key features like Color, Flavour, Stabilizer, Industry, etc. when relevant

Format the response in a clear, readable way with proper numbering and structure. Emphasize that these are similar recipes found in the database."""

        # Make AI call for response formatting
        response = query_llm(prompt, provider="openai")

        if response:
            logger.info(
                f"AI formatted response in {language} for {len(results)} recipes")
            return response
        else:
            logger.warning("AI response formatting failed, using fallback")
            return format_response_in_language(results, language)

    except Exception as e:
        logger.error(f"Error in AI response formatting: {e}")
        logger.exception("Detailed error:")
        return format_response_in_language(results, language)


class RecipeSearchAgent:
    """Agent for handling recipe search functionality"""

    def __init__(self, collection_name: str = "food_recipes_two_step"):
        """Initialize the recipe search agent"""
        self.recipe_manager = None
        self.collection_name = collection_name
        self._initialize_recipe_manager()

    def _initialize_recipe_manager(self):
        """Initialize the recipe search manager using Qdrant"""
        try:
            logger.info("Initializing Qdrant recipe search manager...")
            self.recipe_manager = QdrantRecipeManager(
                collection_name=self.collection_name,
                embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
                qdrant_host="qdrant",
                qdrant_port=6333
            )
            logger.info(
                "Qdrant recipe search manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Qdrant recipe manager: {e}")
            logger.exception("Detailed error:")

    def search_recipes(self,
                       description: str,
                       features: Optional[Union[pd.DataFrame,
                                                List[Dict[str, str]]]] = None,
                       text_top_k: int = 20,
                       final_top_k: int = 3,
                       original_query: Optional[str] = None,
                       country_filter: Optional[Union[str, List[str]]] = None,
                       version_filter: Optional[str] = None,
                       numerical_filters: Optional[Dict[str, Dict[str, Any]]] = None,
                       categorical_filters: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, str, Optional[Dict[str, Any]]]:
        """
        Search for similar recipes based on description and optional features

        Args:
            description: Recipe description (mandatory) - used for semantic search
            features: Optional DataFrame or list of dicts with 'charactDescr' and 'valueCharLong'
            text_top_k: Number of candidates from text search
            final_top_k: Final number of results to return
            original_query: Original user query (used for language detection if provided)
            country_filter: Optional country name(s) to filter results. Can be a single string or a list of strings. None or "All" means no filter.
            version_filter: Optional version filter (P, L, Missing, or "All" means no filter)
            numerical_filters: Optional dict mapping field codes to Qdrant range filters
                Example: {"Z_BRIX": {"gt": 40}, "Z_FRUCHTG": {"gte": 30}}
            categorical_filters: Optional dict mapping field codes to Qdrant match filters
                Example: {"Z_INH04": {"value": "No"}, "Z_INH01H": {"value": "Yes"}}

        Returns:
            Tuple of (results, metadata, formatted_response, detected_language, comparison_table)
        """
        try:
            if not self.recipe_manager:
                logger.error("Recipe manager not initialized")
                return [], {"error": "Recipe service not available"}, "Recipe service not available", "en", None

            # Validate input
            if not description.strip():
                logger.warning("Empty description provided")
                return [], {"error": "Recipe description is required"}, "Recipe description is required", "en", None

            # Detect language from the original user query (if provided) to preserve user's language intent
            # This is important because the description may have been translated/extracted to English
            language_source = original_query if original_query else description
            detected_language = detect_language_with_ai(language_source)
            logger.info(
                f"AI detected language: {detected_language} for query: '{language_source[:100]}...'")

            logger.info(
                f"Searching recipes for description: '{description[:100]}...'")

            # Prepare query DataFrame if features are provided
            query_df = None
            if features is not None:
                try:
                    # Check if features is a DataFrame or list
                    if isinstance(features, pd.DataFrame):
                        if not features.empty:
                            query_df = features
                            logger.info(
                                f"Using {len(features)} features for refinement")
                    elif isinstance(features, list) and len(features) > 0:
                        # Convert list of dicts to DataFrame
                        features_data = []
                        for feature in features:
                            if 'charactDescr' in feature and 'valueCharLong' in feature:
                                features_data.append({
                                    'charactDescr': feature['charactDescr'],
                                    'valueCharLong': feature['valueCharLong']
                                })
                        if features_data:
                            query_df = pd.DataFrame(features_data)
                            logger.info(
                                f"Using {len(features_data)} features for refinement")
                except Exception as e:
                    logger.warning(f"Error processing features: {e}")
                    query_df = None

            # Log numerical filters if present
            if numerical_filters:
                logger.info(f"Applying {len(numerical_filters)} numerical range filter(s) to search:")
                for field_code, range_spec in numerical_filters.items():
                    logger.info(f"  - {field_code}: {range_spec}")
            
            # Log categorical filters if present
            if categorical_filters:
                logger.info(f"Applying {len(categorical_filters)} categorical exact-match filter(s) to search:")
                for field_code, match_spec in categorical_filters.items():
                    logger.info(f"  - {field_code}: {match_spec}")
            
            # Run two-step search using Qdrant
            results, metadata = self.recipe_manager.search_two_step(
                text_description=description,
                query_df=query_df,
                text_top_k=text_top_k,
                final_top_k=final_top_k,
                country_filter=country_filter,
                version_filter=version_filter,
                original_query=original_query,
                numerical_filters=numerical_filters,
                categorical_filters=categorical_filters
            )

            # Format response in the detected language using AI
            formatted_response = format_response_in_language_with_ai(
                results, detected_language, description)

            # Create comparison table for top 3 recipes with translation to detected language
            # Pass constraints so we can compute per-recipe differences
            comparison_table = create_comparison_table(
                results, detected_language,
                numerical_filters=numerical_filters,
                categorical_filters=categorical_filters)

            logger.info(f"Found {len(results)} recipes")
            return results, metadata, formatted_response, detected_language, comparison_table

        except Exception as e:
            logger.error(f"Error in recipe search: {e}")
            logger.exception("Detailed error:")
            return [], {"error": f"Error searching recipes: {str(e)}"}, f"Error searching recipes: {str(e)}", "en", None

    def get_service_status(self) -> Dict[str, Any]:
        """Get the status of the recipe search service"""
        if not self.recipe_manager:
            return {
                "status": "unavailable",
                "message": "Recipe service not initialized",
                "total_recipes": 0
            }

        try:
            # Get collection info from Qdrant
            collection_info = self.recipe_manager.qdrant_client.get_collection(
                self.collection_name)
            total_recipes = collection_info.points_count

            return {
                "status": "available",
                "message": "Recipe service is running",
                "total_recipes": total_recipes,
                "collection_name": self.collection_name,
                "search_capability": "two_step_search",
                "qdrant_connection": "active"
            }
        except Exception as e:
            logger.error(f"Error getting recipe service status: {e}")
            return {
                "status": "error",
                "message": f"Error getting service status: {str(e)}",
                "total_recipes": 0
            }

    def generate_followup_questions(self, search_results: List[Dict[str, Any]], original_query: str = "", language: str = "en") -> List[str]:
        """Generate recipe-specific follow-up questions - always returns the 3 specific top N questions in the detected language"""
        # Translations for the follow-up questions
        # The key format must be preserved for backend detection: "show me the next top N"
        translations = {
            "fr": [
                "Montrez-moi les 10 meilleures recettes suivantes",
                "Montrez-moi les 25 meilleures recettes suivantes",
                "Montrez-moi les 50 meilleures recettes suivantes"
            ],
            "de": [
                "Zeige mir die nächsten Top 10 Rezepte",
                "Zeige mir die nächsten Top 25 Rezepte",
                "Zeige mir die nächsten Top 50 Rezepte"
            ],
            "nl": [
                "Toon mij de volgende top 10 recepten",
                "Toon mij de volgende top 25 recepten",
                "Toon mij de volgende top 50 recepten"
            ],
            "it": [
                "Mostrami le prossime 10 migliori ricette",
                "Mostrami le prossime 25 migliori ricette",
                "Mostrami le prossime 50 migliori ricette"
            ],
            "es": [
                "Muéstrame las siguientes 10 mejores recetas",
                "Muéstrame las siguientes 25 mejores recetas",
                "Muéstrame las siguientes 50 mejores recetas"
            ],
        }
        
        lang_code = language.lower()[:2] if language else "en"
        
        if lang_code in translations:
            return translations[lang_code]
        
        # Default: English
        return [
            "Show me the next top 10 recipes",
            "Show me the next top 25 recipes",
            "Show me the next top 50 recipes"
        ]
