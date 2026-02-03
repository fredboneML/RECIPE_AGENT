#!/usr/bin/env python3
"""
Categorical Constraint Parser for Recipe Briefs

Parses categorical/binary constraints from supplier briefs and converts them to
Qdrant filter format for exact matching.

Binary fields have Yes/No values that can be used as exact match filters.
Examples:
- "No preservatives" → Z_INH04: "No" (Nicht konserviert)
- "Halal required" → Z_INH01H: "Yes" (suitable HALAL)
- "No artificial colors" → Z_INH05: "No" (keine künstl. Farbe)
- "With starch" → Z_INH13: "Yes" (Stärke enthalten)
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CategoricalConstraint:
    """Represents a categorical/binary constraint for Qdrant filtering"""
    field_code: str          # Z_INH04, Z_INH05, etc.
    field_name_en: str       # English description
    value: str               # "Yes" or "No"
    original_text: str = ""  # Original text from brief
    
    def to_qdrant_filter(self) -> Dict[str, Any]:
        """Convert to Qdrant match filter format"""
        return {"value": self.value}


# =============================================================================
# BINARY FIELD MAPPINGS: Brief field names → Z_* Field Codes
# =============================================================================
# These are the binary fields from the 60 specified fields that can be used
# for exact match filtering

BINARY_FIELD_INFO: Dict[str, Dict[str, Any]] = {
    # Sweetener-related
    'Z_INH02': {
        'en': 'Sweetener',
        'de': 'Süßstoff',
        'keywords_yes': ['sweetener', 'süßstoff', 'süssstoff', 'with sweetener', 'mit süßstoff'],
        'keywords_no': ['no sweetener', 'keine süsstoffe', 'ohne süßstoff', 'sweetener-free', 'without sweetener'],
    },
    'Z_INH03': {
        'en': 'Saccharose',
        'de': 'Saccharose',
        'keywords_yes': ['saccharose', 'sucrose', 'with saccharose', 'mit saccharose', 'sugar', 'zucker'],
        'keywords_no': ['no saccharose', 'keine saccharose', 'sugar-free', 'zuckerfrei', 'no added sugar', 'ohne zucker'],
        # Note: Some indexed recipes have "Saccharose" as the value instead of "Yes"
        'yes_values': ['Yes', 'Saccharose'],  # Accept both normalized and original values
    },
    'Z_INH19': {
        'en': 'Aspartame',
        'de': 'Aspartam',
        'keywords_yes': ['aspartame', 'aspartam', 'with aspartame'],
        'keywords_no': ['no aspartame', 'kein aspartam', 'aspartame-free', 'without aspartame'],
    },
    
    # Preservation
    'Z_INH04': {
        'en': 'Preserved',
        'de': 'Konservierung',
        'keywords_yes': ['preserved', 'with preservative', 'konserviert', 'mit konservierung'],
        'keywords_no': ['no preservative', 'preservative-free', 'nicht konserviert', 'ohne konservierung', 
                       'no preservatives', 'preservative free', 'unpreserved'],
    },
    
    # Colors
    'Z_INH18': {
        'en': 'Color',
        'de': 'Farbe',
        'keywords_yes': ['with color', 'colored', 'farbe enthalten', 'mit farbe', 'coloring'],
        'keywords_no': ['no color', 'keine farbe', 'uncolored', 'color-free', 'ohne farbe', 'no coloring'],
    },
    'Z_INH05': {
        'en': 'Artificial colors',
        'de': 'Künstliche Farben',
        'keywords_yes': ['artificial color', 'artificial colours', 'künstliche farbe', 'synthetic color'],
        'keywords_no': ['no artificial color', 'no artificial colours', 'keine künstl. farbe', 
                       'natural colors only', 'no synthetic color', 'artificial color free',
                       'only natural colours', 'n1 colours', 'natcol'],
    },
    
    # Flavors/Aromas
    'Z_INH09': {
        'en': 'Flavour',
        'de': 'Aroma',
        'keywords_yes': ['with flavor', 'with flavour', 'flavored', 'mit aroma', 'aromatisiert'],
        'keywords_no': ['no flavor', 'no flavour', 'unflavored', 'ohne aroma', 'flavor-free'],
    },
    'Z_INH06': {
        'en': 'Nature identical flavor',
        'de': 'naturident/künstliches Aroma',
        'keywords_yes': ['nature identical', 'naturident', 'nature-identical flavor', 'naturidentes aroma'],
        'keywords_no': ['no nature identical', 'kein naturidentes aroma', 'without nature identical',
                       'natural flavor only', 'no artificial flavor'],
    },
    'Z_INH06Z': {
        'en': 'Natural flavor',
        'de': 'Natürliche Aromen',
        'keywords_yes': ['natural flavor', 'natural flavour', 'natürliches aroma', 'natural flavouring',
                        'with natural flavor', 'only natural flavours', 'allowed flavouring agents only natural'],
        'keywords_no': ['no natural flavor', 'no flavoring', 'without flavor', 'without if possible'],
    },
    
    # Vitamins
    'Z_INH21': {
        'en': 'Vitamins',
        'de': 'Vitamine',
        'keywords_yes': ['vitamin', 'vitamine', 'with vitamins', 'fortified', 'vitamin enriched'],
        'keywords_no': ['no vitamins', 'keine vitamine', 'vitamin-free', 'not fortified'],
    },
    
    # Stabilizers
    'Z_INH13': {
        'en': 'Starch',
        'de': 'Stärke',
        'keywords_yes': ['starch', 'stärke', 'with starch', 'stärke enthalten', 'modified starch', '1442'],
        'keywords_no': ['no starch', 'starch-free', 'keine stärke', 'without starch', 'starch free'],
    },
    'Z_INH14': {
        'en': 'Pectin',
        'de': 'Pektin',
        'keywords_yes': ['pectin', 'pektin', 'with pectin', 'pektin enthalten', '440'],
        'keywords_no': ['no pectin', 'pectin-free', 'kein pektin', 'without pectin'],
    },
    'Z_INH15': {
        'en': 'LBG',
        'de': 'IBKM',
        'keywords_yes': ['lbg', 'locust bean gum', 'ibkm', 'johannisbrotkernmehl', '410'],
        'keywords_no': ['no lbg', 'kein ibkm', 'without lbg', 'lbg-free'],
    },
    'Z_INH16': {
        'en': 'Blend',
        'de': 'Mischung',
        'keywords_yes': ['blend', 'mischung', 'with blend', 'blended'],
        'keywords_no': ['no blend', 'keine mischung', 'without blend', 'single origin'],
    },
    'Z_INH20': {
        'en': 'Xanthan',
        'de': 'Xanthan',
        'keywords_yes': ['xanthan', 'with xanthan', 'xanthan enthalten', '415'],
        'keywords_no': ['no xanthan', 'kein xanthan', 'xanthan-free', 'without xanthan'],
    },
    'Z_STABGU': {
        'en': 'Stabilizing System - Guar',
        'de': 'Stabilizing System - Guar',
        'keywords_yes': ['guar', 'guar gum', 'guarkernmehl', 'with guar', '412'],
        'keywords_no': ['no guar', 'guar-free', 'ohne guar', 'without guar'],
    },
    'Z_STABCAR': {
        'en': 'Stabilizing System - Carrageen',
        'de': 'Stabilizing System - Carrageen',
        'keywords_yes': ['carrageen', 'carrageenan', 'with carrageen'],
        'keywords_no': ['no carrageen', 'carrageen-free', 'without carrageen'],
    },
    'Z_STAGEL': {
        'en': 'Stabilizing System - Gellan',
        'de': 'Stabilizing System - Gellan',
        'keywords_yes': ['gellan', 'with gellan', 'gellan gum'],
        'keywords_no': ['no gellan', 'gellan-free', 'without gellan'],
    },
    'Z_INH17': {
        'en': 'Other stabilizer',
        'de': 'Andere Stabilisatoren',
        'keywords_yes': ['other stabilizer', 'andere stabilisatoren', 'additional stabilizer'],
        'keywords_no': ['no other stabilizer', 'keine anderen stabil', 'no additional stabilizer'],
    },
    
    # Allergen-free (Yes = product is allergen-free, No = contains allergens)
    'Z_INH12': {
        'en': 'Allergen-free',
        'de': 'Allergenfrei',
        'keywords_yes': ['allergen-free', 'allergenfrei', 'no allergens', 'ohne allergene', 'allergen free', 'hypoallergenic'],
        'keywords_no': ['contains allergen', 'allergen', 'mit allergenen', 'allergenic', 'milk containing', 'has allergens'],
    },
    
    # GMO
    'Z_KNOGM': {
        'en': 'GMO presence',
        'de': 'GMO enthalten',
        'keywords_yes': ['contains gmo', 'gmo', 'genetically modified', 'nicht genfrei'],
        'keywords_no': ['gmo-free', 'gmo free', 'non-gmo', 'genfrei', 'no gmo', 'without gmo'],
    },
    'Z_INH08': {
        'en': 'Contains GMO',
        'de': 'Nicht Genfrei',
        'keywords_yes': ['contains gmo', 'genetically modified', 'nicht genfrei'],
        'keywords_no': ['gmo-free', 'gmo free', 'non-gmo', 'genfrei', 'no gmo', 'gene manipulated free'],
    },
    
    # Certifications
    'Z_INH01K': {
        'en': 'Kosher',
        'de': 'Kosher',
        'keywords_yes': ['kosher', 'koscher', 'suitable kosher', 'certified kosher', 'kosher certified',
                        'kosher preferred', 'kosher required'],
        'keywords_no': ['not kosher', 'non-kosher', 'not suitable for kosher'],
    },
    'Z_INH01H': {
        'en': 'Halal',
        'de': 'Halal',
        'keywords_yes': ['halal', 'suitable halal', 'halal certified', 'halal preferred', 'halal required'],
        'keywords_no': ['not halal', 'non-halal', 'not suitable halal'],
    },
    'Z_DAIRY': {
        'en': 'Non-Dairy Product',
        'de': 'Non-Dairy Product',
        'keywords_yes': ['non-dairy', 'dairy-free', 'plant-based', 'vegan', 'ohne milch'],
        'keywords_no': ['dairy', 'contains dairy', 'milk', 'mit milch', 'milk containing'],
    },
    
    # Other
    'Z_INH07': {
        'en': 'Extreme recipe',
        'de': 'Extremrezeptur',
        'keywords_yes': ['extreme recipe', 'extremrezeptur', 'extreme formulation'],
        'keywords_no': ['no extreme recipe', 'keine extremrezeptur', 'standard recipe'],
    },
    'Z_INH01': {
        'en': 'Standard product',
        'de': 'Standardprodukt',
        'keywords_yes': ['standard product', 'standardprodukt', 'standard'],
        'keywords_no': ['non-standard', 'custom', 'special', 'bespoke'],
    },
    'Z_BFS': {
        'en': 'Bake/Freeze Stability',
        'de': 'Bake/Freeze Stability',
        'keywords_yes': ['bake stable', 'freeze stable', 'bake/freeze stable', 'freeze-thaw stable'],
        'keywords_no': ['not bake stable', 'not freeze stable'],
    },
}

# Mapping from common brief terms to Z_* field codes
BRIEF_FIELD_TO_CODE: Dict[str, str] = {
    # Sweetener
    'sweetener': 'Z_INH02',
    'süßstoff': 'Z_INH02',
    'saccharose': 'Z_INH03',
    'sucrose': 'Z_INH03',
    'aspartame': 'Z_INH19',
    'aspartam': 'Z_INH19',
    
    # Preservation
    'preservative': 'Z_INH04',
    'preserved': 'Z_INH04',
    'konservierung': 'Z_INH04',
    'preservatives': 'Z_INH04',
    
    # Colors
    'color': 'Z_INH18',
    'colour': 'Z_INH18',
    'farbe': 'Z_INH18',
    'artificial color': 'Z_INH05',
    'artificial colour': 'Z_INH05',
    'artificial colors': 'Z_INH05',
    'artificial colours': 'Z_INH05',
    'künstliche farben': 'Z_INH05',
    'künstliche farbe': 'Z_INH05',
    
    # Flavor
    'flavor': 'Z_INH09',
    'flavour': 'Z_INH09',
    'aroma': 'Z_INH09',
    'nature identical flavor': 'Z_INH06',
    'nature identical flavour': 'Z_INH06',
    'naturidentes aroma': 'Z_INH06',
    'natural flavor': 'Z_INH06Z',
    'natural flavour': 'Z_INH06Z',
    'natural flavors': 'Z_INH06Z',
    'natural flavours': 'Z_INH06Z',
    'natürliche aromen': 'Z_INH06Z',
    'natürliches aroma': 'Z_INH06Z',
    
    # Vitamins
    'vitamins': 'Z_INH21',
    'vitamine': 'Z_INH21',
    
    # Stabilizers
    'starch': 'Z_INH13',
    'stärke': 'Z_INH13',
    'pectin': 'Z_INH14',
    'pektin': 'Z_INH14',
    'lbg': 'Z_INH15',
    'ibkm': 'Z_INH15',
    'locust bean gum': 'Z_INH15',
    'blend': 'Z_INH16',
    'mischung': 'Z_INH16',
    'xanthan': 'Z_INH20',
    'guar': 'Z_STABGU',
    'guar gum': 'Z_STABGU',
    'guarkernmehl': 'Z_STABGU',
    'carrageen': 'Z_STABCAR',
    'carrageenan': 'Z_STABCAR',
    'gellan': 'Z_STAGEL',
    'other stabilizer': 'Z_INH17',
    'andere stabilisatoren': 'Z_INH17',
    
    # Allergens
    'allergen': 'Z_INH12',
    'allergens': 'Z_INH12',
    'allergene': 'Z_INH12',
    'allergen-free': 'Z_INH12',
    'allergenfrei': 'Z_INH12',
    
    # GMO
    'gmo': 'Z_INH08',
    'gmo-free': 'Z_INH08',
    'non-gmo': 'Z_INH08',
    'genfrei': 'Z_INH08',
    
    # Certifications
    'kosher': 'Z_INH01K',
    'koscher': 'Z_INH01K',
    'halal': 'Z_INH01H',
    'dairy': 'Z_DAIRY',
    'non-dairy': 'Z_DAIRY',
    'dairy-free': 'Z_DAIRY',
    
    # Other
    'extreme recipe': 'Z_INH07',
    'extremrezeptur': 'Z_INH07',
    'standard product': 'Z_INH01',
    'standardprodukt': 'Z_INH01',
    'bake/freeze stability': 'Z_BFS',
    'freeze stable': 'Z_BFS',
    'bake stable': 'Z_BFS',
    
    # Texture (descriptive values: "Stückig" or "Flüssig")
    'texture': 'Z_FLST',
    'pieces': 'Z_FLST',
    'with pieces': 'Z_FLST',
    'stückig': 'Z_FLST',
    'flüssig': 'Z_FLST',
    'liquid': 'Z_FLST',
    'smooth': 'Z_FLST',
    'puree': 'Z_FLST',
}


def normalize_to_yes_no(value: str, field_code: Optional[str] = None) -> Optional[str]:
    """
    Normalize various Yes/No representations to standardized "Yes" or "No".
    
    Some fields in the indexed data have descriptive values instead of Yes/No.
    This function handles both cases.
    
    Args:
        value: Input value string
        field_code: Optional Z_* field code for field-specific handling
        
    Returns:
        "Yes", "No", or the appropriate indexed value
    """
    if not value:
        return None
    
    value_lower = value.lower().strip()
    
    # Special handling for fields that have descriptive values in indexed data
    # These fields store the descriptive value (e.g., "Saccharose") instead of "Yes"
    DESCRIPTIVE_VALUE_FIELDS = {
        'Z_INH03': {'yes_value': 'Saccharose', 'keywords': ['saccharose', 'sucrose', 'sugar', 'zucker']},
        'Z_FLST': {'yes_values': ['Stückig', 'Flüssig'], 'keywords': {'stückig': 'Stückig', 'with pieces': 'Stückig', 'pieces': 'Stückig', 'chunky': 'Stückig', 'flüssig': 'Flüssig', 'liquid': 'Flüssig', 'smooth': 'Flüssig', 'puree': 'Flüssig'}},
    }
    
    # Check for field-specific handling
    if field_code and field_code in DESCRIPTIVE_VALUE_FIELDS:
        field_config = DESCRIPTIVE_VALUE_FIELDS[field_code]
        
        # For fields with keyword-to-value mappings (like Z_FLST)
        if 'keywords' in field_config and isinstance(field_config['keywords'], dict):
            for keyword, indexed_value in field_config['keywords'].items():
                if keyword in value_lower:
                    return indexed_value
        # For simple yes_value fields (like Z_INH03)
        elif 'yes_value' in field_config:
            for keyword in field_config.get('keywords', []):
                if keyword in value_lower:
                    return field_config['yes_value']
    
    # Positive indicators → "Yes"
    positive_patterns = [
        r'^yes$', r'^ja$', r'^oui$', r'^si$', r'^true$', r'^1$',
        r'enthalten', r'^with\b', r'^suitable\b', r'^certified\b',
        r'^allowed$', r'^present$', r'^contains$', r'^required$',
        r'^preferred$', r'^vorhanden$', r'^mit\s',
    ]
    
    for pattern in positive_patterns:
        if re.search(pattern, value_lower):
            return "Yes"
    
    # Negative indicators → "No"
    negative_patterns = [
        r'^no$', r'^nein$', r'^non$', r'^false$', r'^0$',
        r'^kein', r'^keine', r'^nicht\b', r'^ohne\b',
        r'^no\s', r'^not\s', r'^without\b', r'-free$',
        r'free$', r'^absent$', r'^forbidden$',
    ]
    
    for pattern in negative_patterns:
        if re.search(pattern, value_lower):
            return "No"
    
    # Check specific values
    if value_lower in ['stärke enthalten', 'pektin enthalten', 'xanthan enthalten']:
        return "Yes"
    if value_lower in ['nicht konserviert', 'allergenfrei', 'genfrei']:
        return "No"
    if 'suitable' in value_lower:
        return "Yes"
    if 'not suitable' in value_lower:
        return "No"
    
    return None


def extract_categorical_from_brief_text(brief_text: str) -> List[CategoricalConstraint]:
    """
    Extract categorical constraints by scanning brief text for keywords.
    
    Args:
        brief_text: The full brief text
        
    Returns:
        List of CategoricalConstraint objects
    """
    constraints = []
    brief_lower = brief_text.lower()
    found_fields = set()  # Avoid duplicates
    
    for field_code, field_info in BINARY_FIELD_INFO.items():
        # Check negative keywords first (more specific)
        for keyword in field_info.get('keywords_no', []):
            if keyword.lower() in brief_lower:
                if field_code not in found_fields:
                    constraints.append(CategoricalConstraint(
                        field_code=field_code,
                        field_name_en=field_info['en'],
                        value="No",
                        original_text=keyword
                    ))
                    found_fields.add(field_code)
                    logger.info(f"Extracted categorical constraint: {field_code} = No (matched '{keyword}')")
                break
        
        # Check positive keywords (only if not already found as negative)
        if field_code not in found_fields:
            for keyword in field_info.get('keywords_yes', []):
                if keyword.lower() in brief_lower:
                    constraints.append(CategoricalConstraint(
                        field_code=field_code,
                        field_name_en=field_info['en'],
                        value="Yes",
                        original_text=keyword
                    ))
                    found_fields.add(field_code)
                    logger.info(f"Extracted categorical constraint: {field_code} = Yes (matched '{keyword}')")
                    break
    
    return constraints


def parse_llm_categorical_constraints(
    llm_constraints: List[Dict[str, str]]
) -> Dict[str, Dict[str, Any]]:
    """
    Parse categorical constraints from LLM output format to Qdrant filter format.
    
    LLM output format:
        [{"field": "Preservative", "value": "No"}, ...]
    
    Returns:
        Dict mapping field_code to Qdrant match filter
        {"Z_INH04": {"value": "No"}, ...}
    """
    filters = {}
    
    for item in llm_constraints:
        field_name = item.get('field', '').lower().strip()
        value = item.get('value', '').strip()
        
        if not field_name or not value:
            continue
        
        # Map field name to Z_* code first (needed for field-specific normalization)
        field_code = BRIEF_FIELD_TO_CODE.get(field_name)
        
        # Try partial matching if exact match not found
        if not field_code:
            for brief_name, code in BRIEF_FIELD_TO_CODE.items():
                if brief_name in field_name or field_name in brief_name:
                    field_code = code
                    break
        
        if not field_code:
            logger.warning(f"Could not map categorical field '{field_name}' to Z_* code")
            continue
        
        # Normalize value to Yes/No (or descriptive value for special fields)
        normalized_value = normalize_to_yes_no(value, field_code)
        if normalized_value is None:
            # Try direct match
            if value.lower() in ['yes', 'no']:
                normalized_value = value.capitalize()
            else:
                logger.warning(f"Could not normalize value '{value}' for field '{field_name}'")
                continue
        
        filters[field_code] = {"value": normalized_value}
        logger.info(f"Parsed categorical constraint: {field_name} ({field_code}) = {normalized_value}")
    
    return filters


def constraints_to_qdrant_filters(
    constraints: List[CategoricalConstraint]
) -> Dict[str, Dict[str, Any]]:
    """
    Convert a list of CategoricalConstraint objects to Qdrant filter format.
    
    Returns:
        Dict mapping field_code to Qdrant match filter
        Example: {"Z_INH04": {"value": "No"}, "Z_INH01H": {"value": "Yes"}}
    """
    filters = {}
    
    for constraint in constraints:
        qdrant_filter = constraint.to_qdrant_filter()
        if qdrant_filter:
            filters[constraint.field_code] = qdrant_filter
    
    return filters


# =============================================================================
# LLM EXTRACTION PROMPT GENERATION
# =============================================================================

def generate_categorical_extraction_prompt() -> str:
    """
    Generate the prompt section for LLM to extract categorical constraints.
    """
    fields_list = []
    for code, info in BINARY_FIELD_INFO.items():
        fields_list.append(f"- {code}: {info['en']} ({info['de']})")
    
    return f"""
CATEGORICAL/BINARY CONSTRAINTS (for exact match filtering):
These binary fields can be filtered with Yes/No values in Qdrant:

{chr(10).join(fields_list)}

Extract categorical constraints as:
"categorical_constraints": [
    {{"field": "Preservative", "value": "No"}},
    {{"field": "Artificial colors", "value": "No"}},
    {{"field": "Halal", "value": "Yes"}},
    {{"field": "Kosher", "value": "Yes"}},
    {{"field": "Natural flavor", "value": "Yes"}},
    {{"field": "Starch", "value": "Yes"}},
    {{"field": "Pectin", "value": "Yes"}}
]

RULES for categorical extraction:
1. Use "Yes" for: required, allowed, contains, with, present, suitable, certified
2. Use "No" for: not allowed, no, without, free, none, absent, forbidden
3. ONLY extract what is EXPLICITLY mentioned in the brief
4. If a field is not mentioned, do NOT include it
5. Map stabilizer codes (1442=starch, 440=pectin, 412=guar, 415=xanthan, 410=LBG) to their field names
"""


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_field_code(field_name: str) -> Optional[str]:
    """Get Z_* field code from field name."""
    return BRIEF_FIELD_TO_CODE.get(field_name.lower().strip())


def get_field_info(field_code: str) -> Optional[Dict[str, Any]]:
    """Get field info for a Z_* field code."""
    return BINARY_FIELD_INFO.get(field_code)


if __name__ == "__main__":
    # Test cases
    test_brief = """
    Peach Apricot Fruit Preparation
    
    Key Product Claims: No Preservatives, Artificial Colours or Flavours
    Allowed Stabilization System: Modified Starch (1442), Pectin (440), Guar (412), Xanthan (415), LBG (410)
    Allowed Flavouring Agents: Only Natural Flavours
    Allowed Colouring Agents: Only Natural Colours (N1)
    Allowed Allergen: Milk containing products
    Religious Certification: Halal & Kosher Preferred
    """
    
    print("Testing categorical constraint extraction:")
    print("=" * 60)
    print(f"Brief:\n{test_brief}")
    print("=" * 60)
    
    constraints = extract_categorical_from_brief_text(test_brief)
    
    print(f"\nExtracted {len(constraints)} constraints:")
    for c in constraints:
        print(f"  {c.field_code} ({c.field_name_en}): {c.value} - matched '{c.original_text}'")
    
    print("\nQdrant filters:")
    filters = constraints_to_qdrant_filters(constraints)
    for code, filter_dict in filters.items():
        print(f"  {code}: {filter_dict}")
