#!/usr/bin/env python3
"""
Numerical Constraint Parser for Recipe Briefs

Parses numerical constraints from supplier briefs and converts them to
Qdrant range filter format.

Handles edge cases:
- Greater than: ">30%", ">50%", "more than 30"
- Less than: "<4.1", "<20ppm", "less than 4.1"
- Range with dash: "6-9", "45-55 days", "4-6°C"
- Range with tolerance: "30+/-5°", "30±5", "30 ± 5"
- Maximum: "12mm max", "max 12mm", "maximum 12"
- Minimum: "Min 3", "Mind. 7", "minimum 3"
- Approximate: "ca. 70", "~70", "approximately 70"
- European decimal: "4,5" → 4.5
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NumericalConstraint:
    """Represents a numerical constraint for Qdrant filtering"""
    field_code: str          # Z_BRIX, Z_PH, etc.
    field_name_en: str       # English description
    operator: str            # 'gt', 'gte', 'lt', 'lte', 'range', 'eq'
    value: Optional[float] = None      # Single value for gt/gte/lt/lte/eq
    min_value: Optional[float] = None  # Min for range
    max_value: Optional[float] = None  # Max for range
    original_text: str = ""            # Original text from brief
    
    def to_qdrant_filter(self) -> Dict[str, Any]:
        """Convert to Qdrant range filter format"""
        if self.operator == 'gt':
            return {"gt": self.value}
        elif self.operator == 'gte':
            return {"gte": self.value}
        elif self.operator == 'lt':
            return {"lt": self.value}
        elif self.operator == 'lte':
            return {"lte": self.value}
        elif self.operator == 'range':
            result = {}
            if self.min_value is not None:
                result["gte"] = self.min_value
            if self.max_value is not None:
                result["lte"] = self.max_value
            return result
        elif self.operator == 'eq':
            # For exact match, use small range around value
            return {"gte": self.value - 0.01, "lte": self.value + 0.01}
        return {}


# =============================================================================
# FIELD MAPPINGS: Brief field names → 60 Specified Field Codes
# =============================================================================

# Mapping from common brief field names to Z_* field codes
BRIEF_FIELD_TO_CODE: Dict[str, str] = {
    # Brix
    'brix': 'Z_BRIX',
    'brix value': 'Z_BRIX',
    'brix content': 'Z_BRIX',
    '°brix': 'Z_BRIX',
    'grad brix': 'Z_BRIX',
    'brixwert': 'Z_BRIX',
    
    # pH
    'ph': 'Z_PH',
    'ph value': 'Z_PH',
    'ph/acidity': 'Z_PH',
    'ph-wert': 'Z_PH',
    'säuregehalt': 'Z_PH',
    'acidity': 'Z_PH',
    'acidité': 'Z_PH',
    
    # Fruit content
    'fruit content': 'Z_FRUCHTG',
    'fruit': 'Z_FRUCHTG',
    'amount of fruit': 'Z_FRUCHTG',
    'fruit amount': 'Z_FRUCHTG',
    'fruchtgehalt': 'Z_FRUCHTG',
    'menge an obst': 'Z_FRUCHTG',
    'fruit percentage': 'Z_FRUCHTG',
    '% fruit': 'Z_FRUCHTG',
    'fruit content %': 'Z_FRUCHTG',
    
    # Viscosity variants
    'viscosity': 'Z_VISK20S',
    'viscosity 20°c': 'Z_VISK20S',
    'viscosity (20°c 60 seconds)': 'Z_VISK20S',
    'viskosität': 'Z_VISK20S',
    'viskosität 20°c': 'Z_VISK20S',
    'viscosity 7°c': 'Z_VISK20S_7C',
    'viscosity 30s': 'Z_VISK30S',
    'viscosity 60s': 'Z_VISK60S',
    'haake viscosity': 'Z_VISKHAAKE',
    'brookfield viscosity': 'ZMX_DD102',
    
    # Fat
    'fat content': 'Z_FETTST',
    'fat': 'Z_FETTST',
    'fat level': 'Z_FETTST',
    'fat %': 'Z_FETTST',
    'fat content (%)': 'Z_FETTST',
    'fettgehalt': 'Z_FETTST',
    'fettstufe': 'Z_FETTST',
    'fett': 'Z_FETTST',
    'fettkonzept': 'Z_FETTST',
    'fett i. tr.': 'Z_FETTST',
    
    # Protein
    'protein': 'Z_PROT',
    'protein content': 'Z_PROT',
    'protein (%)': 'Z_PROT',
    'protein content (%)': 'Z_PROT',
    'proteingehalt': 'Z_PROT',
    'eiweißgehalt': 'Z_PROT',
    
    # Sugar
    'sugar': 'Z_ZUCKER',
    'sugar content': 'Z_ZUCKER',
    'sugar (%)': 'Z_ZUCKER',
    'sugar content (%)': 'Z_ZUCKER',
    'zuckergehalt': 'Z_ZUCKER',
    'zucker': 'Z_ZUCKER',
    
    # Salt
    'salt': 'Z_SALZ',
    'salt content': 'Z_SALZ',
    'salzgehalt': 'Z_SALZ',
    'salz': 'Z_SALZ',
    
    # Dosage
    'dosage': 'Z_DOSIER',
    'dosierung': 'Z_DOSIER',
    'dosage %': 'Z_DOSIER',
    
    # Water activity
    'water activity': 'Z_FGAW',
    'aw': 'Z_FGAW',
    'aw value': 'Z_FGAW',
    'wasseraktivität': 'Z_FGAW',
    
    # Particle/pieces
    'particles size': 'Z_FLST',
    'particle size': 'Z_FLST',
    'partikelgröße': 'Z_FLST',
    
    # NOTE: The following fields are commonly requested but NOT in the 60 specified fields:
    # - Shelf life / Haltbarkeit / Mindesthaltbarkeit: Not indexed as Z_* field
    #   The data exists in MaterialMasterASegment as 'mhdhb' but isn't in Qdrant schema
    #   These constraints will be captured in text_description for semantic search instead
}

# Field codes with their English descriptions (for output)
FIELD_CODE_INFO: Dict[str, Dict[str, str]] = {
    'Z_BRIX': {'en': 'Brix', 'de': 'Brix'},
    'Z_PH': {'en': 'pH', 'de': 'PH'},
    'ZM_PH': {'en': 'pH AFM', 'de': 'PH AFM'},
    'Z_FRUCHTG': {'en': 'Fruit content', 'de': 'Fruchtgehalt'},
    'ZMX_DD108': {'en': 'Fruit Content', 'de': 'Fruchtgehalt'},
    'Z_VISK20S': {'en': 'Viscosity 20s (20°C)', 'de': 'Viskosität 20s (20°C)'},
    'Z_VISK20S_7C': {'en': 'Viscosity 20s (7°C)', 'de': 'Viskosität 20s (7°C)'},
    'Z_VISK30S': {'en': 'Viscosity 30s', 'de': 'Viskosität 30s'},
    'Z_VISK60S': {'en': 'Viscosity 60s', 'de': 'Viskosität 60s'},
    'Z_VISKHAAKE': {'en': 'Viscosity HAAKE', 'de': 'Viskosität HAAKE'},
    'ZMX_DD102': {'en': 'Brookfield Viscosity', 'de': 'Brookfield Viskosität'},
    'ZMX_DD103': {'en': 'Haake Viscosity', 'de': 'Haake Viskosität'},
    'Z_FETTST': {'en': 'Fat level', 'de': 'Fettstufe'},
    'Z_PROT': {'en': 'Protein content', 'de': 'Proteingehalt'},
    'Z_ZUCKER': {'en': 'Sugar', 'de': 'Zucker'},
    'Z_SALZ': {'en': 'Salt', 'de': 'Salz'},
    'Z_DOSIER': {'en': 'Dosage', 'de': 'Dosierung'},
    'Z_FGAW': {'en': 'Water activity', 'de': 'Wasseraktivität'},
    'ZM_AW': {'en': 'Water Activity AFM', 'de': 'Wasseraktivität AFM'},
    'Z_AW': {'en': 'Fruit retention %', 'de': 'Auswaschung %'},
    'ZMX_DD109': {'en': '% Fruit Identity', 'de': '% Dosierung'},
}


# =============================================================================
# PARSING UTILITIES
# =============================================================================

def normalize_decimal(value_str: str) -> str:
    """
    Normalize European decimal format (4,5) to standard format (4.5)
    """
    # Handle European format: "4,5" → "4.5"
    # But be careful not to convert "4,500" (thousands separator)
    value_str = value_str.strip()
    
    # If there's a comma followed by 1-2 digits at the end, it's likely a decimal
    if re.search(r',\d{1,2}$', value_str):
        value_str = value_str.replace(',', '.')
    
    return value_str


def extract_number(text: str) -> Optional[float]:
    """
    Extract a single number from text, handling various formats
    """
    text = normalize_decimal(text)
    
    # Remove common units and suffixes
    text = re.sub(r'[%°℃cm²mm³ml/kg]', '', text, flags=re.IGNORECASE)
    text = text.strip()
    
    # Try to find a number (including negative and decimal)
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass
    return None


def parse_constraint_text(text: str) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Parse constraint text and return (operator, value1, value2)
    
    Returns:
        Tuple of (operator, primary_value, secondary_value)
        - 'gt'/'gte'/'lt'/'lte': (op, value, None)
        - 'range': (op, min_value, max_value)
        - 'eq': (op, value, None)
    """
    text = normalize_decimal(text.lower().strip())
    
    # Pattern: "58% bzw. 60%" or "58 bzw. 60" or "58 or 60" or "58/60" (alternatives)
    # German "bzw." = "beziehungsweise" = "or/respectively"
    # This should be treated as a range from min to max
    match = re.search(r'(\d+\.?\d*)\s*%?\s*(?:bzw\.?|or|/)\s*(\d+\.?\d*)\s*%?', text)
    if match:
        val1 = float(match.group(1))
        val2 = float(match.group(2))
        min_val = min(val1, val2)
        max_val = max(val1, val2)
        return ('range', min_val, max_val)
    
    # Pattern: ">30%" or "> 30" or ">30" (can appear anywhere in text)
    match = re.search(r'>\s*(\d+\.?\d*)', text)
    if match and not re.search(r'>=', text):  # Make sure it's not >=
        return ('gt', float(match.group(1)), None)
    
    # Pattern: ">=30" or ">= 30"
    match = re.search(r'>=\s*(\d+\.?\d*)', text)
    if match:
        return ('gte', float(match.group(1)), None)
    
    # Pattern: "<4.1" or "< 4.1" or "<4,1" (can appear anywhere in text)
    match = re.search(r'<\s*(\d+\.?\d*)', text)
    if match and not re.search(r'<=', text):  # Make sure it's not <=
        return ('lt', float(match.group(1)), None)
    
    # Pattern: "<=4.1" or "<= 4.1"
    match = re.search(r'<=\s*(\d+\.?\d*)', text)
    if match:
        return ('lte', float(match.group(1)), None)
    
    # Pattern: "30+/-5" or "30±5" or "30 ± 5" or "30+-5"
    # Use more specific pattern to avoid false positives
    match = re.search(r'(\d+\.?\d*)\s*(?:±|\+/-|\+-|\+/\-)\s*(\d+\.?\d*)', text)
    if match:
        base = float(match.group(1))
        tolerance = float(match.group(2))
        return ('range', base - tolerance, base + tolerance)
    
    # Pattern: "6-9" or "6 - 9" or "6–9" (en-dash)
    # But not "-5" (negative number) and not "+/-5" patterns
    # Make sure we're matching "number-number" where the dash is surrounded by digits
    match = re.search(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)(?!\d)', text)
    if match:
        min_val = float(match.group(1))
        max_val = float(match.group(2))
        # Sanity check: min should be less than max for valid ranges
        if min_val < max_val:
            return ('range', min_val, max_val)
    
    # Pattern: "max 12" or "maximum 12" or "12mm max" or "12 max" or "maximal 12"
    # Check for "max" keyword with number
    match = re.search(r'max(?:imum|imal)?\.?\s*(\d+\.?\d*)', text)
    if match:
        value = float(match.group(1))
        return ('lte', value, None)
    # Check for "number max" pattern (e.g., "12mm max", "12 max")
    match = re.search(r'(\d+\.?\d*)\s*(?:mm|cm|°|%)?\s*max\b', text)
    if match:
        value = float(match.group(1))
        return ('lte', value, None)
    
    # Pattern: "min 3" or "minimum 3" or "Mind. 7" (German)
    match = re.search(r'(?:min(?:imum|d)?\.?\s*(\d+\.?\d*))|(?:(\d+\.?\d*)\s*min)', text)
    if match:
        value = float(match.group(1) or match.group(2))
        return ('gte', value, None)
    
    # Pattern: "more than 30" or "greater than 30" or "above 30" or "über 30" or "plus de 30"
    match = re.search(r'(?:more|greater|above|over|über|plus\s*de)\s*(?:than)?\s*(\d+\.?\d*)', text)
    if match:
        return ('gt', float(match.group(1)), None)
    
    # Pattern: "less than 4.1" or "below 4.1" or "under 4.1" or "unter 4.1" or "moins de 4.1"
    match = re.search(r'(?:less|below|under|unter|moins\s*de)\s*(?:than)?\s*(\d+\.?\d*)', text)
    if match:
        return ('lt', float(match.group(1)), None)
    
    # Pattern: "at least 30" or "mindestens 30"
    match = re.search(r'(?:at\s*least|mindestens)\s*(\d+\.?\d*)', text)
    if match:
        return ('gte', float(match.group(1)), None)
    
    # Pattern: "at most 30" or "höchstens 30"
    match = re.search(r'(?:at\s*most|höchstens|maximal)\s*(\d+\.?\d*)', text)
    if match:
        return ('lte', float(match.group(1)), None)
    
    # Pattern: "ca. 70" or "~70" or "approximately 70" or "about 70" or "roughly 70"
    match = re.search(r'(?:ca\.?|~|approximately|about|circa|ungefähr|roughly|environ)\s*(\d+\.?\d*)', text)
    if match:
        value = float(match.group(1))
        # Approximate → ±10% range
        return ('range', value * 0.9, value * 1.1)
    
    # Fallback: just extract a single number as exact match
    num = extract_number(text)
    if num is not None:
        return ('eq', num, None)
    
    return ('unknown', None, None)


def identify_field_from_context(text: str) -> Optional[str]:
    """
    Try to identify which Z_* field the constraint refers to
    """
    text_lower = text.lower()
    
    # Check each mapping
    for brief_name, field_code in BRIEF_FIELD_TO_CODE.items():
        if brief_name in text_lower:
            return field_code
    
    return None


def parse_numerical_constraints_from_brief(brief_text: str) -> List[NumericalConstraint]:
    """
    Extract all numerical constraints from a supplier brief text.
    
    Args:
        brief_text: The full brief text
        
    Returns:
        List of NumericalConstraint objects
    """
    constraints = []
    
    # Common patterns to find field:value pairs
    # Pattern 1: "Field Name: value" or "Field Name | value"
    field_value_patterns = [
        r'([A-Za-z\s/\(\)]+):\s*([<>≤≥±~]?\s*\d+[,.]?\d*\s*(?:[+/-]+\s*\d+[,.]?\d*)?%?)',
        r'([A-Za-z\s/\(\)]+)\|\s*([<>≤≥±~]?\s*\d+[,.]?\d*\s*(?:[+/-]+\s*\d+[,.]?\d*)?%?)',
    ]
    
    for pattern in field_value_patterns:
        matches = re.finditer(pattern, brief_text, re.IGNORECASE)
        for match in matches:
            field_name = match.group(1).strip()
            value_text = match.group(2).strip()
            
            # Identify the Z_* field code
            field_code = identify_field_from_context(field_name)
            
            if field_code:
                # Parse the constraint
                operator, val1, val2 = parse_constraint_text(value_text)
                
                if operator != 'unknown':
                    field_info = FIELD_CODE_INFO.get(field_code, {'en': field_code})
                    
                    constraint = NumericalConstraint(
                        field_code=field_code,
                        field_name_en=field_info.get('en', field_code),
                        operator=operator,
                        value=val1 if operator not in ['range'] else None,
                        min_value=val1 if operator == 'range' else None,
                        max_value=val2 if operator == 'range' else None,
                        original_text=f"{field_name}: {value_text}"
                    )
                    constraints.append(constraint)
                    logger.info(f"Extracted constraint: {constraint}")
    
    return constraints


def constraints_to_qdrant_filters(constraints: List[NumericalConstraint]) -> Dict[str, Dict[str, Any]]:
    """
    Convert a list of NumericalConstraint objects to Qdrant filter format.
    
    Returns:
        Dict mapping field_code to Qdrant range filter
        Example: {"Z_BRIX": {"gt": 40}, "Z_FRUCHTG": {"gte": 30}}
    """
    filters = {}
    
    for constraint in constraints:
        qdrant_filter = constraint.to_qdrant_filter()
        if qdrant_filter:
            filters[constraint.field_code] = qdrant_filter
    
    return filters


# =============================================================================
# LLM-BASED EXTRACTION (for more complex briefs)
# =============================================================================

def generate_extraction_prompt_for_llm() -> str:
    """
    Generate a prompt for LLM-based extraction of numerical constraints.
    """
    fields_list = "\n".join([
        f"- {code}: {info.get('en', code)}"
        for code, info in FIELD_CODE_INFO.items()
    ])
    
    return f"""Extract numerical constraints from the brief and map them to these database fields:

{fields_list}

For each numerical constraint found, extract:
1. field_code: The Z_* code from the list above
2. operator: One of "gt" (>), "gte" (>=), "lt" (<), "lte" (<=), "range" (min-max), "eq" (exact)
3. value: The numerical value (for gt/gte/lt/lte/eq)
4. min_value: Minimum value (for range)
5. max_value: Maximum value (for range)

Examples:
- "Brix: >40" → {{"field_code": "Z_BRIX", "operator": "gt", "value": 40}}
- "pH: <4.1" → {{"field_code": "Z_PH", "operator": "lt", "value": 4.1}}
- "Fruit content: >30%" → {{"field_code": "Z_FRUCHTG", "operator": "gt", "value": 30}}
- "Brix: 30+/-5°" → {{"field_code": "Z_BRIX", "operator": "range", "min_value": 25, "max_value": 35}}
- "Viscosity: 6-9" → {{"field_code": "Z_VISK20S", "operator": "range", "min_value": 6, "max_value": 9}}
- "Particle size: 12mm max" → {{"field_code": "Z_FLST", "operator": "lte", "value": 12}}
- "Fat: ca. 58%" → {{"field_code": "Z_FETTST", "operator": "range", "min_value": 52, "max_value": 64}}

Return a JSON array of constraints. If no numerical constraints found, return [].
"""


# =============================================================================
# CONVENIENCE FUNCTIONS FOR TESTS
# =============================================================================

# Alias for tests - maps common brief terms to Z_* codes
NUMERICAL_FIELD_MAPPINGS: Dict[str, str] = {
    **{k.lower(): v for k, v in BRIEF_FIELD_TO_CODE.items()},
    # Add additional mappings with original case
    'Brix': 'Z_BRIX',
    'pH': 'Z_PH',
    'Fruit content': 'Z_FRUCHTG',
    'Fruchtgehalt': 'Z_FRUCHTG',
    'Viscosity': 'Z_VISK20S',
    'Viskosität': 'Z_VISK20S',
    'Dosage': 'Z_DOSIER',
    'Dosierung': 'Z_DOSIER',
    'Sugar': 'Z_ZUCKER',
    'Zucker': 'Z_ZUCKER',
    'Fat': 'Z_FETTST',
    'Protein': 'Z_PROT',
    'Salt': 'Z_SALZ',
    'Water activity': 'Z_FGAW',
}


def parse_numerical_constraint(text: str) -> Optional[Dict[str, float]]:
    """
    Parse a numerical constraint from text and return Qdrant filter format.
    
    This is a convenience function that returns just the filter dict.
    
    Args:
        text: Constraint text like ">30%", "<4.1", "6-9", "30+/-5°", etc.
        
    Returns:
        Dict in Qdrant format: {"gt": 30.0}, {"gte": 25.0, "lte": 35.0}, etc.
        Returns None if parsing fails or text is a placeholder.
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Skip placeholder values
    placeholders = ['n/a', 'n.a.', 'tbd', 'x', '-', 'to be confirmed', 
                   'to be determined', 'na', 'none', '']
    if text.lower() in placeholders:
        return None
    
    # Skip non-numeric text (no digits at all)
    if not any(c.isdigit() for c in text):
        return None
    
    operator, val1, val2 = parse_constraint_text(text)
    
    if operator == 'unknown' or val1 is None:
        return None
    
    if operator == 'gt':
        return {"gt": val1}
    elif operator == 'gte':
        return {"gte": val1}
    elif operator == 'lt':
        return {"lt": val1}
    elif operator == 'lte':
        return {"lte": val1}
    elif operator == 'range':
        result = {}
        if val1 is not None:
            result["gte"] = val1
        if val2 is not None:
            result["lte"] = val2
        return result if result else None
    elif operator == 'eq':
        return {"eq": val1}
    
    return None


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ">30%",
        "<4.1",
        "30+/-5°",
        "30 ± 5",
        "6-9",
        "45-55 days",
        "12mm max",
        "max 12mm",
        "Min 3 months",
        "Mind. 7",
        "ca. 70",
        "~70",
        "4,5",  # European decimal
        "more than 30",
        "less than 4.1",
        "at least 30%",
        "approximately 70",
        # Alternative values (German "bzw." = or/respectively)
        "58% bzw. 60%",
        "58 bzw. 60",
        "58 or 60",
        "58/60%",
    ]
    
    print("Testing numerical constraint parser:")
    print("=" * 60)
    
    for text in test_cases:
        operator, val1, val2 = parse_constraint_text(text)
        if operator == 'range':
            print(f"'{text}' → {operator}: [{val1}, {val2}]")
        else:
            print(f"'{text}' → {operator}: {val1}")
