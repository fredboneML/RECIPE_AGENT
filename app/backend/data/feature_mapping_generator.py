#!/usr/bin/env python3
"""
Feature Mapping Generator - Creates mappings for DataExtractorRouterAgent

Generates intelligent mappings from:
- User terminology → Database charactDescr names
- User values → Database valueCharLong values
- Handles: multilingual, synonyms, case variations, abbreviations
"""
import json
import logging
from typing import Dict, List, Set, Any
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureMappingGenerator:
    """Generates feature mappings for data extraction"""
    
    def __init__(self, feature_map_path: str):
        """Initialize with charactDescr_valueCharLong_map.json"""
        self.feature_map_path = feature_map_path
        self.feature_map = self._load_map()
        
        # Output mappings
        self.feature_name_mappings = {}  # User term → charactDescr
        self.value_mappings = {}  # charactDescr → {user_value → db_value}
        
    def _load_map(self) -> Dict[str, List[str]]:
        """Load the feature map"""
        try:
            with open(self.feature_map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded feature map with {len(data)} unique features")
            return data
        except Exception as e:
            logger.error(f"Error loading feature map: {e}")
            return {}
    
    def generate_all_mappings(self) -> Dict[str, Any]:
        """Generate all mappings for data extraction"""
        logger.info("Generating feature name mappings...")
        self._generate_feature_name_mappings()
        
        logger.info("Generating value normalization mappings...")
        self._generate_value_mappings()
        
        return {
            'feature_name_mappings': self.feature_name_mappings,
            'value_mappings': self.value_mappings,
            'stats': {
                'total_features': len(self.feature_map),
                'feature_name_variants': len(self.feature_name_mappings),
                'features_with_value_mappings': len(self.value_mappings)
            }
        }
    
    def _generate_feature_name_mappings(self):
        """Generate mappings from user terms to exact charactDescr"""
        for charact_descr in self.feature_map.keys():
            # Add exact name (case-insensitive)
            self._add_mapping(charact_descr.lower(), charact_descr)
            
            # Add variations
            self._add_feature_name_variations(charact_descr)
    
    def _add_feature_name_variations(self, charact_descr: str):
        """Add common variations of feature names"""
        lower = charact_descr.lower()
        
        # Common synonyms and variations
        variations = {
            # Flavor/Flavour
            'flavor': 'Flavour',
            'flavour': 'Flavour',
            'flavour OR flavor': 'Flavour',
            'flavor OR flavour': 'Flavour',
            'aroma': 'Flavour',
            'taste': 'Flavour',
            'geschmack': 'Flavour',
            
            # Color/Colour
            'color': 'Color',
            'colour': 'Color',
            'color OR farbe': 'Color',
            'farbe': 'Color',
            'col': 'Color',
            
            # Application
            'application': 'Application (Fruit filling)',
            'application (fruit filling)': 'Application (Fruit filling)',
            'fruit prep application': 'Application (Fruit filling)',
            'use': 'Application (Fruit filling)',
            'usage': 'Application (Fruit filling)',
            'anwendung': 'Application (Fruit filling)',
            
            # Stabilizers
            'starch': 'Starch',
            'stärke': 'Starch',
            'modified starch': 'Starch',
            'pectin': 'Pectin',
            'pektin': 'Pectin',
            'xanthan': 'Xanthan',
            'xanthan gum': 'Xanthan',
            'guar': 'LBG',
            'guar gum': 'LBG',
            'locust bean gum': 'LBG',
            'guarkernmehl': 'Guarkernmehl',
            'carrageenan': 'Carrageenan',
            'carragenan': 'Carrageenan',
            
            # Certifications
            'halal': 'HALAL',
            'halal certified': 'HALAL',
            'kosher': 'KOSHER',
            'kosher certified': 'KOSHER',
            'vegan': 'VEGAN',
            'vegan certified': 'VEGAN',
            'organic': 'ORGANIC',
            'bio': 'ORGANIC',
            'organic certified': 'ORGANIC',
            
            # Colors
            'artificial colors': 'Artificial colors',
            'artificial colours': 'Artificial colors',
            'artificial coloring': 'Artificial colors',
            'synthetic colors': 'Artificial colors',
            'natural colors': 'Natural colors',
            'natural colours': 'Natural colors',
            'natural coloring': 'Natural colors',
            
            # Technical parameters
            'ph': 'pH range',
            'ph range': 'pH range',
            'ph value': 'pH range',
            'ph-wert': 'pH range',
            'brix': 'Brix range',
            'brix range': 'Brix range',
            'brix value': 'Brix range',
            'sugar content': 'Brix range',
            'acid': 'Acid content',
            'acidity': 'Acid content',
            'acid content': 'Acid content',
            'titratable acid': 'Acid content',
            
            # Product properties
            'fruit content': 'Fruit content',
            'fruit %': 'Fruit content',
            'fruit percentage': 'Fruit content',
            'fruchtanteil': 'Fruit content',
            'particulates': 'Particulates',
            'particles': 'Particulates',
            'pieces': 'Particulates',
            'stücke': 'Particulates',
        }
        
        # Add predefined variations
        for variant, target in variations.items():
            if target == charact_descr or variant in lower:
                self._add_mapping(variant, charact_descr)
        
        # Add word-based variations
        words = charact_descr.lower().split()
        
        # For multi-word features, add combinations
        if len(words) > 1:
            # First word
            self._add_mapping(words[0], charact_descr)
            # Last word
            self._add_mapping(words[-1], charact_descr)
            # Without common words
            filtered = [w for w in words if w not in ['or', 'and', 'the', '(', ')', '-']]
            if filtered:
                self._add_mapping(' '.join(filtered), charact_descr)
    
    def _add_mapping(self, user_term: str, charact_descr: str):
        """Add a mapping from user term to charactDescr"""
        user_term = user_term.lower().strip()
        if user_term and user_term not in self.feature_name_mappings:
            self.feature_name_mappings[user_term] = charact_descr
    
    def _generate_value_mappings(self):
        """Generate value normalization mappings for each feature"""
        for charact_descr, values in self.feature_map.items():
            if not values:
                continue
            
            value_map = {}
            unique_values = list(set(str(v).strip() for v in values if v and str(v).strip()))
            
            # Check if this is a binary feature
            if self._is_binary_feature(unique_values):
                value_map.update(self._get_binary_value_mappings(unique_values))
            
            # Check if this is a numerical feature with ranges
            elif self._has_range_pattern(unique_values):
                # For ranges, we'll keep exact values
                for val in unique_values:
                    value_map[val.lower()] = val
            
            # For categorical features
            else:
                # Map lowercase and variations to exact values
                for val in unique_values[:100]:  # Limit to avoid huge mappings
                    value_map[val.lower()] = val
                    # Add without spaces/hyphens
                    clean = val.replace(' ', '').replace('-', '').lower()
                    if clean != val.lower():
                        value_map[clean] = val
            
            if value_map:
                self.value_mappings[charact_descr] = value_map
    
    def _is_binary_feature(self, values: List[str]) -> bool:
        """Check if feature is binary"""
        if len(values) > 10:
            return False
        
        values_lower = set(v.lower() for v in values)
        
        # Check for binary patterns
        binary_patterns = [
            {'yes', 'no'},
            {'ja', 'nein'},
            {'oui', 'non'},
            {'allowed', 'not allowed'},
            {'true', 'false'},
        ]
        
        for pattern in binary_patterns:
            if pattern.issubset(values_lower):
                return True
        
        return False
    
    def _get_binary_value_mappings(self, values: List[str]) -> Dict[str, str]:
        """Get comprehensive mappings for binary values"""
        # Find the canonical positive and negative values
        positive_canonical = None
        negative_canonical = None
        
        values_lower = [v.lower() for v in values]
        
        # Identify canonical values
        for val in values:
            val_lower = val.lower()
            if val_lower in ['yes', 'ja', 'oui', 'si', 'allowed', 'true']:
                if not positive_canonical or val_lower == 'yes':
                    positive_canonical = val
            elif val_lower in ['no', 'nein', 'non', 'not allowed', 'false']:
                if not negative_canonical or val_lower == 'no':
                    negative_canonical = val
        
        # Create comprehensive mappings
        mappings = {}
        
        # Positive variants → positive canonical
        if positive_canonical:
            positive_variants = [
                'yes', 'ja', 'oui', 'si', 'sim', 'da',
                'allowed', 'permitted', 'erlaubt', 'autorisé',
                'true', '1', 'active', 'present'
            ]
            for variant in positive_variants:
                mappings[variant] = positive_canonical
        
        # Negative variants → negative canonical
        if negative_canonical:
            negative_variants = [
                'no', 'nein', 'non', 'não', 'nee',
                'not allowed', 'forbidden', 'verboten', 'interdit',
                'false', '0', 'inactive', 'absent'
            ]
            for variant in negative_variants:
                mappings[variant] = negative_canonical
        
        return mappings
    
    def _has_range_pattern(self, values: List[str]) -> bool:
        """Check if values contain ranges"""
        import re
        range_pattern = r'\d+\.?\d*\s*[-–]\s*\d+\.?\d*'
        
        range_count = sum(1 for v in values[:20] if re.match(range_pattern, v))
        return range_count / min(len(values), 20) > 0.3
    
    def save_mappings(self, output_path: str):
        """Save mappings to JSON file"""
        mappings = self.generate_all_mappings()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Mappings saved to {output_path}")
            logger.info(f"   Feature name mappings: {len(mappings['feature_name_mappings'])}")
            logger.info(f"   Features with value mappings: {len(mappings['value_mappings'])}")
        except Exception as e:
            logger.error(f"Error saving mappings: {e}")
    
    def print_summary(self):
        """Print summary of generated mappings"""
        print("\n" + "=" * 80)
        print("FEATURE MAPPING SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Total Features in Database: {len(self.feature_map)}")
        print(f"\n🔤 Feature Name Mappings: {len(self.feature_name_mappings)}")
        print("   Examples:")
        for i, (user_term, db_name) in enumerate(list(self.feature_name_mappings.items())[:10]):
            print(f"   '{user_term}' → '{db_name}'")
        
        print(f"\n🔄 Features with Value Mappings: {len(self.value_mappings)}")
        print("   Examples:")
        for i, (feature, value_map) in enumerate(list(self.value_mappings.items())[:5]):
            print(f"   {feature}: {len(value_map)} value mappings")
            for j, (user_val, db_val) in enumerate(list(value_map.items())[:3]):
                print(f"      '{user_val}' → '{db_val}'")
        
        print("\n" + "=" * 80)


def main():
    """Main function"""
    import sys
    
    map_file = '/usr/src/app/Test_Input/charactDescr_valueCharLong_map.json'
    output_file = '/usr/src/app/data/feature_extraction_mappings.json'
    
    if len(sys.argv) > 1:
        map_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    logger.info("=" * 80)
    logger.info("FEATURE MAPPING GENERATOR")
    logger.info("=" * 80)
    logger.info(f"Input: {map_file}")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 80)
    
    generator = FeatureMappingGenerator(map_file)
    generator.save_mappings(output_file)
    generator.print_summary()
    
    logger.info("\n✅ Feature mappings generated successfully!")


if __name__ == "__main__":
    main()

