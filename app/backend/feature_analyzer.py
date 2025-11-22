#!/usr/bin/env python3
"""
Feature Analyzer - Pre-analyzes features using charactDescr_valueCharLong_map.json

This script analyzes the comprehensive feature map to determine:
- Binary features (Yes/No, Ja/Nein, allowed/not allowed)
- Numerical features (Brix, pH, percentages, ranges)
- Categorical features (colors, flavors, etc.)
- Multi-value patterns

The results are used to optimize feature encoding during indexing.
"""
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Analyzes features from the charactDescr_valueCharLong map"""
    
    def __init__(self, map_file_path: str):
        """
        Initialize analyzer with feature map
        
        Args:
            map_file_path: Path to charactDescr_valueCharLong_map.json
        """
        self.map_file_path = map_file_path
        self.feature_map = self._load_map()
        
        # Feature categorization
        self.binary_features: Dict[str, Dict[str, Any]] = {}
        self.numerical_features: Dict[str, Dict[str, Any]] = {}
        self.categorical_features: Dict[str, Dict[str, Any]] = {}
        self.range_features: Dict[str, Dict[str, Any]] = {}
        
        # Boolean indicators (multilingual)
        self.positive_indicators = {
            'yes', 'ja', 'oui', 'si', 'sim', 'da', 
            'allowed', 'permitted', 'erlaubt', 'autorisé',
            'true', '1', 'active', 'present'
        }
        self.negative_indicators = {
            'no', 'nein', 'non', 'não', 'não', 'nee',
            'not allowed', 'forbidden', 'verboten', 'interdit',
            'false', '0', 'inactive', 'absent'
        }
        
    def _load_map(self) -> Dict[str, List[str]]:
        """Load the feature map from JSON file"""
        try:
            with open(self.map_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded feature map with {len(data)} unique features")
            return data
        except Exception as e:
            logger.error(f"Error loading feature map: {e}")
            return {}
    
    def analyze_all_features(self) -> Dict[str, Any]:
        """
        Analyze all features and categorize them
        
        Returns:
            Dictionary with feature analysis results
        """
        logger.info("Starting comprehensive feature analysis...")
        
        for feature_name, values in self.feature_map.items():
            if not values:
                continue
                
            # Clean values
            clean_values = [str(v).strip() for v in values if v and str(v).strip()]
            unique_values = list(set(clean_values))
            
            # Categorize feature
            if self._is_binary_feature(feature_name, unique_values):
                self.binary_features[feature_name] = {
                    'values': unique_values,
                    'count': len(unique_values),
                    'type': 'binary'
                }
            elif self._is_numerical_feature(feature_name, unique_values):
                self.numerical_features[feature_name] = {
                    'values': unique_values[:10],  # Sample
                    'count': len(unique_values),
                    'type': 'numerical',
                    'subtype': self._detect_numerical_subtype(feature_name, unique_values)
                }
            elif self._is_range_feature(feature_name, unique_values):
                self.range_features[feature_name] = {
                    'values': unique_values[:10],  # Sample
                    'count': len(unique_values),
                    'type': 'range'
                }
            else:
                self.categorical_features[feature_name] = {
                    'values': unique_values[:20],  # Sample
                    'count': len(unique_values),
                    'type': 'categorical'
                }
        
        logger.info(f"✅ Binary features: {len(self.binary_features)}")
        logger.info(f"✅ Numerical features: {len(self.numerical_features)}")
        logger.info(f"✅ Range features: {len(self.range_features)}")
        logger.info(f"✅ Categorical features: {len(self.categorical_features)}")
        
        return self._create_analysis_report()
    
    def _is_binary_feature(self, feature_name: str, values: List[str]) -> bool:
        """Determine if feature is binary (Yes/No, allowed/not allowed, etc.)"""
        if len(values) > 10:  # Too many values for binary
            return False
        
        values_lower = [v.lower().strip() for v in values]
        
        # Check for explicit binary patterns
        binary_patterns = [
            {'yes', 'no'},
            {'ja', 'nein'},
            {'oui', 'non'},
            {'si', 'no'},
            {'allowed', 'not allowed'},
            {'true', 'false'},
            {'active', 'inactive'},
            {'present', 'absent'},
        ]
        
        values_set = set(values_lower)
        for pattern in binary_patterns:
            if pattern.issubset(values_set) or values_set.issubset(pattern):
                return True
        
        # Check if all values are positive or negative indicators
        positive_count = sum(1 for v in values_lower if v in self.positive_indicators)
        negative_count = sum(1 for v in values_lower if v in self.negative_indicators)
        
        if positive_count > 0 and negative_count > 0:
            return True
        
        # Check feature name for binary keywords
        binary_keywords = [
            'halal', 'kosher', 'vegan', 'organic', 'gmo', 'artificial',
            'natural', 'allergen', 'gluten', 'lactose', 'certification',
            'approved', 'compliant', 'free', 'contains'
        ]
        name_lower = feature_name.lower()
        if any(keyword in name_lower for keyword in binary_keywords):
            if len(values) <= 5:  # Limited values suggest binary
                return True
        
        return False
    
    def _is_numerical_feature(self, feature_name: str, values: List[str]) -> bool:
        """Determine if feature is numerical (pH, Brix, percentages, etc.)"""
        # Check feature name for numerical keywords
        numerical_keywords = [
            'brix', 'ph', 'acid', 'percent', '%', 'temperature',
            'content', 'concentration', 'weight', 'volume', 'density',
            'viscosity', 'ratio', 'index', 'value', 'level', 'count'
        ]
        
        name_lower = feature_name.lower()
        if any(keyword in name_lower for keyword in numerical_keywords):
            # Verify that values are actually numerical
            numerical_count = 0
            for v in values[:50]:  # Sample first 50
                if self._looks_like_number(v):
                    numerical_count += 1
            
            if numerical_count / min(len(values), 50) > 0.5:  # >50% are numbers
                return True
        
        # Check if majority of values are numerical
        numerical_count = 0
        for v in values[:100]:  # Sample
            if self._looks_like_number(v):
                numerical_count += 1
        
        return numerical_count / min(len(values), 100) > 0.7  # >70% are numbers
    
    def _is_range_feature(self, feature_name: str, values: List[str]) -> bool:
        """Determine if feature represents ranges (e.g., '3.0-4.5', '25-35')"""
        range_pattern = r'^\d+\.?\d*\s*[-–]\s*\d+\.?\d*$'
        range_count = 0
        
        for v in values[:50]:  # Sample
            if re.match(range_pattern, v.strip()):
                range_count += 1
        
        return range_count / min(len(values), 50) > 0.5  # >50% are ranges
    
    def _looks_like_number(self, value: str) -> bool:
        """Check if value looks like a number"""
        value_clean = value.strip().replace(',', '.').replace('%', '').replace('±', '')
        
        # Simple number
        try:
            float(value_clean)
            return True
        except ValueError:
            pass
        
        # Percentage
        if '%' in value:
            return True
        
        # Range (e.g., "3.0-4.5")
        if re.match(r'^\d+\.?\d*\s*[-–]\s*\d+\.?\d*', value):
            return True
        
        # Plus-minus notation (e.g., "30±5")
        if '±' in value:
            return True
        
        # Scientific notation
        if re.match(r'^\d+\.?\d*[eE][+-]?\d+$', value_clean):
            return True
        
        return False
    
    def _detect_numerical_subtype(self, feature_name: str, values: List[str]) -> str:
        """Detect the subtype of numerical feature"""
        name_lower = feature_name.lower()
        
        if 'brix' in name_lower:
            return 'brix'
        elif 'ph' in name_lower:
            return 'ph'
        elif 'percent' in name_lower or '%' in name_lower:
            return 'percentage'
        elif 'temperature' in name_lower or 'temp' in name_lower:
            return 'temperature'
        elif 'content' in name_lower or 'concentration' in name_lower:
            return 'concentration'
        else:
            # Check values for percentage signs
            if any('%' in str(v) for v in values[:20]):
                return 'percentage'
            return 'general'
    
    def _create_analysis_report(self) -> Dict[str, Any]:
        """Create comprehensive analysis report"""
        return {
            'total_features': len(self.feature_map),
            'binary_features': {
                'count': len(self.binary_features),
                'features': self.binary_features
            },
            'numerical_features': {
                'count': len(self.numerical_features),
                'features': self.numerical_features
            },
            'range_features': {
                'count': len(self.range_features),
                'features': self.range_features
            },
            'categorical_features': {
                'count': len(self.categorical_features),
                'features': self.categorical_features
            }
        }
    
    def save_analysis(self, output_path: str):
        """Save analysis results to JSON file"""
        analysis = self.analyze_all_features()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Analysis saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
    
    def get_feature_config_for_indexing(self) -> Dict[str, str]:
        """
        Get simplified feature type configuration for indexing
        
        Returns:
            Dictionary mapping feature names to types ('binary', 'numerical', 'range', 'categorical')
        """
        config = {}
        
        for feature_name in self.binary_features:
            config[feature_name] = 'binary'
        
        for feature_name in self.numerical_features:
            config[feature_name] = 'numerical'
        
        for feature_name in self.range_features:
            config[feature_name] = 'range'
        
        for feature_name in self.categorical_features:
            config[feature_name] = 'categorical'
        
        return config
    
    def print_summary(self):
        """Print summary of feature analysis"""
        print("\n" + "=" * 80)
        print("FEATURE ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Total Features: {len(self.feature_map)}")
        print(f"\n✅ Binary Features: {len(self.binary_features)}")
        if self.binary_features:
            print("   Examples:")
            for i, (name, info) in enumerate(list(self.binary_features.items())[:5]):
                print(f"   - {name}: {info['values'][:5]}")
        
        print(f"\n🔢 Numerical Features: {len(self.numerical_features)}")
        if self.numerical_features:
            print("   Examples:")
            for i, (name, info) in enumerate(list(self.numerical_features.items())[:5]):
                print(f"   - {name} ({info['subtype']}): {info['values'][:3]}...")
        
        print(f"\n📏 Range Features: {len(self.range_features)}")
        if self.range_features:
            print("   Examples:")
            for i, (name, info) in enumerate(list(self.range_features.items())[:5]):
                print(f"   - {name}: {info['values'][:3]}...")
        
        print(f"\n📂 Categorical Features: {len(self.categorical_features)}")
        if self.categorical_features:
            print("   Examples:")
            for i, (name, info) in enumerate(list(self.categorical_features.items())[:5]):
                count = info['count']
                print(f"   - {name} ({count} unique values): {info['values'][:3]}...")
        
        print("\n" + "=" * 80)


def main():
    """Main function to run feature analysis"""
    import sys
    
    # Path to feature map
    map_file = '/usr/src/app/Test_Input/charactDescr_valueCharLong_map.json'
    output_file = '/usr/src/app/data/feature_analysis.json'
    config_file = '/usr/src/app/data/feature_config.json'
    
    # Allow overriding via command line
    if len(sys.argv) > 1:
        map_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    logger.info("=" * 80)
    logger.info("FEATURE ANALYZER - Pre-analysis for Enhanced Indexing")
    logger.info("=" * 80)
    logger.info(f"Input: {map_file}")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 80)
    
    # Create analyzer
    analyzer = FeatureAnalyzer(map_file)
    
    # Run analysis
    analyzer.analyze_all_features()
    
    # Save results
    analyzer.save_analysis(output_file)
    
    # Save simplified config for indexing
    config = analyzer.get_feature_config_for_indexing()
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Feature config saved to {config_file}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")
    
    # Print summary
    analyzer.print_summary()
    
    logger.info("\n✅ Feature analysis completed successfully!")


if __name__ == "__main__":
    main()

