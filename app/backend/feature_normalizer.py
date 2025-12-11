#!/usr/bin/env python3
"""
Feature Normalizer for Multilingual Recipe Data

Normalizes feature names and values from German, French, Polish, etc. to 
standardized English format for consistent vector search matching.

This is applied during indexing to ensure that:
1. German "Stärke: Stärke enthalten" becomes "Starch: Yes"
2. German "Künstliche Farben: keine künstl. Farbe" becomes "Artificial colors: No"
3. German "Kosher relevant: Nein" becomes "Kosher: No"

This enables English queries to match German-indexed recipes via vector similarity.
"""
import json
import os
import logging
import re
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('feature_normalizer')


class FeatureNormalizer:
    """
    Normalizes multilingual feature names and values to standardized English format.
    
    Uses mappings from feature_extraction_mappings.json plus additional
    German/English value translations for consistent search matching.
    """
    
    def __init__(self, mappings_path: Optional[str] = None):
        """
        Initialize the normalizer with mappings.
        
        Args:
            mappings_path: Path to feature_extraction_mappings.json
        """
        self.feature_name_mappings: Dict[str, str] = {}
        self.value_mappings: Dict[str, Dict[str, str]] = {}
        
        # Load mappings from file if provided
        if mappings_path and os.path.exists(mappings_path):
            self._load_mappings(mappings_path)
        
        # Add additional German → English feature name mappings
        self._add_german_feature_mappings()
        
        # Add universal value normalization (German/French/etc → English)
        self._add_universal_value_mappings()
    
    def _load_mappings(self, mappings_path: str) -> None:
        """Load mappings from JSON file."""
        try:
            with open(mappings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'feature_name_mappings' in data:
                self.feature_name_mappings = data['feature_name_mappings']
            
            if 'value_mappings' in data:
                self.value_mappings = data['value_mappings']
            
            logger.info(f"Loaded {len(self.feature_name_mappings)} feature name mappings")
            logger.info(f"Loaded {len(self.value_mappings)} value mapping categories")
            
        except Exception as e:
            logger.error(f"Error loading mappings from {mappings_path}: {e}")
    
    def _add_german_feature_mappings(self) -> None:
        """Add German → English feature name mappings for key features."""
        german_mappings = {
            # Stabilizers
            "stärke": "Starch",
            "pektin": "Pectin",
            "xanthan": "Xanthan",
            "guarkernmehl": "Guar",
            "johannisbrotkernmehl": "LBG",
            "ibkm": "LBG",  # IBKM = Johannisbrotkernmehl
            "andere stabilisatoren": "Other stabilizer",
            
            # Colors and flavors
            "künstliche farben": "Artificial colors",
            "farbe": "Color",
            "aroma": "Flavour",
            "natürliche aromen": "Natural flavor",
            "naturident/künstliches aroma": "Nature identical flavor",
            
            # Certifications
            "kosher relevant": "Kosher",
            "koscher typ": "Kosher",
            "halal": "Halal",
            
            # Preservation
            "konservierung": "Preserved",
            
            # Sugar/Sweetener
            "süßstoff": "Sweetener",
            "saccharose": "Saccharose",
            
            # GMO
            "nicht genfrei": "Contains GMO",
            "gmo enthalten": "Contains GMO",
            
            # Allergens
            "allergene": "Allergens",
            "allergenfrei": "Allergens",
            
            # Other
            "extremrezeptur": "Extreme recipe",
            "mischung": "Blend",
            "aspartam": "Aspartame",
            "laktosefrei": "Lactose free",
            "standardprodukt": "Standard product",
            "industrie (sd reporting)": "Industry (SD Reporting)",
            "produktsegment (sd reporting)": "Produktsegment (SD Reporting)",
            "materialkurztext": "Material short text",
            "nachbau": "Rebuild",
            "flüssig/stückig": "Puree/with pieces",
            
            # Vegan/Vegetarian
            "typ vegan / vegetarisch": "Type vegan / vegetarian",
        }
        
        # Add to mappings (lowercase keys for matching)
        for german, english in german_mappings.items():
            self.feature_name_mappings[german.lower()] = english
    
    def _add_universal_value_mappings(self) -> None:
        """Add universal value mappings for German/French/Polish/etc → English."""
        # Universal positive values → "Yes"
        # Covers: German, English, French, Italian, Spanish, Portuguese, Dutch, Romanian, Polish, Hungarian, Czech, Turkish
        positive_values = [
            # Basic yes in multiple languages
            "ja", "yes", "oui", "si", "sim", "da", "tak", "igen", "ano", "evet",
            # English variations
            "allowed", "permitted", "active", "present", "true", "1",
            # German variations
            "enthalten", "mit", "vorhanden", "erlaubt",
            # French variations
            "autorisé", "permis", "actif", "présent",
            # Specific positive values
            "halal", "kosher", "koscher",
            "stärke enthalten", "pektin enthalten", "pectin",
            "mit aroma", "mit allergenen",
            "natürliches aroma", "natural flavour", "natural flavor",
            "allergens", "allergen", "allergenen",
            "starch", "lbg", "guar", "xanthan",
            "saccharose", "saccarose",
            "sweetener", "süssstoff",
            "standard product", "standardprodukt",
            "pasteurized", "pasteurisiert",
        ]
        
        # Universal negative values → "No"
        # Covers: German, English, French, Italian, Spanish, Portuguese, Dutch, Romanian, Polish, Hungarian, Czech, Turkish
        negative_values = [
            # Basic no in multiple languages
            "nein", "no", "non", "não", "nee", "nie", "nem", "hayır",
            # English variations
            "forbidden", "not allowed", "inactive", "absent", "false", "0",
            # German variations
            "kein", "keine", "nicht", "ohne", "verboten",
            # French variations
            "interdit", "pas", "sans", "aucun",
            # Specific negative values
            "keine künstl. farbe", "no artificial colors",
            "keine süsstoffe", "no sweetener",
            "kein naturidentes aroma", "no nature identical flavour",
            "nicht konserviert", "no preservative",
            "allergenfrei", "no allergens",
            "keine anderen stabil enthalten", "no other stabilizer",
            "keine farbe enthalten", "no color",
            "kein aspartam", "no aspartame",
            "kein xanthan", "no xanthan",
            "kein pektin", "no pectin",
            "kein ibkm", "no lbg",
            "keine mischung", "no blend",
            "keine extremrezeptur", "no extreme recipe",
            "nicht laktosefrei", "not lactose free",
            "kein nachbau", "no rebuild",
            "no saccarose", "keine saccharose",
            "no starch", "kein stärke",
            "not gene manipulated", "genfrei",
            "non-plant-based", "non-halal",
        ]
        
        # Create lookup sets for fast matching
        self._positive_values = set(v.lower() for v in positive_values)
        self._negative_values = set(v.lower() for v in negative_values)
        
        # Specific value normalizations (exact matches)
        self._exact_value_mappings = {
            # Starch
            "stärke enthalten": "Yes",
            "starch": "Yes",
            "no starch": "No",
            
            # Pectin
            "pektin enthalten": "Yes",
            "pectin": "Yes",
            "kein pektin": "No",
            "no pectin": "No",
            
            # LBG/IBKM
            "lbg": "Yes",
            "kein ibkm": "No",
            "no lbg": "No",
            
            # Xanthan
            "kein xanthan": "No",
            "no xanthan": "No",
            
            # Guar
            "guarkernmehl": "Yes",
            "guar": "Yes",
            
            # Colors
            "keine künstl. farbe": "No",
            "no artificial colors": "No",
            "keine farbe enthalten": "No",
            "no color": "No",
            "farbe enthalten": "Yes",
            "color": "Yes",
            
            # Flavor/Aroma
            "mit aroma": "Yes",
            "flavour": "Yes",
            "natürliches aroma": "Yes",
            "natural flavour": "Yes",
            "natural flavor": "Yes",
            "kein naturidentes aroma": "No",
            "no nature identical flavour": "No",
            "naturidentes aroma": "Yes",
            
            # Preservatives
            "nicht konserviert": "No",
            "no preservative": "No",
            
            # Sweetener
            "keine süsstoffe": "No",
            "no sweetener": "No",
            "sweetener": "Yes",
            
            # Allergens
            "allergenfrei": "No",
            "no allergens": "No",
            "mit allergenen": "Yes",
            "allergens": "Yes",
            
            # GMO
            "nicht genfrei": "Yes",  # Contains GMO
            "genfrei": "No",  # GMO-free
            "not gene manipulated": "No",
            
            # Halal/Kosher
            "halal": "Yes",
            "non-halal": "No",
            "kosher": "Yes",
            
            # Other stabilizers
            "keine anderen stabil enthalten": "No",
            "no other stabilizer": "No",
            "andere stabilisatoren": "Yes",
            "other stabilizer": "Yes",
            
            # Aspartame
            "kein aspartam": "No",
            "no aspartame": "No",
            
            # Blend
            "keine mischung": "No",
            "no blend": "No",
            
            # Standard product
            "standardprodukt": "Yes",
            "standard product": "Yes",
            
            # Puree/Pieces
            "stückig": "with pieces",
            "with pieces": "with pieces",
            "puree": "puree",
        }
    
    def normalize_feature_name(self, feature_name: str) -> str:
        """
        Normalize a feature name to standardized English format.
        
        Args:
            feature_name: Original feature name (may be German, etc.)
            
        Returns:
            Normalized English feature name
        """
        if not feature_name:
            return feature_name
        
        # Try exact match first (lowercase)
        lower_name = feature_name.lower().strip()
        
        if lower_name in self.feature_name_mappings:
            return self.feature_name_mappings[lower_name]
        
        # Return original if no mapping found
        return feature_name
    
    def normalize_value(self, feature_name: str, value: str) -> str:
        """
        Normalize a feature value to standardized English format.
        
        Args:
            feature_name: The feature name (for context-specific mappings)
            value: Original value (may be German, etc.)
            
        Returns:
            Normalized English value
        """
        if not value:
            return value
        
        lower_value = value.lower().strip()
        
        # Check exact value mappings first
        if lower_value in self._exact_value_mappings:
            return self._exact_value_mappings[lower_value]
        
        # Check feature-specific value mappings
        normalized_feature = self.normalize_feature_name(feature_name)
        if normalized_feature in self.value_mappings:
            if lower_value in self.value_mappings[normalized_feature]:
                mapped = self.value_mappings[normalized_feature][lower_value]
                # Further normalize German results to English
                if mapped.lower() in ("ja", "yes", "true", "1"):
                    return "Yes"
                elif mapped.lower() in ("nein", "no", "false", "0"):
                    return "No"
                return mapped
        
        # Check if value contains negative indicators
        for neg in ["kein", "keine", "nicht", "ohne", "no ", "not "]:
            if neg in lower_value:
                return "No"
        
        # Check if value contains positive indicators
        for pos in ["enthalten", "mit ", "yes", "ja"]:
            if pos in lower_value:
                return "Yes"
        
        # Return original if no mapping found
        return value
    
    def normalize_features(
        self, 
        features: List[str], 
        values: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Normalize a list of features and their values.
        
        Args:
            features: List of feature names
            values: List of corresponding values
            
        Returns:
            Tuple of (normalized_features, normalized_values)
        """
        normalized_features = []
        normalized_values = []
        
        for feature, value in zip(features, values):
            norm_feature = self.normalize_feature_name(feature)
            norm_value = self.normalize_value(feature, str(value) if value else "")
            normalized_features.append(norm_feature)
            normalized_values.append(norm_value)
        
        return normalized_features, normalized_values
    
    def enhance_description(
        self,
        original_description: str,
        features: List[str],
        values: List[str]
    ) -> str:
        """
        Enhance the recipe description with key searchable terms.
        
        Adds important feature information to make text search more effective
        without full translation.
        
        Args:
            original_description: Original recipe description
            features: List of feature names
            values: List of corresponding values
            
        Returns:
            Enhanced description with key searchable terms
        """
        description_parts = [original_description]
        
        # Key features to extract for description enhancement
        key_feature_mappings = {
            # Flavour keywords
            "flavour": "Flavour",
            "flavor": "Flavour",
            "geschmack": "Flavour",
            "materialkurztext": None,  # Already in description usually
            "material short text": None,
            
            # Application
            "application (fruit filling)": "Application",
            "fruit prep application": "Application",
            "produktsegment (sd reporting)": "Segment",
            
            # Project
            "projekttitel": "Project",
            "project title": "Project",
        }
        
        # Extract matcha/flavor terms from MaterialMasterShorttext (in values)
        for feature, value in zip(features, values):
            if not value:
                continue
            
            value_str = str(value)
            lower_feature = feature.lower()
            
            # Look for flavor-indicating terms in short text
            if lower_feature in ("materialkurztext", "material short text"):
                # Extract key words from material text
                matcha_match = re.search(r'\b(matcha|green\s*tea)\b', value_str, re.IGNORECASE)
                if matcha_match:
                    description_parts.append(f"Flavour: Matcha")
            
            # Look for flavor feature
            if lower_feature in ("flavour", "flavor", "geschmack"):
                if value_str.lower() not in ("flavour", "flavor", "no flavour"):
                    description_parts.append(f"Flavour: {value_str}")
            
            # Look for projekttitel containing brand/flavor info
            if lower_feature == "projekttitel":
                # Check if it contains flavor hints
                if re.search(r'\b(matcha|siggis|siggi)\b', value_str, re.IGNORECASE):
                    description_parts.append(f"Project: {value_str}")
        
        # Combine and deduplicate
        enhanced = ", ".join(description_parts)
        
        # Limit length
        if len(enhanced) > 1000:
            enhanced = enhanced[:997] + "..."
        
        return enhanced


def create_normalizer(mappings_path: Optional[str] = None) -> FeatureNormalizer:
    """
    Factory function to create a feature normalizer.
    
    Args:
        mappings_path: Path to feature_extraction_mappings.json
        
    Returns:
        Configured FeatureNormalizer instance
    """
    return FeatureNormalizer(mappings_path)

