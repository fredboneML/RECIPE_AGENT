#!/usr/bin/env python3
"""
Qdrant Recipe Manager

This manager actually uses Qdrant for persistent recipe storage and search,
replacing the in-memory approach of EnhancedTwoStepRecipeManager.

IMPORTANT: Uses EnhancedTwoStepRecipeManager for feature encoding to ensure
binary opposition mapping and all feature encoding is IDENTICAL to indexing.
"""
import logging
import re
import os
import json
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny, MatchText, Range

# Import EnhancedTwoStepRecipeManager for consistent encoding with indexing
from src.two_step_recipe_search import EnhancedTwoStepRecipeManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature encoding constants (kept for backward compatibility)
NEGATIVE_INDICATORS = [
    'no ', 'not ', 'without ', 'non-', 'non ', 'free', 'absent',
    'none', 'zero', 'nil', 'false', 'inactive', 'negative',
    'excluded', 'forbidden', 'prohibited', 'banned', 'restricted',
    'nein', 'kein', 'ohne', 'frei von', 'nicht'  # German
]

POSITIVE_INDICATORS = [
    'yes', 'with ', 'contains', 'includes', 'present', 'active',
    'true', 'positive', 'allowed', 'permitted', 'approved',
    'ja', 'mit ', 'enthält', 'aktiv'  # German
]

BINARY_FEATURE_NAMES = [
    'allergen', 'preserve', 'artificial', 'natural', 'gmo', 'organic',
    'kosher', 'halal', 'color', 'flavor', 'sweetener', 'starch',
    'pectin', 'blend', 'aspartame', 'additive', 'chemical'
]


class QdrantRecipeManager:
    """
    Recipe manager that uses Qdrant for storage and search.
    Compatible with the existing agent interface but uses persistent storage.
    """

    def __init__(self,
                 collection_name: str = "food_recipes_two_step",
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 qdrant_host: str = "qdrant",
                 qdrant_port: int = 6333,
                 feature_map_path: str = "/usr/src/app/Test_Input/charactDescr_valueCharLong_map.json"):
        """
        Initialize the Qdrant Recipe Manager

        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Sentence transformer model name
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            feature_map_path: Path to the feature map JSON for pre-analysis
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

        # Initialize Qdrant client with increased timeout for large databases
        try:
            self.qdrant_client = QdrantClient(
                host=qdrant_host, 
                port=qdrant_port,
                timeout=120.0  # 2 minutes timeout for 600K+ recipe database
            )
            logger.info(
                f"Connected to Qdrant at {qdrant_host}:{qdrant_port} with 120s timeout")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

        # Initialize embedding model - use SentenceTransformer (same as indexing)
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Initialize EnhancedTwoStepRecipeManager for CONSISTENT feature encoding
        # This ensures binary opposition mapping and all feature encoding matches indexing
        self._init_feature_encoder(embedding_model, feature_map_path)

        # Check if collection exists
        self._validate_collection()

    def _init_feature_encoder(self, embedding_model: str, feature_map_path: str):
        """
        Initialize EnhancedTwoStepRecipeManager for feature encoding.

        This ensures that feature vectors created during search match exactly
        how they were created during indexing, including:
        - Binary opposition mapping (e.g., 'no sugar' vs 'sugar')
        - Numerical feature normalization
        - Categorical feature encoding

        OPTIMIZATION: Shares the embedding model instance to avoid loading twice.
        """
        try:
            logger.info(
                "Initializing feature encoder (EnhancedTwoStepRecipeManager)...")

            # Create the manager for encoding
            self.feature_encoder = EnhancedTwoStepRecipeManager(
                collection_name=self.collection_name,
                embedding_model=embedding_model,
                max_features=200
            )

            # OPTIMIZATION: Share the embedding model instance to avoid double memory usage
            # and speed up initialization
            self.feature_encoder.embedding_model = self.embedding_model
            logger.info("Shared embedding model instance with feature encoder")

            # Load and inject pre-analyzed feature types (same as indexing)
            self._load_and_inject_feature_types(feature_map_path)

            # Run feature analysis on the feature map to build binary oppositions
            self._analyze_feature_map(feature_map_path)

            logger.info("Feature encoder initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize feature encoder: {e}")
            logger.warning(
                "Falling back to basic encoding (may affect search quality)")
            self.feature_encoder = None

    def _load_and_inject_feature_types(self, feature_map_path: str):
        """Load pre-analyzed feature types and inject into the encoder"""
        if not os.path.exists(feature_map_path):
            logger.warning(f"Feature map not found at {feature_map_path}")
            return

        try:
            # Use FeatureAnalyzer if available, otherwise do simple analysis
            try:
                from feature_analyzer import FeatureAnalyzer
                analyzer = FeatureAnalyzer(feature_map_path)
                analyzer.analyze_all_features()
                feature_config = analyzer.get_feature_config_for_indexing()

                # Inject feature types into encoder
                for feature_name, feature_type in feature_config.items():
                    if feature_type == 'binary':
                        self.feature_encoder.feature_types[feature_name] = 'binary'
                    elif feature_type in ['numerical', 'range']:
                        self.feature_encoder.feature_types[feature_name] = 'numerical'

                logger.info(
                    f"Injected {len(feature_config)} pre-analyzed feature types")
                logger.info(
                    f"  Binary: {sum(1 for t in feature_config.values() if t == 'binary')}")
                logger.info(
                    f"  Numerical: {sum(1 for t in feature_config.values() if t in ['numerical', 'range'])}")

            except ImportError:
                logger.warning(
                    "FeatureAnalyzer not available, using basic feature type detection")

        except Exception as e:
            logger.warning(f"Error loading feature types: {e}")

    def _analyze_feature_map(self, feature_map_path: str):
        """
        Analyze the feature map to build binary opposition mappings.
        This matches what indexing does with analyze_features_for_binary_patterns().
        """
        if not os.path.exists(feature_map_path):
            return

        try:
            logger.info(
                "Analyzing feature map for binary opposition patterns...")

            with open(feature_map_path, 'r', encoding='utf-8') as f:
                feature_map = json.load(f)

            # Convert feature map to features/values format for analysis
            features_list = []
            values_list = []

            for feature_name, values in feature_map.items():
                if values:
                    for value in values:
                        features_list.append([feature_name])
                        values_list.append([value])

            # Run the encoder's feature analysis to build binary oppositions
            self.feature_encoder._analyze_feature_values(
                features_list, values_list)

            logger.info(f"Binary opposition analysis complete:")
            logger.info(
                f"  Feature types detected: {len(self.feature_encoder.feature_types)}")
            logger.info(
                f"  Binary features with oppositions: {len(self.feature_encoder.binary_features)}")

        except Exception as e:
            logger.warning(f"Error analyzing feature map: {e}")

    def _validate_collection(self):
        """Validate that the collection exists in Qdrant"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.warning(
                    f"Collection '{self.collection_name}' not found in Qdrant. "
                    f"Available collections: {collection_names}")
                logger.warning(
                    "Please run init_vector_index.py to create and populate the collection.")
            else:
                collection_info = self.qdrant_client.get_collection(
                    self.collection_name)
                logger.info(
                    f"Collection '{self.collection_name}' found with "
                    f"{collection_info.points_count} recipes")
        except Exception as e:
            logger.error(f"Error validating collection: {e}")

    def _create_feature_text(self, features: List[str], values: List[Any]) -> str:
        """
        Create text representation from features and values for embedding.

        Args:
            features: List of feature names
            values: List of feature values

        Returns:
            Text representation of features
        """
        parts = []
        for feat, val in zip(features, values):
            val_str = str(val).strip() if val is not None else ""
            if val_str and val_str.lower() not in ['nan', 'none', '']:
                parts.append(f"{feat}: {val_str}")
        return ", ".join(parts)

    def _detect_value_type(self, value: Any) -> Tuple[str, float]:
        """
        Detect value type and extract numerical value if applicable.
        Matches the logic from EnhancedTwoStepRecipeManager.
        """
        if pd.isna(value) or value is None or str(value).strip() == '' or str(value).strip().lower() == 'none':
            return 'missing', 0.0

        value_str = str(value).strip()

        # Handle European percentage format (e.g., "45,5%")
        percentage_pattern = r'^(\d+(?:,\d+)?)\s*%?$'
        if value_str.endswith('%') or re.match(percentage_pattern, value_str):
            try:
                num_str = value_str.replace('%', '').replace(',', '.').strip()
                num_val = float(num_str)
                return 'percentage', num_val
            except ValueError:
                return 'categorical', 0.0

        # Handle European decimal format (e.g., "3,5")
        european_number_pattern = r'^(\d+(?:,\d+)?)\s*[a-zA-Z]*$'
        if re.match(european_number_pattern, value_str):
            try:
                number_match = re.match(r'^(\d+(?:,\d+)?)', value_str)
                if number_match:
                    num_str = number_match.group(1).replace(',', '.')
                    num_val = float(num_str)
                    return 'numerical', num_val
            except ValueError:
                pass

        # Try standard number parsing
        try:
            num_val = float(value_str)
            return 'numerical', num_val
        except ValueError:
            pass

        return 'categorical', 0.0

    def _is_likely_binary_feature(self, feature_name: str, value: str) -> bool:
        """Check if a feature is likely binary based on name and value."""
        value_lower = value.lower() if value else ""

        # Check if feature name suggests binary
        name_suggests_binary = any(name in feature_name.lower()
                                   for name in BINARY_FEATURE_NAMES)

        # Check if value contains binary indicators
        has_negative = any(neg in value_lower for neg in NEGATIVE_INDICATORS)
        has_positive = any(pos in value_lower for pos in POSITIVE_INDICATORS)

        return name_suggests_binary or has_negative or has_positive

    def _encode_binary_feature(self, feature_name: str, value: str) -> float:
        """
        Encode binary feature value to -1.0, 0.0, or 1.0.
        Matches the logic from EnhancedTwoStepRecipeManager.
        """
        if not value or pd.isna(value):
            return 0.0

        value_clean = str(value).lower().strip()

        # Check for negative indicators
        for neg_indicator in NEGATIVE_INDICATORS:
            if neg_indicator in value_clean:
                return -1.0

        # Check for positive indicators
        for pos_indicator in POSITIVE_INDICATORS:
            if pos_indicator in value_clean:
                return 1.0

        # If feature name is in value, it's likely positive
        if feature_name.lower() in value_clean or any(
            word in value_clean for word in feature_name.lower().split()
        ):
            return 1.0

        return 0.0

    def _encode_categorical_feature(self, feature_name: str, value: str) -> float:
        """
        Encode categorical feature using hash-based approach.
        This approximates the LabelEncoder behavior from indexing.
        """
        if not value or pd.isna(value):
            return 0.0

        # Use hash to get consistent encoding for same value
        # Normalize to [0, 1] range and scale similarly to indexing (/ 20.0)
        value_clean = str(value).lower().strip()
        hash_val = hash(f"{feature_name}:{value_clean}") % 1000
        return (hash_val / 1000.0) / 20.0  # Match indexing scale

    def _encode_mixed_features(self, features: List[str], values: List[Any]) -> np.ndarray:
        """
        Encode features using the same logic as EnhancedTwoStepRecipeManager.
        Creates a 100-dimensional categorical vector.

        This handles:
        - Binary features (encoded as -1.0, 0.0, 1.0)
        - Numerical/percentage features (normalized to [-1, 1])
        - Categorical features (hash-based encoding)
        """
        categorical_dim = 100
        categorical_vector = np.zeros(categorical_dim)

        # Sort features for consistent ordering (matching indexing behavior)
        feature_dict = {feat: val for feat, val in zip(features, values)}
        sorted_features = sorted(feature_dict.keys())

        for i, feature_name in enumerate(sorted_features[:categorical_dim]):
            value = feature_dict[feature_name]
            value_str = str(value) if value is not None else ""

            # Detect value type
            value_type, num_val = self._detect_value_type(value)

            if value_type == 'missing':
                categorical_vector[i] = 0.0

            elif value_type in ['numerical', 'percentage']:
                # Normalize to [-1, 1] range (matching indexing: / 100.0)
                categorical_vector[i] = min(max(num_val / 100.0, -1.0), 1.0)

            elif self._is_likely_binary_feature(feature_name, value_str):
                # Binary encoding
                categorical_vector[i] = self._encode_binary_feature(
                    feature_name, value_str)

            else:
                # Categorical encoding (hash-based approximation)
                categorical_vector[i] = self._encode_categorical_feature(
                    feature_name, value_str)

        return categorical_vector

    def search_by_features(self,
                           query_features: List[str],
                           query_values: List[Any],
                           top_k: int = 20,
                           country_filter: Optional[Union[str, List[str]]] = None,
                           version_filter: Optional[str] = None,
                           numerical_filters: Optional[Dict[str, Dict[str, Any]]] = None,
                           categorical_filters: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Search recipes by features using the feature vector in Qdrant.

        This method creates a feature vector that matches the indexing structure:
        - First 384 dim: Text embedding of feature text
        - Last 100 dim: Encoded features (binary, numerical, categorical)

        IMPORTANT: Uses EnhancedTwoStepRecipeManager for encoding to ensure
        binary opposition mapping matches exactly how recipes were indexed.

        Args:
            query_features: List of feature names
            query_values: List of feature values
            top_k: Number of results to return
            country_filter: Optional country name(s) to filter results. Can be a single string or a list of strings. None or "All" means no filter.
            version_filter: Optional version filter (P, L, Missing, or "All" means no filter)
            numerical_filters: Optional dict mapping field codes to Qdrant range filters
                Example: {"Z_BRIX": {"gt": 40}, "Z_FRUCHTG": {"gte": 30}}
            categorical_filters: Optional dict mapping field codes to Qdrant match filters
                Example: {"Z_INH04": {"value": "No"}, "Z_INH01H": {"value": "Yes"}}

        Returns:
            List of matching recipes with scores
        """
        try:
            if not query_features:
                logger.warning("No features provided for feature search")
                return []

            # Use EnhancedTwoStepRecipeManager for CONSISTENT encoding with indexing
            if self.feature_encoder is not None:
                # This ensures binary opposition mapping matches indexing exactly
                feature_vector = self.feature_encoder._create_feature_vector(
                    query_features, query_values, fit=False)

                logger.info(
                    f"Feature search: using EnhancedTwoStepRecipeManager encoding, "
                    f"vector shape={feature_vector.shape}"
                )
            else:
                # Fallback to basic encoding if encoder not available
                logger.warning(
                    "Feature encoder not available, using basic encoding")
                feature_text = self._create_feature_text(
                    query_features, query_values)

                if not feature_text:
                    logger.warning(
                        "No valid features provided for feature search")
                    return []

                # Create embedding for the feature text (384 dim)
                text_embedding = self.embedding_model.encode(feature_text)

                # Create categorical encoding (100 dim)
                categorical_vector = self._encode_mixed_features(
                    query_features, query_values)

                # Combine to create full feature vector (484 dim)
                feature_vector = np.concatenate(
                    [text_embedding, categorical_vector])

                logger.info(
                    f"Feature search: text_embedding={text_embedding.shape}, "
                    f"categorical={categorical_vector.shape}, "
                    f"combined={feature_vector.shape}"
                )

            # Build filter if country or version is specified
            query_filter = None
            filter_conditions = []

            if country_filter and country_filter != "All":
                if isinstance(country_filter, list):
                    # Multiple countries: use MatchAny
                    if len(country_filter) > 0:
                        filter_conditions.append(
                            FieldCondition(
                                key="country",
                                match=MatchAny(any=country_filter)
                            )
                        )
                        logger.info(
                            f"Applying multi-country filter in feature search: {country_filter} ({len(country_filter)} countries)")
                    else:
                        logger.info("Empty country filter list, skipping country filter")
                else:
                    # Single country: use MatchValue (backward compatibility)
                    filter_conditions.append(
                        FieldCondition(
                            key="country",
                            match=MatchValue(value=country_filter)
                        )
                    )
                    logger.info(
                        f"Applying single country filter in feature search: {country_filter}")

            if version_filter and version_filter != "All":
                filter_conditions.append(
                    FieldCondition(
                        key="version",
                        match=MatchValue(value=version_filter)
                    )
                )
                logger.info(
                    f"Applying version filter in feature search: {version_filter}")

            # Add numerical range filters (e.g., Brix > 40, pH < 4.1)
            if numerical_filters:
                for field_code, range_spec in numerical_filters.items():
                    # Build Qdrant Range object from filter spec
                    # range_spec can be: {"gt": 40}, {"lt": 4.1}, {"gte": 30, "lte": 50}, etc.
                    range_obj = Range(
                        gt=range_spec.get('gt'),
                        gte=range_spec.get('gte'),
                        lt=range_spec.get('lt'),
                        lte=range_spec.get('lte')
                    )
                    
                    # Filter on the numerical payload field (e.g., "numerical.Z_BRIX")
                    filter_conditions.append(
                        FieldCondition(
                            key=f"numerical.{field_code}",
                            range=range_obj
                        )
                    )
                    logger.info(
                        f"Applying numerical range filter: {field_code} → {range_spec}")

            # Add categorical exact-match filters (e.g., Preservative: No, Halal: Yes)
            if categorical_filters:
                for field_code, match_spec in categorical_filters.items():
                    # Categorical filters use spec_fields.Z_xxx for the payload path
                    filter_conditions.append(
                        FieldCondition(
                            key=f"spec_fields.{field_code}",
                            match=MatchValue(value=match_spec.get('value'))
                        )
                    )
                    logger.info(
                        f"Applying categorical exact-match filter in feature search: {field_code} → {match_spec}")

            if filter_conditions:
                query_filter = Filter(must=filter_conditions)

            # Search using the "features" named vector
            try:
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=("features", feature_vector.tolist()),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                    timeout=60  # Increased timeout for large databases with filters
                )
            except Exception as e:
                logger.warning(f"Feature vector search failed: {e}")
                return []

            # Format results
            results = []
            for result in search_results:
                payload = result.payload if result.payload is not None else {}
                recipe_data = {
                    "id": result.id,
                    "feature_search_score": float(result.score),
                    "text_score": 0.0,  # Will be updated if also found in text search
                    "recipe_name": payload.get("recipe_name", ""),
                    "description": payload.get("description", ""),
                    "features": payload.get("features", []),
                    "values": payload.get("values", []),
                    "num_features": payload.get("num_features", 0),
                    "metadata": {
                        "recipe_name": payload.get("recipe_name", "")
                    },
                    "search_source": "features",
                    "payload": payload  # Include full payload for comparison table
                }
                results.append(recipe_data)

            logger.info(
                f"Feature search found {len(results)} recipes for {len(query_features)} features")
            return results

        except Exception as e:
            logger.error(f"Error in feature search: {e}")
            return []

    def search_by_text_description(self,
                                   text_description: str,
                                   top_k: int = 20,
                                   country_filter: Optional[Union[str, List[str]]] = None,
                                   version_filter: Optional[str] = None,
                                   numerical_filters: Optional[Dict[str, Dict[str, Any]]] = None,
                                   categorical_filters: Optional[Dict[str, Dict[str, Any]]] = None,
                                   return_embedding: bool = False):
        """
        Search recipes by text description using Qdrant vector search

        Args:
            text_description: Text description for search
            top_k: Number of results to return
            country_filter: Optional country name(s) to filter results. Can be a single string or a list of strings. None or "All" means no filter.
            version_filter: Optional version filter (P, L, Missing, or "All" means no filter)
            numerical_filters: Optional dict mapping field codes to Qdrant range filters
                Example: {"Z_BRIX": {"gt": 40}, "Z_FRUCHTG": {"gte": 30}}
            categorical_filters: Optional dict mapping field codes to Qdrant match filters
                Example: {"Z_INH04": {"value": "No"}, "Z_INH01H": {"value": "Yes"}}
            return_embedding: If True, also return the query embedding for reuse

        Returns:
            List of matching recipes with scores
            If return_embedding=True: Tuple of (results, query_embedding)
        """
        try:
            # Create embedding for query using SentenceTransformer
            query_vector = self.embedding_model.encode(text_description)

            # Build filter if country or version is specified
            query_filter = None
            filter_conditions = []

            if country_filter and country_filter != "All":
                if isinstance(country_filter, list):
                    # Multiple countries: use MatchAny
                    if len(country_filter) > 0:
                        filter_conditions.append(
                            FieldCondition(
                                key="country",
                                match=MatchAny(any=country_filter)
                            )
                        )
                        logger.info(
                            f"Applying multi-country filter in text search: {country_filter} ({len(country_filter)} countries)")
                    else:
                        logger.info("Empty country filter list, skipping country filter")
                else:
                    # Single country: use MatchValue (backward compatibility)
                    filter_conditions.append(
                        FieldCondition(
                            key="country",
                            match=MatchValue(value=country_filter)
                        )
                    )
                    logger.info(
                        f"Applying single country filter in text search: {country_filter}")

            if version_filter and version_filter != "All":
                filter_conditions.append(
                    FieldCondition(
                        key="version",
                        match=MatchValue(value=version_filter)
                    )
                )
                logger.info(f"Applying version filter: {version_filter}")

            # Add numerical range filters (e.g., Brix > 40, pH < 4.1)
            if numerical_filters:
                for field_code, range_spec in numerical_filters.items():
                    range_obj = Range(
                        gt=range_spec.get('gt'),
                        gte=range_spec.get('gte'),
                        lt=range_spec.get('lt'),
                        lte=range_spec.get('lte')
                    )
                    filter_conditions.append(
                        FieldCondition(
                            key=f"numerical.{field_code}",
                            range=range_obj
                        )
                    )
                    logger.info(
                        f"Applying numerical range filter in text search: {field_code} → {range_spec}")

            # Add categorical exact-match filters (e.g., Preservative: No, Halal: Yes)
            if categorical_filters:
                for field_code, match_spec in categorical_filters.items():
                    # Categorical filters use spec_fields.Z_xxx for the payload path
                    filter_conditions.append(
                        FieldCondition(
                            key=f"spec_fields.{field_code}",
                            match=MatchValue(value=match_spec.get('value'))
                        )
                    )
                    logger.info(
                        f"Applying categorical exact-match filter in text search: {field_code} → {match_spec}")

            if filter_conditions:
                query_filter = Filter(must=filter_conditions)

            # Try searching with named vector first (new format)
            # If that fails, fallback to unnamed vector (old format)
            try:
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    # Named vector
                    query_vector=("text", query_vector.tolist()),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                    timeout=60  # Increased timeout for large databases with filters
                )
            except Exception as e:
                logger.warning(
                    f"Named vector search failed, trying unnamed vector: {e}")
                # Fallback to unnamed vector for backward compatibility
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                    timeout=60  # Increased timeout for large databases with filters
                )

            # Format results
            results = []
            for result in search_results:
                payload = result.payload if result.payload is not None else {}
                recipe_data = {
                    "id": result.id,
                    "text_score": float(result.score),
                    "feature_search_score": 0.0,  # Will be updated if also found in feature search
                    "recipe_name": payload.get("recipe_name", ""),
                    "description": payload.get("description", ""),
                    "features": payload.get("features", []),
                    "values": payload.get("values", []),
                    "num_features": payload.get("num_features", 0),
                    "metadata": {
                        "recipe_name": payload.get("recipe_name", "")
                    },
                    "search_source": "text",
                    "payload": payload  # Include full payload for comparison table
                }
                results.append(recipe_data)

            logger.info(
                f"Text search found {len(results)} recipes for query: '{text_description[:50]}...'")

            if return_embedding:
                return results, query_vector
            return results

        except Exception as e:
            logger.error(f"Error in Qdrant search: {e}")
            if return_embedding:
                return [], None
            return []

    def _merge_candidates(self,
                          text_candidates: List[Dict[str, Any]],
                          feature_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge candidates from text and feature searches, deduplicating by ID.

        Args:
            text_candidates: Candidates from text-based search
            feature_candidates: Candidates from feature-based search

        Returns:
            Merged and deduplicated list of candidates
        """
        merged = {}

        # Add text candidates first
        for candidate in text_candidates:
            recipe_id = candidate["id"]
            merged[recipe_id] = candidate.copy()
            merged[recipe_id]["search_source"] = "text"

        # Add or merge feature candidates
        for candidate in feature_candidates:
            recipe_id = candidate["id"]
            if recipe_id in merged:
                # Recipe found in both searches - merge scores
                merged[recipe_id]["feature_search_score"] = candidate.get(
                    "feature_search_score", 0.0)
                merged[recipe_id]["search_source"] = "both"
            else:
                # New recipe from feature search only
                merged[recipe_id] = candidate.copy()
                merged[recipe_id]["search_source"] = "features"

        return list(merged.values())

    def _calculate_text_scores_for_feature_only(self,
                                                candidates: List[Dict[str, Any]],
                                                text_description: str,
                                                query_embedding: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Calculate text similarity scores for candidates found only in feature search.

        This ensures fair comparison in the combined scoring by computing text scores
        for recipes that were discovered via feature search but not text search.

        OPTIMIZED: Uses batch encoding and accepts pre-computed query embedding.

        Args:
            candidates: List of merged candidates
            text_description: The query text description
            query_embedding: Optional pre-computed query embedding (avoids re-encoding)

        Returns:
            Updated candidates with text scores calculated for feature-only recipes
        """
        # Find feature-only candidates that need text scores
        feature_only_candidates = [
            c for c in candidates if c.get("search_source") == "features"
        ]

        if not feature_only_candidates:
            return candidates

        logger.info(
            f"Calculating text scores for {len(feature_only_candidates)} feature-only candidates")

        try:
            # Use pre-computed query embedding if available, otherwise compute it
            if query_embedding is None:
                query_embedding = self.embedding_model.encode(text_description)

            # OPTIMIZATION: Batch encode all candidate descriptions at once
            descriptions = []
            valid_indices = []
            for i, candidate in enumerate(feature_only_candidates):
                description = candidate.get("description", "")
                if description:
                    descriptions.append(description)
                    valid_indices.append(i)
                else:
                    candidate["text_score"] = 0.0

            if descriptions:
                # Single batch encode call instead of multiple individual calls
                candidate_embeddings = self.embedding_model.encode(
                    descriptions,
                    batch_size=32,
                    show_progress_bar=False
                )

                # Calculate cosine similarities in batch
                similarities = cosine_similarity(
                    [query_embedding], candidate_embeddings
                )[0]

                # Assign scores
                for idx, valid_idx in enumerate(valid_indices):
                    candidate = feature_only_candidates[valid_idx]
                    similarity = float(similarities[idx])
                    candidate["text_score"] = similarity
                    logger.info(
                        f"  Calculated text score for {candidate.get('recipe_name', 'Unknown')}: {similarity:.4f}"
                    )

        except Exception as e:
            logger.error(f"Error calculating text scores: {e}")
            # Keep text_score as 0.0 if calculation fails

        return candidates

    def _calculate_feature_scores_for_keyword_matches(self,
                                                      candidates: List[Dict[str, Any]],
                                                      query_features: List[str],
                                                      query_values: List[Any]) -> List[Dict[str, Any]]:
        """
        Calculate feature search scores for candidates found via keyword search.

        This gives keyword-matched recipes a fair chance to compete with feature-search
        results by computing the same cosine similarity score they would have received
        if they had been found via feature search.

        Args:
            candidates: List of keyword-matched candidates needing feature scores
            query_features: Query feature names
            query_values: Query feature values

        Returns:
            Updated candidates with feature_search_score calculated
        """
        if not candidates or not query_features:
            return candidates

        try:
            logger.info(
                f"Calculating feature scores for {len(candidates)} keyword-matched candidates")

            # Create query feature vector using the same method as feature search
            if self.feature_encoder is not None:
                query_feature_vector = self.feature_encoder._create_feature_vector(
                    query_features, query_values, fit=False)
            else:
                # Fallback to basic encoding
                feature_text = self._create_feature_text(
                    query_features, query_values)
                if not feature_text:
                    return candidates
                text_embedding = self.embedding_model.encode(feature_text)
                categorical_vector = self._encode_mixed_features(
                    query_features, query_values)
                query_feature_vector = np.concatenate(
                    [text_embedding, categorical_vector])

            # Get recipe IDs to retrieve their feature vectors
            recipe_ids = [c.get("id") for c in candidates if c.get("id")]

            if not recipe_ids:
                return candidates

            # Retrieve feature vectors from Qdrant for these specific recipes
            try:
                points = self.qdrant_client.retrieve(
                    collection_name=self.collection_name,
                    ids=recipe_ids,
                    with_payload=False,
                    with_vectors=["features"]  # Only get the features vector
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve feature vectors: {e}")
                return candidates

            # Build a map of recipe_id -> feature_vector
            recipe_vectors = {}
            for point in points:
                if point.vector and "features" in point.vector:
                    recipe_vectors[point.id] = np.array(
                        point.vector["features"])

            # Calculate cosine similarity for each candidate
            for candidate in candidates:
                recipe_id = candidate.get("id")
                if recipe_id in recipe_vectors:
                    recipe_vector = recipe_vectors[recipe_id]

                    # Cosine similarity
                    dot_product = np.dot(query_feature_vector, recipe_vector)
                    query_norm = np.linalg.norm(query_feature_vector)
                    recipe_norm = np.linalg.norm(recipe_vector)

                    if query_norm > 0 and recipe_norm > 0:
                        similarity = dot_product / (query_norm * recipe_norm)
                        candidate["feature_search_score"] = float(similarity)
                        mst_part = ""
                        cand_desc = candidate.get("description", "")
                        if "MaterialMasterShorttext:" in cand_desc:
                            try:
                                mst_part = cand_desc.split("MaterialMasterShorttext:")[1].split(",")[0].strip()
                            except Exception:
                                mst_part = ""
                        display_name = candidate.get('recipe_name', 'Unknown')[:50]
                        if mst_part:
                            display_name = f"{display_name} | MST: {mst_part[:60]}"
                        logger.info(
                            f"  Calculated feature score for {display_name}: {similarity:.4f}"
                        )
                    else:
                        candidate["feature_search_score"] = 0.0
                else:
                    candidate["feature_search_score"] = 0.0

        except Exception as e:
            logger.error(
                f"Error calculating feature scores for keyword matches: {e}")
            # Keep feature_search_score as 0.0 if calculation fails

        return candidates

    def _check_exact_recipe_name_match(self,
                                       query: str,
                                       country_filter: Optional[Union[str, List[str]]] = None,
                                       version_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Check if the query is an exact or partial recipe name match.

        This handles cases where users search with just a recipe name like
        "FZ APFEL MATCHA SIGGIS o.A." or partial names like "FZ Orange Mango Grüner Tee"
        which should find "FZ Orange Mango Grüner Tee Smoothie".

        Args:
            query: The search query (potentially a recipe name)
            country_filter: Optional country name(s) to filter results. Can be a single string or a list of strings.
            version_filter: Optional version filter

        Returns:
            List of recipe dicts with high text_score (0.95 for exact, 0.90 for partial) if matches found
        """
        matches = []
        try:
            # Heuristic: Check if query looks like a recipe name
            # Recipe names are typically:
            # - Short to medium length (20-150 chars)
            # - May contain underscores, numbers, abbreviations
            # - Often start with codes like "FZ", "ZB", etc.
            query_clean = query.strip()

            # Skip if query is too long (likely a description, not a name)
            if len(query_clean) > 150:
                return []

            # Skip if query is too short (likely not a recipe name)
            if len(query_clean) < 5:
                return []

            # Try to find exact/partial match in description field (where MaterialMasterShorttext is stored)
            # Recipe names are typically in format: "MaterialMasterShorttext: FZ APFEL MATCHA SIGGIS o.A."
            # So we search for the query text in the description field using full-text search
            filter_conditions = [
                FieldCondition(
                    key="description",
                    match=MatchText(text=query_clean)
                )
            ]

            # Add country filter if specified
            if country_filter and country_filter != "All":
                if isinstance(country_filter, list):
                    # Multiple countries: use MatchAny
                    if len(country_filter) > 0:
                        filter_conditions.append(
                            FieldCondition(
                                key="country",
                                match=MatchAny(any=country_filter)
                            )
                        )
                        logger.info(
                            f"Applying multi-country filter in recipe name match: {country_filter} ({len(country_filter)} countries)")
                    # else: empty list, skip filter
                else:
                    # Single country: use MatchValue (backward compatibility)
                    filter_conditions.append(
                        FieldCondition(
                            key="country",
                            match=MatchValue(value=country_filter)
                        )
                    )
                    logger.info(
                        f"Applying single country filter in recipe name match: {country_filter}")

            # Add version filter if specified
            if version_filter and version_filter != "All":
                filter_conditions.append(
                    FieldCondition(
                        key="version",
                        match=MatchValue(value=version_filter)
                    )
                )

            exact_filter = Filter(must=filter_conditions)

            # Search for matches using scroll (more efficient for exact matches)
            scroll_results, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=exact_filter,
                limit=20,  # Get more results to find best matches
                with_payload=True,
                with_vectors=False,
                timeout=60  # Increased timeout for large databases with filters
            )

            if scroll_results and len(scroll_results) > 0:
                # Find best matches - prioritize exact matches, then partial matches
                query_lower = query_clean.lower()
                exact_matches = []
                partial_matches = []

                for point in scroll_results:
                    payload = point.payload if point.payload is not None else {}
                    description = payload.get("description", "")

                    # Check if query appears in MaterialMasterShorttext part of description
                    # Format is typically: "MaterialMasterShorttext: FZ APFEL MATCHA SIGGIS o.A., ..."
                    mst_part = ""
                    if "MaterialMasterShorttext:" in description:
                        # Extract the MaterialMasterShorttext value
                        mst_part = description.split("MaterialMasterShorttext:")[
                            1].split(",")[0].strip()
                        
                        if query_lower == mst_part.lower():
                            # Exact match - highest priority
                            exact_matches.append((point, 0.95))
                        elif query_lower in mst_part.lower():
                            # Partial match - query is contained in recipe name
                            partial_matches.append((point, 0.92))
                        elif mst_part.lower().startswith(query_lower):
                            # Prefix match - recipe name starts with query
                            partial_matches.append((point, 0.90))
                
                # Sort partial matches by how similar the name length is (prefer shorter, closer matches)
                partial_matches.sort(key=lambda x: len(x[0].payload.get("description", "")))
                
                # Combine: exact matches first, then partial matches
                all_matches = exact_matches + partial_matches[:10]  # Limit to top 10 partial matches
                
                logger.info(f"Recipe name match search for '{query_clean}': "
                           f"found {len(exact_matches)} exact, {len(partial_matches)} partial matches")
                
                for point, score in all_matches:
                    payload = point.payload if point.payload is not None else {}
                    
                    # Create recipe data with appropriate text score
                    recipe_data = {
                        "id": point.id,
                        "text_score": score,  # 0.95 for exact, 0.92/0.90 for partial
                        "feature_search_score": 0.0,
                        "recipe_name": payload.get("recipe_name", ""),
                        "description": payload.get("description", ""),
                        "features": payload.get("features", []),
                        "values": payload.get("values", []),
                        "num_features": payload.get("num_features", 0),
                        "metadata": {
                            "recipe_name": payload.get("recipe_name", "")
                        },
                        "search_source": "text",
                        "_name_match": True,  # Flag to indicate this is a name match
                        "_match_score": score,
                        "payload": payload  # Include full payload for comparison table
                    }
                    matches.append(recipe_data)
                    
                    logger.info(
                        f"  ✓ Name match (score={score:.2f}): '{query_clean}' → "
                        f"{recipe_data.get('recipe_name', 'Unknown')[:60]}"
                    )

            return matches

        except Exception as e:
            # Don't fail the whole search if name match check fails
            logger.debug(f"Recipe name match check failed: {e}")
            return []

    def _apply_filters_to_candidates(self,
                                      candidates: List[Dict[str, Any]],
                                      numerical_filters: Optional[Dict[str, Dict[str, Any]]] = None,
                                      categorical_filters: Optional[Dict[str, Dict[str, Any]]] = None,
                                      relaxation_level: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Apply numerical and categorical filters to candidates in-memory.

        This is used in the new search architecture where:
        1. Text search finds top 100 semantically similar recipes (without filters)
        2. Filters are applied to those 100 candidates (this method)

        Args:
            candidates: List of candidate recipes with payload data
            numerical_filters: Dict of field codes to range filters
                Example: {"Z_BRIX": {"gt": 40}, "Z_FRUCHTG": {"gte": 30, "lte": 50}}
            categorical_filters: Dict of field codes to exact match filters
                Example: {"Z_INH04": {"value": "No"}, "Z_INH01H": {"value": "Yes"}}
            relaxation_level: Level of filter relaxation (0=strict, 1=10% tolerance, 2=20% tolerance, 3=no numerical)

        Returns:
            Tuple of (filtered_candidates, filter_stats)
        """
        if not candidates:
            return [], {"total": 0, "passed": 0, "relaxation_level": relaxation_level}

        if not numerical_filters and not categorical_filters:
            return candidates, {"total": len(candidates), "passed": len(candidates), "relaxation_level": 0}

        filtered = []
        filter_stats = {
            "total": len(candidates),
            "passed": 0,
            "relaxation_level": relaxation_level,
            "numerical_matches": {},
            "categorical_matches": {},
            "failed_numerical": {},
            "failed_categorical": {},
            "categorical_unknown": {},
            "categorical_mismatch": {}
        }

        # Initialize counters
        if numerical_filters:
            for field_code in numerical_filters:
                filter_stats["numerical_matches"][field_code] = 0
                filter_stats["failed_numerical"][field_code] = 0
        if categorical_filters:
            for field_code in categorical_filters:
                filter_stats["categorical_matches"][field_code] = 0
                filter_stats["failed_categorical"][field_code] = 0
                filter_stats["categorical_unknown"][field_code] = 0
                filter_stats["categorical_mismatch"][field_code] = 0

        # Calculate tolerance based on relaxation level
        # Level 0: strict (exact match)
        # Level 1: 10% tolerance on numerical
        # Level 2: 20% tolerance on numerical
        # Level 3: Skip numerical filters entirely
        numerical_tolerance = 0.0
        if relaxation_level == 1:
            numerical_tolerance = 0.10
        elif relaxation_level == 2:
            numerical_tolerance = 0.20
        elif relaxation_level >= 3:
            numerical_tolerance = float('inf')  # Skip numerical filters

        for candidate in candidates:
            payload = candidate.get("payload", {})
            passed_all = True
            passed_strict = True
            numerical_penalty = 0.0
            categorical_unknowns = []
            categorical_mismatches = []
            categorical_penalty = 0.0

            # Check numerical filters
            if numerical_filters and relaxation_level < 3:
                for field_code, range_spec in numerical_filters.items():
                    # Get the numerical value from payload
                    numerical_data = payload.get("numerical", {})
                    value = numerical_data.get(field_code)

                    if value is None:
                        # Field not present - fail the filter
                        filter_stats["failed_numerical"][field_code] = filter_stats["failed_numerical"].get(field_code, 0) + 1
                        numerical_penalty += 0.25
                        passed_strict = False
                        continue

                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        filter_stats["failed_numerical"][field_code] = filter_stats["failed_numerical"].get(field_code, 0) + 1
                        numerical_penalty += 0.25
                        passed_strict = False
                        continue

                    # Apply range checks with tolerance
                    if "gt" in range_spec:
                        threshold = range_spec["gt"]
                        # With tolerance: value > threshold * (1 - tolerance)
                        if value <= threshold * (1 - numerical_tolerance):
                            filter_stats["failed_numerical"][field_code] = filter_stats["failed_numerical"].get(field_code, 0) + 1
                            allowed_min = threshold * (1 - numerical_tolerance)
                            denom = max(abs(allowed_min), 1e-6)
                            numerical_penalty += max(0.0, (allowed_min - value) / denom)
                            passed_strict = False
                            continue

                    if "gte" in range_spec:
                        threshold = range_spec["gte"]
                        if value < threshold * (1 - numerical_tolerance):
                            filter_stats["failed_numerical"][field_code] = filter_stats["failed_numerical"].get(field_code, 0) + 1
                            allowed_min = threshold * (1 - numerical_tolerance)
                            denom = max(abs(allowed_min), 1e-6)
                            numerical_penalty += max(0.0, (allowed_min - value) / denom)
                            passed_strict = False
                            continue

                    if "lt" in range_spec:
                        threshold = range_spec["lt"]
                        if value >= threshold * (1 + numerical_tolerance):
                            filter_stats["failed_numerical"][field_code] = filter_stats["failed_numerical"].get(field_code, 0) + 1
                            allowed_max = threshold * (1 + numerical_tolerance)
                            denom = max(abs(allowed_max), 1e-6)
                            numerical_penalty += max(0.0, (value - allowed_max) / denom)
                            passed_strict = False
                            continue

                    if "lte" in range_spec:
                        threshold = range_spec["lte"]
                        if value > threshold * (1 + numerical_tolerance):
                            filter_stats["failed_numerical"][field_code] = filter_stats["failed_numerical"].get(field_code, 0) + 1
                            allowed_max = threshold * (1 + numerical_tolerance)
                            denom = max(abs(allowed_max), 1e-6)
                            numerical_penalty += max(0.0, (value - allowed_max) / denom)
                            passed_strict = False
                            continue

                    # Passed this numerical filter
                    filter_stats["numerical_matches"][field_code] = filter_stats["numerical_matches"].get(field_code, 0) + 1

            # Check categorical filters
            if categorical_filters and passed_all:
                for field_code, match_spec in categorical_filters.items():
                    expected_value = match_spec.get("value")
                    is_strict = match_spec.get("strict", False)

                    # Get the categorical value from payload (spec_fields)
                    spec_fields = payload.get("spec_fields", {})
                    actual_value = spec_fields.get(field_code)

                    if actual_value is None:
                        # Field not present - fail the filter
                        if is_strict:
                            passed_all = False
                            filter_stats["failed_categorical"][field_code] = filter_stats["failed_categorical"].get(field_code, 0) + 1
                            continue
                        # Treat missing categorical field as unknown (soft)
                        filter_stats["categorical_unknown"][field_code] = filter_stats["categorical_unknown"].get(field_code, 0) + 1
                        categorical_unknowns.append(field_code)
                        categorical_penalty += 0.15
                        passed_strict = False
                        continue

                    # Normalize for comparison
                    actual_norm = str(actual_value).strip().lower()
                    expected_norm = str(expected_value).strip().lower()

                    # Handle Yes/No variations
                    yes_values = {'yes', 'ja', 'oui', 'si', 'true', '1'}
                    no_values = {'no', 'nein', 'non', 'false', '0'}

                    if expected_norm in yes_values:
                        if actual_norm not in yes_values:
                            if is_strict:
                                passed_all = False
                                filter_stats["failed_categorical"][field_code] = filter_stats["failed_categorical"].get(field_code, 0) + 1
                                continue
                            filter_stats["categorical_mismatch"][field_code] = filter_stats["categorical_mismatch"].get(field_code, 0) + 1
                            categorical_mismatches.append(field_code)
                            categorical_penalty += 0.30
                            passed_strict = False
                            continue
                    elif expected_norm in no_values:
                        if actual_norm not in no_values:
                            if is_strict:
                                passed_all = False
                                filter_stats["failed_categorical"][field_code] = filter_stats["failed_categorical"].get(field_code, 0) + 1
                                continue
                            filter_stats["categorical_mismatch"][field_code] = filter_stats["categorical_mismatch"].get(field_code, 0) + 1
                            categorical_mismatches.append(field_code)
                            categorical_penalty += 0.30
                            passed_strict = False
                            continue
                    elif actual_norm != expected_norm:
                        if is_strict:
                            passed_all = False
                            filter_stats["failed_categorical"][field_code] = filter_stats["failed_categorical"].get(field_code, 0) + 1
                            continue
                        filter_stats["categorical_mismatch"][field_code] = filter_stats["categorical_mismatch"].get(field_code, 0) + 1
                        categorical_mismatches.append(field_code)
                        categorical_penalty += 0.30
                        passed_strict = False
                        continue

                    # Passed this categorical filter
                    filter_stats["categorical_matches"][field_code] = filter_stats["categorical_matches"].get(field_code, 0) + 1

            if passed_all:
                if passed_strict:
                    filter_stats["passed"] += 1
                # Mark if filters were relaxed
                if relaxation_level > 0:
                    candidate["_filter_relaxed"] = True
                    candidate["_relaxation_level"] = relaxation_level
                if numerical_penalty > 0:
                    candidate["_numerical_penalty"] = numerical_penalty
                if categorical_unknowns:
                    candidate["_categorical_unknowns"] = categorical_unknowns
                if categorical_mismatches:
                    candidate["_categorical_mismatches"] = categorical_mismatches
                if categorical_penalty > 0:
                    candidate["_categorical_penalty"] = categorical_penalty
                filtered.append(candidate)

        return filtered, filter_stats

    def search_two_step(self,
                        text_description: str,
                        query_df: Optional[pd.DataFrame] = None,
                        text_top_k: int = 50,
                        final_top_k: int = 10,
                        country_filter: Optional[Union[str, List[str]]] = None,
                        version_filter: Optional[str] = None,
                        original_query: Optional[str] = None,
                        numerical_filters: Optional[Dict[str, Dict[str, Any]]] = None,
                        categorical_filters: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Two-step hybrid search with TEXT-FIRST architecture:
        - Step 0: Exact/partial recipe name matching (unchanged)
        - Step 1a: Text-based semantic search for top 100 similar recipes (NO feature filters)
        - Step 1b: Apply numerical/categorical filters ON those 100 candidates
        - Step 2: Feature-based refinement and ranking

        This architecture ensures:
        1. Semantically similar recipes are found first (no legume when searching for mango)
        2. Filters are applied to the relevant pool, not to the entire 600K+ database
        3. Progressive filter relaxation if strict filters return 0 results

        Args:
            text_description: Recipe description for search
            query_df: Optional DataFrame with features for refinement
            text_top_k: Number of text search candidates (now 100 by default)
            final_top_k: Final number of results
            country_filter: Optional country name(s) to filter results. Can be a single string or a list of strings. None or "All" means no filter.
            version_filter: Optional version filter (P, L, Missing, or "All" means no filter)
            numerical_filters: Optional dict mapping field codes to Qdrant range filters
                Example: {"Z_BRIX": {"gt": 40}, "Z_FRUCHTG": {"gte": 30}}
            categorical_filters: Optional dict mapping field codes to Qdrant match filters
                Example: {"Z_INH04": {"value": "No"}, "Z_INH01H": {"value": "Yes"}}

        Returns:
            Tuple of (results, metadata)
        """
        # Use 100 candidates for text search (the main change)
        TEXT_SEARCH_K = 100

        search_metadata = {
            "search_type": "two_step_text_first",
            "text_description": text_description,
            "has_feature_refinement": query_df is not None,
            "text_candidates_requested": TEXT_SEARCH_K,
            "final_results": final_top_k,
            "country_filter": country_filter,
            "version_filter": version_filter,
            "numerical_filters": numerical_filters if numerical_filters else {},
            "categorical_filters": categorical_filters if categorical_filters else {}
        }

        # Log the new architecture
        logger.info("=" * 80)
        logger.info("SEARCH ARCHITECTURE: Text-First (Semantic → Filter)")
        logger.info("=" * 80)

        def _log_search_text(step_label: str, text: Optional[str]) -> None:
            if text is None:
                logger.info(f"{step_label} search text: <none>")
                return
            cleaned = text.replace("\n", "\\n").strip()
            logger.info(f"{step_label} search text: '{cleaned}'")

        # Log filters that will be applied AFTER text search
        has_filters = bool(numerical_filters) or bool(categorical_filters)
        if numerical_filters:
            logger.info(f"Numerical filters to apply (Step 1b): {len(numerical_filters)} filter(s)")
            for field_code, range_spec in numerical_filters.items():
                logger.info(f"  - {field_code}: {range_spec}")

        if categorical_filters:
            logger.info(f"Categorical filters to apply (Step 1b): {len(categorical_filters)} filter(s)")
            for field_code, match_spec in categorical_filters.items():
                logger.info(f"  - {field_code}: {match_spec}")

        if not has_filters:
            logger.info("No numerical or categorical filters - will use text similarity only")

        # Extract features for feature-based refinement
        query_features = []
        query_values = []
        if query_df is not None and not query_df.empty:
            if 'charactDescr' in query_df.columns and 'valueCharLong' in query_df.columns:
                query_features = query_df['charactDescr'].tolist()
                query_values = query_df['valueCharLong'].tolist()
            else:
                query_features = query_df.iloc[:, 0].tolist()
                query_values = query_df.iloc[:, 1].tolist()

        # =====================================================================
        # Step 0: Check for exact/partial recipe name matches (UNCHANGED)
        # =====================================================================
        query_for_name_match = original_query if original_query else text_description
        selected_name_line = False
        # If we have explicit flavour terms, prefer those for name matching only when the name looks noisy
        flavor_terms = []
        if query_features and query_values:
            for feature, value in zip(query_features, query_values):
                if not feature or not value:
                    continue
                if str(feature).strip().lower() in ("flavour", "flavor", "geschmack"):
                    flavor_terms.extend(
                        [term.strip() for term in str(value).split(",") if term.strip()]
                    )

        if original_query and "\n" in original_query:
            # Prefer the best-looking name line (may be partial) when a full brief is provided.
            # This avoids choosing constraint lines or document headers.
            candidate_lines = []
            for line in original_query.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                lower = stripped.lower()
                if any(marker in lower for marker in ("[documents uploaded", "[extracted from document", "user description")):
                    continue
                candidate_lines.append(stripped)

            def _name_line_score(text: str, flavors: list) -> tuple:
                lower = text.lower()
                alpha = sum(1 for c in text if c.isalpha())
                upper = sum(1 for c in text if c.isupper())
                upper_ratio = (upper / max(alpha, 1)) if alpha else 0.0
                flavor_hits = sum(1 for term in flavors if term and term.lower() in lower)
                # Penalize common constraint lines
                penalty_terms = ("kein", "ohne", "nicht", "dosage", "frucht", "saccharose", "allergen", "stärke", "pektin")
                penalty = 1.0 if any(term in lower for term in penalty_terms) else 0.0
                score = (alpha / 10.0) + (upper_ratio * 2.0) + (flavor_hits * 2.0) - penalty
                return score, flavor_hits, penalty

            best_line = ""
            best_score = -1.0
            best_flavor_hits = 0
            best_penalty = 0.0
            for line in candidate_lines:
                score, hits, penalty = _name_line_score(line, flavor_terms)
                if score > best_score:
                    best_score = score
                    best_line = line
                    best_flavor_hits = hits
                    best_penalty = penalty

            if best_line and (best_score >= 2.0 or (best_flavor_hits >= 1 and best_penalty == 0.0)):
                query_for_name_match = best_line
                selected_name_line = True
        if flavor_terms:
            noisy_markers = ("[documents uploaded", "[extracted from document", "user description")
            looks_like_name = bool(re.match(r"^[A-Z]{1,3}\b", query_for_name_match)) and len(query_for_name_match) <= 80
            has_noise = any(marker in query_for_name_match.lower() for marker in noisy_markers)
            if (not selected_name_line) and ((not query_for_name_match) or has_noise or (not looks_like_name)):
                # Use all extracted flavors to keep multi-flavor names searchable
                query_for_name_match = ", ".join(flavor_terms)
        _log_search_text("Step 0 (name match)", query_for_name_match)
        name_matches = self._check_exact_recipe_name_match(
            query_for_name_match, country_filter, version_filter)

        # =====================================================================
        # Step 1a: Text-based semantic search for top 100 candidates
        # NOTE: NO numerical/categorical filters here - only country/version
        # =====================================================================
        logger.info(f"Step 1a: Text-based semantic search (top {TEXT_SEARCH_K}) - NO feature filters")
        _log_search_text("Step 1a", text_description)

        text_search_result = self.search_by_text_description(
            text_description,
            top_k=TEXT_SEARCH_K,
            country_filter=country_filter,
            version_filter=version_filter,
            numerical_filters=None,  # NO filters in text search
            categorical_filters=None,  # NO filters in text search
            return_embedding=True
        )

        # Handle return value (may be tuple if return_embedding=True)
        if isinstance(text_search_result, tuple):
            text_candidates, cached_query_embedding = text_search_result
        else:
            text_candidates = text_search_result
            cached_query_embedding = None

        # Step 1a-: Optional name-line semantic search to recover misspellings/partials
        name_line_candidates = []
        if query_for_name_match and query_for_name_match.strip() and query_for_name_match != text_description:
            try:
                logger.info("Step 1a-: Name-line semantic search (top 50)")
                _log_search_text("Step 1a-", query_for_name_match)
                name_line_candidates = self.search_by_text_description(
                    query_for_name_match,
                    top_k=min(50, TEXT_SEARCH_K),
                    country_filter=country_filter,
                    version_filter=version_filter,
                    numerical_filters=None,
                    categorical_filters=None,
                    return_embedding=False
                )
                if isinstance(name_line_candidates, tuple):
                    name_line_candidates = name_line_candidates[0]
                for cand in name_line_candidates:
                    cand["search_source"] = "name_line"
                if name_line_candidates:
                    logger.info(
                        f"  Added {len(name_line_candidates)} name-line candidates"
                    )
            except Exception as e:
                logger.warning(f"Name-line semantic search failed: {e}")

        # Merge name matches with text candidates (name matches go first)
        if name_matches:
            name_match_ids = {nm.get("id") for nm in name_matches}
            text_candidates = [
                c for c in text_candidates
                if c.get("id") not in name_match_ids
            ]
            text_candidates = name_matches + text_candidates
            logger.info(f"  ✓ Found {len(name_matches)} recipe name match(es) for original query")

        # Merge name-line candidates (avoid duplicates)
        if name_line_candidates:
            existing_ids = {c.get("id") for c in text_candidates if c.get("id")}
            extra = [c for c in name_line_candidates if c.get("id") not in existing_ids]
            if extra:
                text_candidates.extend(extra)

        # Also search with original query if different (for German product names)
        if original_query and original_query.strip().lower() != text_description.strip().lower():
            logger.info(f"Step 1a+: Additional semantic search using original query")
            _log_search_text("Step 1a+", original_query)

            original_query_results = self.search_by_text_description(
                original_query,
                top_k=TEXT_SEARCH_K,
                country_filter=country_filter,
                version_filter=version_filter,
                numerical_filters=None,  # NO filters
                categorical_filters=None,  # NO filters
                return_embedding=False
            )

            if isinstance(original_query_results, tuple):
                original_query_results = original_query_results[0]

            # Merge with existing candidates
            existing_ids = {c.get("id") for c in text_candidates}
            added_from_original = 0
            for candidate in original_query_results:
                if candidate.get("id") not in existing_ids:
                    text_candidates.append(candidate)
                    existing_ids.add(candidate.get("id"))
                    added_from_original += 1

            if added_from_original > 0:
                logger.info(f"  Added {added_from_original} additional candidates from original query")

        search_metadata["text_candidates_found"] = len(text_candidates)
        logger.info(f"Step 1a complete: Found {len(text_candidates)} semantically similar candidates")

        # Log top 10 text candidates
        logger.info("Top 10 text search candidates:")
        for i, candidate in enumerate(text_candidates[:10], 1):
            recipe_name = candidate.get('recipe_name', 'Unknown')
            text_score = candidate.get('text_score', 0.0)
            description = candidate.get('description', '')[:60]
            logger.info(f"  {i}. {recipe_name} | Score: {text_score:.4f} | {description}...")

        # =====================================================================
        # Step 1b: Apply numerical/categorical filters ON the text candidates
        # =====================================================================
        all_candidates = text_candidates
        filter_stats = None

        if has_filters:
            logger.info(f"Step 1b: Applying filters to {len(text_candidates)} candidates")

            # Try strict filters first
            filtered_candidates, filter_stats = self._apply_filters_to_candidates(
                text_candidates,
                numerical_filters=numerical_filters,
                categorical_filters=categorical_filters,
                relaxation_level=0
            )

            logger.info(f"  Strict filters: {filter_stats['passed']}/{filter_stats['total']} candidates passed")

            # Progressive relaxation if strict filters return too few results
            if len(filtered_candidates) < final_top_k:
                # Level 1: 10% tolerance on numerical
                if numerical_filters and len(filtered_candidates) < final_top_k:
                    logger.info(f"  → Relaxing numerical filters (10% tolerance)...")
                    filtered_candidates, filter_stats = self._apply_filters_to_candidates(
                        text_candidates,
                        numerical_filters=numerical_filters,
                        categorical_filters=categorical_filters,
                        relaxation_level=1
                    )
                    logger.info(f"  10% tolerance: {filter_stats['passed']}/{filter_stats['total']} candidates passed")

                # Level 2: 20% tolerance on numerical
                if numerical_filters and len(filtered_candidates) < final_top_k:
                    logger.info(f"  → Relaxing numerical filters (20% tolerance)...")
                    filtered_candidates, filter_stats = self._apply_filters_to_candidates(
                        text_candidates,
                        numerical_filters=numerical_filters,
                        categorical_filters=categorical_filters,
                        relaxation_level=2
                    )
                    logger.info(f"  20% tolerance: {filter_stats['passed']}/{filter_stats['total']} candidates passed")

                # Level 3: Skip numerical filters entirely, keep only categorical
                if len(filtered_candidates) < final_top_k and numerical_filters:
                    logger.info(f"  → Skipping numerical filters (categorical only)...")
                    filtered_candidates, filter_stats = self._apply_filters_to_candidates(
                        text_candidates,
                        numerical_filters=numerical_filters,
                        categorical_filters=categorical_filters,
                        relaxation_level=3
                    )
                    logger.info(f"  Categorical only: {filter_stats['passed']}/{filter_stats['total']} candidates passed")

                # Level 4: No filters at all - return text search results with warning
                if len(filtered_candidates) == 0:
                    logger.warning(f"  ⚠ No candidates passed ANY filters - returning unfiltered results")
                    logger.warning(f"    Note: Results are semantically similar but may not match all specifications")
                    # Mark all candidates as not matching filters
                    for c in text_candidates:
                        c["_filter_warning"] = "No recipes matched all filters - showing semantically similar recipes"
                        c["_filter_relaxed"] = True
                        c["_relaxation_level"] = 4
                    filtered_candidates = text_candidates
                    filter_stats["relaxation_level"] = 4

            all_candidates = filtered_candidates
            search_metadata["filter_stats"] = filter_stats
            search_metadata["filtered_candidates"] = len(all_candidates)

            logger.info(f"Step 1b complete: {len(all_candidates)} candidates after filtering")
            if filter_stats["relaxation_level"] > 0:
                logger.info(f"  ⚠ Filter relaxation level: {filter_stats['relaxation_level']}")
        else:
            logger.info("Step 1b: Skipped (no numerical/categorical filters)")

        # =====================================================================
        # Calculate feature search scores for all candidates
        # This is done IN-MEMORY using the feature encoder, not via Qdrant search
        # =====================================================================
        if query_features and all_candidates:
            logger.info(f"Calculating feature scores for {len(all_candidates)} candidates...")
            all_candidates = self._calculate_feature_scores_for_keyword_matches(
                all_candidates, query_features, query_values
            )

        # =====================================================================
        # SAFEGUARD: Flavor boost - search for flavor keywords if present
        # This adds additional candidates that might have been missed
        # NOTE: In text-first architecture, we search WITHOUT filters first,
        # then apply filters in-memory for consistency
        # =====================================================================
        if query_features:
            FLAVOR_FEATURES = {'flavour', 'flavor', 'geschmack', 'aroma'}
            query_flavor = None
            for qf, qv in zip(query_features, query_values):
                if any(flav in qf.lower() for flav in FLAVOR_FEATURES):
                    query_flavor = str(qv).strip()
                    break

            if query_flavor:
                # Extract flavor keywords
                flavor_keywords = set()
                for flavor_phrase in query_flavor.split(','):
                    flavor_phrase = flavor_phrase.strip()
                    if flavor_phrase:
                        flavor_keywords.add(flavor_phrase.lower())
                        for word in flavor_phrase.split():
                            if len(word) >= 3:
                                flavor_keywords.add(word.lower())

                existing_ids = {c.get("id") for c in all_candidates}
                flavor_matched_candidates = []
                skipped_already_in_pool = 0

                # Vector search for flavor - NO FILTERS (consistent with text-first architecture)
                _log_search_text("Flavor safeguard", query_flavor)
                logger.info(f"Flavor safeguard: Searching for '{query_flavor}' (NO filters)")
                vector_flavor_results = self.search_by_text_description(
                    query_flavor,
                    top_k=50,
                    country_filter=country_filter,
                    version_filter=version_filter,
                    numerical_filters=None,  # NO filters in vector search
                    categorical_filters=None   # NO filters in vector search
                )

                logger.info(f"  Found {len(vector_flavor_results)} flavor candidates")

                # Add to pool (will be filtered later in refinement)
                for candidate in vector_flavor_results:
                    if candidate.get("id") in existing_ids:
                        skipped_already_in_pool += 1
                        continue
                    flavor_matched_candidates.append(candidate)
                    existing_ids.add(candidate.get("id"))

                # Apply filters to flavor candidates if filters are present
                if has_filters and flavor_matched_candidates:
                    pre_filter_count = len(flavor_matched_candidates)
                    flavor_matched_candidates, flavor_filter_stats = self._apply_filters_to_candidates(
                        flavor_matched_candidates,
                        numerical_filters=numerical_filters,
                        categorical_filters=categorical_filters,
                        relaxation_level=filter_stats.get("relaxation_level", 0) if filter_stats else 0
                    )
                    logger.info(f"  After filtering: {len(flavor_matched_candidates)}/{pre_filter_count} passed")

                logger.info(
                    f"Flavor safeguard: {skipped_already_in_pool} already in pool, "
                    f"{len(flavor_matched_candidates)} new flavor matches added"
                )

                # Add flavor-matched candidates to pool
                if flavor_matched_candidates:
                    # Calculate text scores for new candidates
                    for c in flavor_matched_candidates:
                        c["search_source"] = "flavor"
                    flavor_matched_candidates = self._calculate_text_scores_for_feature_only(
                        flavor_matched_candidates, text_description, cached_query_embedding
                    )

                    # Calculate feature scores for new candidates
                    if query_features:
                        self._calculate_feature_scores_for_keyword_matches(
                            flavor_matched_candidates, query_features, query_values
                        )

                    all_candidates.extend(flavor_matched_candidates)
                    logger.info(f"Added {len(flavor_matched_candidates)} flavor-matched recipes to candidate pool")

        # =====================================================================
        # Final candidate pool summary
        # =====================================================================
        search_metadata["total_candidates"] = len(all_candidates)

        if not all_candidates:
            logger.warning("No candidates found after all search steps")
            return [], search_metadata

        # Count sources
        source_counts = {}
        for c in all_candidates:
            src = c.get("search_source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        logger.info(f"Final candidate pool: {len(all_candidates)} recipes")
        for src, count in sorted(source_counts.items()):
            logger.info(f"  - {src}: {count}")

        # =====================================================================
        # Step 2: Feature refinement and final ranking
        # =====================================================================
        if query_features:
            logger.info("Step 2: Refining candidates with feature-based similarity")
            search_metadata["query_features_count"] = len(query_features)

            final_results = self._refine_by_features(
                all_candidates, query_features, query_values, final_top_k,
                original_query=original_query,
                name_match_query=query_for_name_match,
                numerical_filters=numerical_filters
            )
            search_metadata["refinement_completed"] = True

        else:
            logger.info("Step 2: Skipped (no features) - using text similarity only")
            # Sort by text_score when no features are provided
            all_candidates.sort(key=lambda x: x.get(
                "text_score", 0.0), reverse=True)
            final_results = all_candidates[:final_top_k]
            search_metadata["refinement_completed"] = False

        search_metadata["final_results_count"] = len(final_results)

        return final_results, search_metadata

    def _normalize_boolean(self, value: str) -> str:
        """Normalize boolean-like values to a standard format"""
        value = value.lower().strip()

        # Positive values
        if value in ['yes', 'ja', 'oui', 'si', 'sim', 'allowed', 'permitted', 'true', '1']:
            return 'yes'

        # Negative values
        if value in ['no', 'nein', 'non', 'não', 'not allowed', 'forbidden', 'false', '0']:
            return 'no'

        # Return original if not a boolean
        return value

    def _match_feature_value(self, query_val: str, cand_val: str) -> bool:
        """
        Enhanced feature value matching compatible with EnhancedTwoStepRecipeManager encoding.
        Uses multiple strategies for robust matching.
        """
        query_lower = query_val.lower().strip()
        cand_lower = cand_val.lower().strip()

        # Strategy 1: Exact match
        if query_lower == cand_lower:
            return True

        # Strategy 2: Normalized boolean match
        query_norm = self._normalize_boolean(query_val)
        cand_norm = self._normalize_boolean(cand_val)
        if query_norm in ['yes', 'no'] and cand_norm in ['yes', 'no']:
            if query_norm == cand_norm:
                return True

        # Strategy 3: Substring match (bidirectional)
        if query_lower in cand_lower or cand_lower in query_lower:
            return True

        # Strategy 4: Partial word match (for multi-word values like flavors)
        query_words = set(
            word for word in query_lower.split() if len(word) > 2)
        cand_words = set(word for word in cand_lower.split() if len(word) > 2)
        if query_words and cand_words and not query_words.isdisjoint(cand_words):
            return True

        return False

    def _refine_by_features(self,
                            candidates: List[Dict[str, Any]],
                            query_features: List[str],
                            query_values: List[Any],
                            top_k: int,
                            original_query: Optional[str] = None,
                            name_match_query: Optional[str] = None,
                            numerical_filters: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Refine candidates using feature-based similarity with language-independent scoring.

        The combined score uses three components:
        - Text score (15%): Language-dependent, minimal influence
        - Feature search score (65%): Partially language-independent (includes categorical encoding)
        - Feature refinement score (20%): Fully language-independent (exact feature matching)

        ENHANCED with:
        - Flavor boosting: Recipes matching the query Flavour get a significant bonus
        - Recipe name match boost: Recipes with names matching the original query get priority

        Args:
            candidates: Candidate recipes from text and feature search
            query_features: Feature names to match
            query_values: Feature values to match
            top_k: Number of results to return
            original_query: Original search query (for recipe name matching)

        Returns:
            Refined and reranked results
        """
        try:
            # Define flavor-related feature names
            FLAVOR_FEATURES = {'flavour', 'flavor', 'geschmack', 'aroma'}
            FLAVOR_BONUS = 0.20  # Base bonus added to combined score for flavor match
            # Extra boost for text-only candidates (compensates for missing feature search score)
            TEXT_ONLY_FLAVOR_BOOST = 0.25  # Additional boost when feature_search_score is 0
            # Boost for exact recipe name matches (when user searches by recipe name)
            RECIPE_NAME_EXACT_BOOST = 0.50  # Significant boost for exact name match
            RECIPE_NAME_PARTIAL_BOOST = 0.35  # Boost for partial/substring name match
            
            # Flag to track if this is a short query (recipe name search mode)
            is_short_query_mode = False

            # Pre-compute recipe name matching for all candidates
            # Heuristic: if query is SHORT (< 50 chars) AND no specific features/constraints extracted,
            # treat it as a recipe name search and boost candidates based on string similarity
            recipe_name_matches = {}  # candidate_id -> (match_type, boost, similarity)
            
            # Determine if this is a "recipe name search" vs "feature-based search"
            # Recipe name search mode is used when:
            # - Query is short (< 50 chars)
            # - No numerical constraints extracted (like pH, Brix, fruit content %)
            # - Few or generic features (≤ 2, typically just Flavour and Produktsegment)
            num_features = len(query_features)
            has_numerical_filters = numerical_filters and len(numerical_filters) > 0
            
            # Check for "specific" features beyond Flavour and Produktsegment
            generic_feature_names = {'flavour', 'flavor', 'produktsegment', 'produktsegment (sd reporting)'}
            specific_features = [f for f in query_features if f.lower() not in generic_feature_names]
            has_specific_features = len(specific_features) > 0

            name_similarity_query = None
            if name_match_query:
                cleaned_name_query = name_match_query.strip()
                if 3 <= len(cleaned_name_query) <= 80:
                    name_similarity_query = cleaned_name_query
            
            if original_query:
                query_clean = original_query.strip()
                query_len = len(query_clean)
                
                # Short queries (< 50 chars) are likely recipe name searches
                # BUT only if no specific features/constraints or numerical filters were extracted
                if query_len < 50 and query_len >= 3 and not has_specific_features and not has_numerical_filters:
                    is_short_query_mode = True
                    query_lower = query_clean.lower()
                    query_words = set(query_lower.split())
                    logger.info(
                        f"Short query detected ({query_len} chars, {num_features} features, no specific constraints) "
                        f"- enabling recipe name similarity matching for: '{query_clean}'"
                    )
                    
                    # Define similarity calculation function for name matching
                    def calculate_name_similarity(query: str, candidate_name: str) -> float:
                        """Calculate similarity between query and candidate name (0.0 to 1.0)"""
                        if not candidate_name:
                            return 0.0
                        q = query.lower()
                        c = candidate_name.lower()
                        
                        # Exact match
                        if q == c:
                            return 1.0
                        
                        # Query is substring of candidate name
                        if q in c:
                            return 0.85 + (len(q) / len(c)) * 0.15  # 0.85-1.0 based on coverage
                        
                        # Candidate name is substring of query
                        if c in q:
                            return 0.80 + (len(c) / len(q)) * 0.15  # 0.80-0.95
                        
                        # Word-level matching (for partial queries like "ALOE VERA-PASSIONFRUIT D")
                        q_words = set(q.replace('-', ' ').replace('_', ' ').split())
                        c_words = set(c.replace('-', ' ').replace('_', ' ').split())
                        
                        if q_words and c_words:
                            common_words = q_words & c_words
                            if common_words:
                                # Calculate word overlap ratio
                                word_similarity = len(common_words) / max(len(q_words), len(c_words))
                                
                                # Also check character-level similarity for the common parts
                                q_common = ' '.join(sorted(common_words))
                                c_common = ' '.join(w for w in c.split() if w in common_words or any(cw in w for cw in common_words))
                                char_similarity = len(q_common) / max(len(q), len(c)) if max(len(q), len(c)) > 0 else 0
                                
                                return 0.5 + word_similarity * 0.3 + char_similarity * 0.2
                        
                        return 0.0
                    
                    for candidate in candidates:
                        cand_id = candidate.get("id", "")
                        cand_name = candidate.get("recipe_name", "")
                        cand_desc = candidate.get("description", "")
                        
                        # Extract MaterialMasterShorttext from description
                        mst_part = ""
                        if "MaterialMasterShorttext:" in cand_desc:
                            try:
                                mst_part = cand_desc.split("MaterialMasterShorttext:")[1].split(",")[0].strip()
                            except:
                                mst_part = ""
                        
                        # Calculate similarity with both recipe_name and MaterialMasterShorttext
                        sim_name = calculate_name_similarity(query_clean, cand_name)
                        sim_mst = calculate_name_similarity(query_clean, mst_part)
                        best_sim = max(sim_name, sim_mst)
                        
                        # For short queries, calculate boost for ALL candidates with any similarity
                        # This ensures the most similar recipes rank highest even without exact matches
                        if best_sim >= 0.95:
                            # Near-exact match
                            boost = RECIPE_NAME_EXACT_BOOST
                            match_type = "exact"
                            recipe_name_matches[cand_id] = (match_type, boost, best_sim)
                            logger.info(f"  ★★★ EXACT NAME MATCH (sim={best_sim:.2f}): {mst_part or cand_name[:60]} → boost +{boost}")
                        elif best_sim >= 0.7:
                            # Strong partial match
                            boost = RECIPE_NAME_PARTIAL_BOOST * (best_sim / 0.95)  # Scale by similarity
                            match_type = "partial"
                            recipe_name_matches[cand_id] = (match_type, boost, best_sim)
                            logger.info(f"  ★★ STRONG NAME MATCH (sim={best_sim:.2f}): {mst_part or cand_name[:60]} → boost +{boost:.2f}")
                        elif best_sim >= 0.5:
                            # Weak partial match
                            boost = RECIPE_NAME_PARTIAL_BOOST * 0.5 * (best_sim / 0.7)
                            match_type = "weak"
                            recipe_name_matches[cand_id] = (match_type, boost, best_sim)
                            logger.debug(f"  ★ WEAK NAME MATCH (sim={best_sim:.2f}): {mst_part or cand_name[:60]} → boost +{boost:.2f}")
                        elif best_sim >= 0.3:
                            # Very weak match - still useful for ranking similar alternatives
                            boost = RECIPE_NAME_PARTIAL_BOOST * 0.25 * best_sim
                            match_type = "similar"
                            recipe_name_matches[cand_id] = (match_type, boost, best_sim)
                            # Don't log these to avoid noise, but they help rank similar alternatives
                    
                    if recipe_name_matches:
                        # Sort and show top matches
                        top_matches = sorted(recipe_name_matches.items(), key=lambda x: x[1][2], reverse=True)[:5]
                        logger.info(f"Found {len(recipe_name_matches)} recipe name matches for short query. Top 5:")
                        for cid, (mtype, boost, sim) in top_matches:
                            cand = next((c for c in candidates if c.get("id") == cid), None)
                            if cand:
                                logger.info(f"    {mtype}: {cand.get('recipe_name', 'Unknown')[:50]} (sim={sim:.2f}, boost=+{boost:.2f})")
                                
                elif query_len < 50 and query_len >= 3 and (has_specific_features or has_numerical_filters):
                    # Short query but has specific features/constraints - use feature-based search
                    reason = []
                    if has_specific_features:
                        reason.append(f"specific features: {specific_features[:3]}")
                    if has_numerical_filters:
                        reason.append(f"numerical filters: {list(numerical_filters.keys())}")
                    logger.info(
                        f"Short query ({query_len} chars) but has {', '.join(reason)} "
                        f"- using normal feature-based search"
                    )
                elif name_similarity_query and not has_specific_features and not has_numerical_filters:
                    # Long brief but a concise name-like line is available; apply name similarity boosts
                    logger.info(
                        f"Name-like line detected for similarity matching: '{name_similarity_query}'"
                    )
                    name_boost_scale = 0.6

                    def calculate_name_similarity(query: str, candidate_name: str) -> float:
                        if not candidate_name:
                            return 0.0
                        q = query.lower()
                        c = candidate_name.lower()
                        if q == c:
                            return 1.0
                        if q in c:
                            return 0.85 + (len(q) / len(c)) * 0.15
                        if c in q:
                            return 0.80 + (len(c) / len(q)) * 0.15
                        q_words = set(q.replace('-', ' ').replace('_', ' ').split())
                        c_words = set(c.replace('-', ' ').replace('_', ' ').split())
                        if q_words and c_words:
                            common_words = q_words & c_words
                            if common_words:
                                word_similarity = len(common_words) / max(len(q_words), len(c_words))
                                q_common = ' '.join(sorted(common_words))
                                c_common = ' '.join(
                                    w for w in c.split() if w in common_words or any(cw in w for cw in common_words)
                                )
                                char_similarity = len(q_common) / max(len(q), len(c)) if max(len(q), len(c)) > 0 else 0
                                return 0.5 + word_similarity * 0.3 + char_similarity * 0.2
                        return 0.0

                    for candidate in candidates:
                        cand_id = candidate.get("id", "")
                        cand_name = candidate.get("recipe_name", "")
                        cand_desc = candidate.get("description", "")

                        mst_part = ""
                        if "MaterialMasterShorttext:" in cand_desc:
                            try:
                                mst_part = cand_desc.split("MaterialMasterShorttext:")[1].split(",")[0].strip()
                            except Exception:
                                mst_part = ""

                        sim_name = calculate_name_similarity(name_similarity_query, cand_name)
                        sim_mst = calculate_name_similarity(name_similarity_query, mst_part)
                        best_sim = max(sim_name, sim_mst)

                        if best_sim >= 0.95:
                            boost = RECIPE_NAME_EXACT_BOOST * name_boost_scale
                            match_type = "exact"
                        elif best_sim >= 0.7:
                            boost = RECIPE_NAME_PARTIAL_BOOST * (best_sim / 0.95) * name_boost_scale
                            match_type = "partial"
                        elif best_sim >= 0.5:
                            boost = RECIPE_NAME_PARTIAL_BOOST * 0.5 * (best_sim / 0.7) * name_boost_scale
                            match_type = "weak"
                        else:
                            continue

                        recipe_name_matches[cand_id] = (match_type, boost, best_sim)

                    if recipe_name_matches:
                        top_matches = sorted(recipe_name_matches.items(), key=lambda x: x[1][2], reverse=True)[:5]
                        logger.info(f"Found {len(recipe_name_matches)} recipe name matches for name-like line. Top 5:")
                        for cid, (mtype, boost, sim) in top_matches:
                            cand = next((c for c in candidates if c.get("id") == cid), None)
                            if cand:
                                logger.info(
                                    f"    {mtype}: {cand.get('recipe_name', 'Unknown')[:50]} "
                                    f"(sim={sim:.2f}, boost=+{boost:.2f})"
                                )

            # Extract query flavor value for boosting
            query_flavor = None
            for qf, qv in zip(query_features, query_values):
                if any(flav in qf.lower() for flav in FLAVOR_FEATURES):
                    query_flavor = str(qv).lower().strip()
                    break

            # Log extracted query flavors for debugging
            if query_flavor:
                logger.info(
                    f"Flavor boost enabled. Query flavors: {query_flavor[:100]}...")

            # Determine scoring weights based on number of features
            # SPECIAL CASE: For short queries (recipe name searches), use name similarity as primary factor
            num_features = len(query_features)
            if is_short_query_mode and recipe_name_matches:
                # Short query with name matches: NAME SIMILARITY IS THE ONLY FACTOR
                # Text and feature scores are ignored - ranking is purely by name match quality
                # This ensures "FP ALOE VERA-PASSIONFRUIT DY" finds the closest matching recipe name
                text_weight = 0.0  # Ignore text similarity for name searches
                feature_search_weight = 0.0  # Ignore feature similarity
                feature_refinement_weight = 0.0  # Ignore feature matching
                weight_scheme = "name-only"
                logger.info(f"SHORT QUERY MODE: Name similarity is the ONLY ranking factor (text/feature weights = 0)")
            elif num_features == 0:
                text_weight = 1.0
                feature_search_weight = 0.0
                feature_refinement_weight = 0.0
                weight_scheme = "text-only"
            elif num_features <= 3:
                text_weight = 0.60
                feature_search_weight = 0.25
                feature_refinement_weight = 0.15
                weight_scheme = "text-heavy"
            elif num_features <= 8:
                text_weight = 0.30
                feature_search_weight = 0.50
                feature_refinement_weight = 0.20
                weight_scheme = "balanced"
            else:
                text_weight = 0.35
                feature_search_weight = 0.45
                feature_refinement_weight = 0.20
                weight_scheme = "feature-heavy"

            logger.info(
                f"Scoring weights ({weight_scheme}, {num_features} features): "
                f"Text={text_weight:.0%}, FeatureSearch={feature_search_weight:.0%}, "
                f"FeatureMatch={feature_refinement_weight:.0%}"
            )

            # Calculate feature similarity for each candidate
            for candidate in candidates:
                candidate_features = candidate.get("features", [])
                candidate_values = candidate.get("values", [])

                # Count matching features (standard approach)
                matching_count = 0
                total_features = len(query_features)
                flavor_matched = False

                for query_feat, query_val in zip(query_features, query_values):
                    query_feat_lower = query_feat.lower()

                    # Check if this feature exists in candidate
                    for j, cand_feat in enumerate(candidate_features):
                        if cand_feat.lower() == query_feat_lower:
                            cand_val = str(
                                candidate_values[j]) if j < len(candidate_values) else ""

                            # Use enhanced matching logic
                            if self._match_feature_value(str(query_val), cand_val):
                                matching_count += 1
                            break

                # Check for flavor match in description or recipe name (semantic matching)
                # This catches cases where Flavour feature name doesn't match exactly
                # ENHANCED: Supports multiple comma-separated flavors AND multi-word flavors
                if query_flavor:
                    recipe_desc = candidate.get("description", "").lower()
                    recipe_name = candidate.get("recipe_name", "").lower()

                    # Split query_flavor into individual flavor KEYWORDS for matching
                    # Step 1: Split by comma: "Gyros, Honey BBQ" → ["Gyros", "Honey BBQ"]
                    # Step 2: Split each by space: ["Gyros", "Honey", "BBQ"]
                    # This handles "Matcha tea" → ["matcha", "tea"] so "matcha" can match "MATCHA"
                    query_flavor_keywords = set()
                    for flavor_phrase in query_flavor.split(','):
                        flavor_phrase = flavor_phrase.strip().lower()
                        if flavor_phrase:
                            # Add the full phrase
                            query_flavor_keywords.add(flavor_phrase)
                            # Also add individual words (≥3 chars to avoid noise)
                            for word in flavor_phrase.split():
                                if len(word) >= 3:
                                    query_flavor_keywords.add(word)

                    # Also check candidate's Flavour feature value
                    for j, cand_feat in enumerate(candidate_features):
                        if any(flav in cand_feat.lower() for flav in FLAVOR_FEATURES):
                            cand_val = str(candidate_values[j]).lower(
                            ) if j < len(candidate_values) else ""
                            # Match if ANY query flavor keyword matches the candidate flavor
                            for qf in query_flavor_keywords:
                                if len(qf) >= 3 and (qf in cand_val or cand_val in qf):
                                    flavor_matched = True
                                    break
                            if flavor_matched:
                                break

                    # Also check description and recipe name for flavor keywords
                    if not flavor_matched:
                        matched_keyword = None
                        for qf in query_flavor_keywords:
                            # Skip very short flavor names to avoid false matches
                            if len(qf) >= 3 and (qf in recipe_desc or qf in recipe_name):
                                flavor_matched = True
                                matched_keyword = qf
                                break

                        # Log flavor match found in description/name (for debugging)
                        if flavor_matched:
                            logger.debug(
                                f"  ★ Flavor match: '{matched_keyword}' found in {candidate.get('recipe_name', 'Unknown')[:50]}"
                            )

                # Calculate feature refinement score
                feature_refinement_score = matching_count / \
                    total_features if total_features > 0 else 0

                # Store debug info
                candidate["_flavor_matched"] = flavor_matched

                # Get scores for combined calculation
                text_score = candidate.get("text_score", 0.0)
                feature_search_score = candidate.get(
                    "feature_search_score", 0.0)

                # Use weights determined at the start of the method
                combined_score = (
                    text_weight * text_score +
                    feature_search_weight * feature_search_score +
                    feature_refinement_weight * feature_refinement_score
                )

                # Apply soft filter penalty (numerical deviations + unknown categorical fields)
                filter_penalty = candidate.get("_numerical_penalty", 0.0)
                filter_penalty += candidate.get("_categorical_penalty", 0.0)
                unknown_count = len(candidate.get("_categorical_unknowns", []))
                if unknown_count:
                    filter_penalty += 0.10 * unknown_count

                if filter_penalty > 0:
                    FILTER_PENALTY_WEIGHT = 0.06
                    filter_penalty = min(filter_penalty, 1.5)
                    combined_score = max(
                        0.0, combined_score - filter_penalty * FILTER_PENALTY_WEIGHT)

                # Apply flavor bonus - this is the key differentiator!
                # Recipes matching the query flavor get a significant boost
                if flavor_matched:
                    combined_score += FLAVOR_BONUS

                    # EXTRA BOOST: Text-only candidates (found via text search but not feature search)
                    # get additional bonus to compensate for missing feature_search_score
                    # This ensures flavor-matched recipes found via text search can compete
                    # BUT: Scale the boost by feature_refinement_score to prioritize recipes
                    # that also match other specified features (not JUST the flavor)
                    is_text_only = feature_search_score == 0.0 and text_score > 0.0
                    if is_text_only:
                        # Scale TEXT_ONLY_FLAVOR_BOOST by feature match score
                        # - If recipe matches other features (feature_refinement_score > 0), get full boost
                        # - If recipe matches NO features (feature_refinement_score = 0), get reduced boost
                        # This prevents recipes that ONLY match flavor from outranking recipes
                        # that match flavor + other criteria (like "Aromafrei", "Quark/Topfen", etc.)
                        feature_match_multiplier = min(1.0, 0.3 + feature_refinement_score * 7.0)
                        actual_boost = TEXT_ONLY_FLAVOR_BOOST * feature_match_multiplier
                        combined_score += actual_boost
                        candidate["_text_only_boost"] = True
                        candidate["_actual_text_boost"] = actual_boost
                        if feature_refinement_score > 0:
                            logger.info(
                                f"  ★★ TEXT-ONLY FLAVOR BOOST: {candidate.get('recipe_name', 'Unknown')[:50]} "
                                f"+{actual_boost:.2f} (full boost, matches {feature_refinement_score:.2%} features)"
                            )
                        else:
                            logger.info(
                                f"  ★ REDUCED FLAVOR BOOST: {candidate.get('recipe_name', 'Unknown')[:50]} "
                                f"+{actual_boost:.2f} (reduced - matches flavor only, no other features)"
                            )

                # Apply recipe name match boost (if query looks like a recipe name)
                cand_id = candidate.get("id", "")
                if cand_id in recipe_name_matches:
                    match_info = recipe_name_matches[cand_id]
                    # Handle both old (match_type, boost) and new (match_type, boost, similarity) formats
                    if len(match_info) == 3:
                        match_type, name_boost, name_similarity = match_info
                    else:
                        match_type, name_boost = match_info
                        name_similarity = 0.0
                    combined_score += name_boost
                    candidate["_name_match"] = True
                    candidate["_name_match_type"] = match_type
                    candidate["_name_boost"] = name_boost
                    candidate["_name_similarity"] = name_similarity

                candidate["feature_score"] = feature_refinement_score
                candidate["feature_search_score"] = feature_search_score
                candidate["combined_score"] = combined_score

            # Sort by combined score
            refined_results = sorted(
                candidates, key=lambda x: x["combined_score"], reverse=True)

            logger.info(
                f"Feature refinement completed, returning top {min(top_k, len(refined_results))} results")

            # Log refined results with all scores for debugging
            logger.info("Refined results (sorted by combined score):")
            for i, result in enumerate(refined_results[:top_k], 1):
                recipe_name = result.get('recipe_name', 'Unknown')
                text_score = result.get('text_score', 0.0)
                feature_search_score = result.get('feature_search_score', 0.0)
                feature_score = result.get('feature_score', 0.0)
                combined_score = result.get('combined_score', 0.0)
                flavor_matched = result.get('_flavor_matched', False)
                text_only_boost = result.get('_text_only_boost', False)

                # Show flavor boost info with correct amounts
                boost_info_parts = []
                
                # Name match boost
                name_match = result.get('_name_match', False)
                if name_match:
                    name_boost = result.get('_name_boost', 0)
                    name_sim = result.get('_name_similarity', 0)
                    match_type = result.get('_name_match_type', 'partial')
                    if match_type == "exact":
                        boost_info_parts.append(f"★★★NAME(sim={name_sim:.2f})+{name_boost:.2f}")
                    elif match_type in ("partial", "weak"):
                        boost_info_parts.append(f"★★NAME(sim={name_sim:.2f})+{name_boost:.2f}")
                    else:
                        boost_info_parts.append(f"★NAME(sim={name_sim:.2f})+{name_boost:.2f}")
                
                # Flavor boost
                if flavor_matched:
                    if text_only_boost:
                        actual_text_boost = result.get('_actual_text_boost', TEXT_ONLY_FLAVOR_BOOST)
                        total_boost = FLAVOR_BONUS + actual_text_boost
                        # Use ★★ for full boost, ★ for reduced boost
                        star_symbol = "★★" if actual_text_boost >= TEXT_ONLY_FLAVOR_BOOST * 0.8 else "★"
                        boost_info_parts.append(f"{star_symbol}FLAVOR+{total_boost:.2f}")
                    else:
                        boost_info_parts.append(f"★FLAVOR+{FLAVOR_BONUS}")
                
                boost_info = " " + " ".join(boost_info_parts) if boost_info_parts else ""

                logger.info(
                    f"  {i}. {recipe_name}{boost_info} | Combined: {combined_score:.4f} "
                    f"(Text: {text_score:.4f}×{text_weight:.2f} + FeatSearch: {feature_search_score:.4f}×{feature_search_weight:.2f} + FeatMatch: {feature_score:.4f}×{feature_refinement_weight:.2f})"
                )

            return refined_results[:top_k]

        except Exception as e:
            logger.error(f"Error in feature refinement: {e}")
            # Fallback to text results
            return candidates[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant collection"""
        try:
            collection_info = self.qdrant_client.get_collection(
                self.collection_name)

            return {
                "total_recipes": collection_info.points_count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "qdrant_host": self.qdrant_host,
                "qdrant_port": self.qdrant_port,
                "search_capability": "qdrant_vector_search",
                "storage": "persistent"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_recipes": 0,
                "collection_name": self.collection_name,
                "error": str(e)
            }

    def get_feature_analysis(self) -> Dict[str, Any]:
        """Get feature analysis (compatibility method)"""
        # This is for compatibility with the old interface
        # We don't have the same detailed feature analysis in Qdrant
        return {
            "total_features": "N/A (Qdrant storage)",
            "feature_types": {},
            "binary_features": {},
            "numerical_features": [],
            "categorical_features": [],
            "binary_feature_names": []
        }

    @property
    def recipes(self) -> List[Dict[str, Any]]:
        """
        Property for compatibility with old interface
        Returns a list representation of recipes (limited to first 100)
        """
        try:
            # Get a sample of recipes from Qdrant
            search_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False,
                timeout=30  # Timeout for sample retrieval
            )[0]

            recipes_list = []
            for result in search_results:
                payload = result.payload if result.payload is not None else {}
                recipe_data = {
                    "id": result.id,
                    "payload": payload
                }
                recipes_list.append(recipe_data)

            return recipes_list
        except Exception as e:
            logger.error(f"Error getting recipes: {e}")
            return []
