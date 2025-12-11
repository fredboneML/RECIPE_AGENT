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
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

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

        # Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(
                host=qdrant_host, port=qdrant_port)
            logger.info(
                f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
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
            logger.info("Initializing feature encoder (EnhancedTwoStepRecipeManager)...")
            
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
            logger.warning("Falling back to basic encoding (may affect search quality)")
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
                
                logger.info(f"Injected {len(feature_config)} pre-analyzed feature types")
                logger.info(f"  Binary: {sum(1 for t in feature_config.values() if t == 'binary')}")
                logger.info(f"  Numerical: {sum(1 for t in feature_config.values() if t in ['numerical', 'range'])}")
                
            except ImportError:
                logger.warning("FeatureAnalyzer not available, using basic feature type detection")
                
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
            logger.info("Analyzing feature map for binary opposition patterns...")
            
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
            self.feature_encoder._analyze_feature_values(features_list, values_list)
            
            logger.info(f"Binary opposition analysis complete:")
            logger.info(f"  Feature types detected: {len(self.feature_encoder.feature_types)}")
            logger.info(f"  Binary features with oppositions: {len(self.feature_encoder.binary_features)}")
            
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
                           country_filter: Optional[str] = None) -> List[Dict[str, Any]]:
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
            country_filter: Optional country name to filter results (None or "All" means no filter)

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
                logger.warning("Feature encoder not available, using basic encoding")
                feature_text = self._create_feature_text(
                    query_features, query_values)

                if not feature_text:
                    logger.warning("No valid features provided for feature search")
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

            # Build filter if country is specified
            query_filter = None
            if country_filter and country_filter != "All":
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="country",
                            match=MatchValue(value=country_filter)
                        )
                    ]
                )
                logger.info(
                    f"Applying country filter in feature search: {country_filter}")

            # Search using the "features" named vector
            try:
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=("features", feature_vector.tolist()),
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False
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
                    "search_source": "features"
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
                                   country_filter: Optional[str] = None,
                                   return_embedding: bool = False):
        """
        Search recipes by text description using Qdrant vector search

        Args:
            text_description: Text description for search
            top_k: Number of results to return
            country_filter: Optional country name to filter results (None or "All" means no filter)
            return_embedding: If True, also return the query embedding for reuse

        Returns:
            List of matching recipes with scores
            If return_embedding=True: Tuple of (results, query_embedding)
        """
        try:
            # Create embedding for query using SentenceTransformer
            query_vector = self.embedding_model.encode(text_description)

            # Build filter if country is specified
            query_filter = None
            if country_filter and country_filter != "All":
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="country",
                            match=MatchValue(value=country_filter)
                        )
                    ]
                )
                logger.info(f"Applying country filter: {country_filter}")

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
                    with_vectors=False
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
                    with_vectors=False
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
                    "search_source": "text"
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

    def search_two_step(self,
                        text_description: str,
                        query_df: Optional[pd.DataFrame] = None,
                        text_top_k: int = 50,
                        final_top_k: int = 10,
                        country_filter: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Two-step hybrid search: 
        - Step 1: Get candidates from both text search AND feature search (50/50 split)
        - Step 2: Feature-based refinement on merged candidates

        Args:
            text_description: Recipe description for search
            query_df: Optional DataFrame with features for refinement
            text_top_k: Total number of candidates (split between text and feature search)
            final_top_k: Final number of results
            country_filter: Optional country name to filter results (None or "All" means no filter)

        Returns:
            Tuple of (results, metadata)
        """
        search_metadata = {
            "search_type": "two_step_hybrid",
            "text_description": text_description,
            "has_feature_refinement": query_df is not None,
            "total_candidates_requested": text_top_k,
            "final_results": final_top_k,
            "country_filter": country_filter
        }

        # Calculate split for hybrid search (half text, half feature)
        half_k = text_top_k // 2
        text_search_k = half_k
        feature_search_k = text_top_k - half_k  # Handle odd numbers

        # Extract features for feature-based search
        query_features = []
        query_values = []
        if query_df is not None and not query_df.empty:
            if 'charactDescr' in query_df.columns and 'valueCharLong' in query_df.columns:
                query_features = query_df['charactDescr'].tolist()
                query_values = query_df['valueCharLong'].tolist()
            else:
                query_features = query_df.iloc[:, 0].tolist()
                query_values = query_df.iloc[:, 1].tolist()

        # Step 1a: Text-based search using Qdrant
        # OPTIMIZATION: Get query embedding to reuse later (avoids re-encoding)
        logger.info(
            f"Step 1a: Text-based search in Qdrant (top {text_search_k})")
        text_search_result = self.search_by_text_description(
            text_description, text_search_k, country_filter, return_embedding=True)
        
        # Handle return value (may be tuple if return_embedding=True)
        if isinstance(text_search_result, tuple):
            text_candidates, cached_query_embedding = text_search_result
        else:
            text_candidates = text_search_result
            cached_query_embedding = None

        search_metadata["text_results_found"] = len(text_candidates)

        # Log text search candidates
        logger.info(f"Text search found {len(text_candidates)} candidates:")
        for i, candidate in enumerate(text_candidates, 1):
            recipe_name = candidate.get('recipe_name', 'Unknown')
            text_score = candidate.get('text_score', 0.0)
            num_features = candidate.get('num_features', 0)
            description = candidate.get('description', '')
            description_preview = description[:80] + \
                '...' if len(description) > 80 else description
            logger.info(
                f"  {i}. {recipe_name} | Text Score: {text_score:.4f} | Features: {num_features} | Desc: {description_preview}")

        # Step 1b: Feature-based search using Qdrant (if features available)
        feature_candidates = []
        if query_features:
            logger.info(
                f"Step 1b: Feature-based search in Qdrant (top {feature_search_k})")
            feature_candidates = self.search_by_features(
                query_features, query_values, feature_search_k, country_filter)

            search_metadata["feature_search_results_found"] = len(
                feature_candidates)

            # Log feature search candidates
            logger.info(
                f"Feature search found {len(feature_candidates)} candidates:")
            for i, candidate in enumerate(feature_candidates, 1):
                recipe_name = candidate.get('recipe_name', 'Unknown')
                feature_search_score = candidate.get(
                    'feature_search_score', 0.0)
                num_features = candidate.get('num_features', 0)
                description = candidate.get('description', '')
                description_preview = description[:80] + \
                    '...' if len(description) > 80 else description
                logger.info(
                    f"  {i}. {recipe_name} | Feature Search Score: {feature_search_score:.4f} | Features: {num_features} | Desc: {description_preview}")
        else:
            logger.info(
                "Step 1b: Skipped feature search (no features provided)")
            # If no features, use full text search instead
            if len(text_candidates) < text_top_k:
                additional_text = self.search_by_text_description(
                    text_description, text_top_k)
                text_candidates = additional_text

        # Merge candidates from both searches
        if feature_candidates:
            all_candidates = self._merge_candidates(
                text_candidates, feature_candidates)
            logger.info(f"Merged candidates: {len(all_candidates)} unique recipes "
                        f"(from {len(text_candidates)} text + {len(feature_candidates)} feature, "
                        f"with deduplication)")

            # Calculate text scores for feature-only candidates
            # This ensures fair comparison in combined scoring
            # OPTIMIZATION: Reuse cached query embedding from text search
            all_candidates = self._calculate_text_scores_for_feature_only(
                all_candidates, text_description, cached_query_embedding)
        else:
            all_candidates = text_candidates

        search_metadata["merged_candidates"] = len(all_candidates)

        if not all_candidates:
            logger.warning("No candidates found in either search")
            return [], search_metadata

        # Count sources
        text_only = sum(1 for c in all_candidates if c.get(
            "search_source") == "text")
        feature_only = sum(1 for c in all_candidates if c.get(
            "search_source") == "features")
        both = sum(1 for c in all_candidates if c.get(
            "search_source") == "both")
        logger.info(
            f"Candidate sources: {text_only} text-only, {feature_only} feature-only, {both} from both")

        # Step 2: Feature refinement on merged candidates
        if query_features:
            logger.info(
                "Step 2: Refining merged candidates with feature-based similarity")
            search_metadata["query_features_count"] = len(query_features)

            # Refine all_candidates (merged from text + feature search) based on feature matching
            final_results = self._refine_by_features(
                all_candidates, query_features, query_values, final_top_k
            )
            search_metadata["refinement_completed"] = True

        else:
            logger.info("Step 2: Skipped (no feature data provided)")
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
                            top_k: int) -> List[Dict[str, Any]]:
        """
        Refine candidates using feature-based similarity with language-independent scoring.

        The combined score uses three components:
        - Text score (10%): Language-dependent, minimal influence
        - Feature search score (20%): Partially language-independent (includes categorical encoding)
        - Feature refinement score (70%): Fully language-independent (exact feature matching)

        Args:
            candidates: Candidate recipes from text and feature search
            query_features: Feature names to match
            query_values: Feature values to match
            top_k: Number of results to return

        Returns:
            Refined and reranked results
        """
        try:
            # Calculate feature similarity for each candidate
            for candidate in candidates:
                candidate_features = candidate.get("features", [])
                candidate_values = candidate.get("values", [])

                # Count matching features
                matching_count = 0
                total_features = len(query_features)

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

                # Calculate feature refinement score (language-independent)
                feature_refinement_score = matching_count / \
                    total_features if total_features > 0 else 0

                # Get scores for combined calculation
                text_score = candidate.get("text_score", 0.0)
                feature_search_score = candidate.get(
                    "feature_search_score", 0.0)

                # Combined score with language-independent weighting:
                # - Text (10%): Minimal influence, language-dependent
                # - Feature Search (50%): Vector similarity with categorical encoding (multilingual embeddings)
                # - Feature Refinement (40%): Exact feature matching (can be language-dependent on feature names)
                text_weight = 0.1
                feature_search_weight = 0.5
                feature_refinement_weight = 0.4

                combined_score = (
                    text_weight * text_score +
                    feature_search_weight * feature_search_score +
                    feature_refinement_weight * feature_refinement_score
                )

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
                logger.info(
                    f"  {i}. {recipe_name} | Combined: {combined_score:.4f} "
                    f"(Text: {text_score:.4f}×0.1 + FeatSearch: {feature_search_score:.4f}×0.5 + FeatMatch: {feature_score:.4f}×0.4)"
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
                with_vectors=False
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
