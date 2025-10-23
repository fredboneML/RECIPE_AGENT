import uuid
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pickle
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Recipe:
    """Enhanced data class representing a recipe with features, values, and description."""
    id: str
    features: List[str]
    values: List[Any]
    description: str  # New field for text description
    metadata: Optional[Dict[str, Any]] = None


class EnhancedTwoStepRecipeManager:
    """
    Enhanced recipe manager with 2-step search capability:
    Step 1: Text-based search using description
    Step 2: Feature-based refinement (optional)
    """

    def __init__(self,
                 collection_name: str = "recipes",
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 max_features: int = 200):
        """Initialize the enhanced two-step manager."""
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model)
        self.max_features = max_features

        # Feature type detection and encoding (existing)
        self.categorical_encoders = {}
        self.numerical_scaler = StandardScaler()
        self.all_feature_names = set()
        self.feature_types = {}
        self.binary_features = {}
        self.feature_value_stats = defaultdict(lambda: defaultdict(int))

        # Storage - now with multiple vectors per recipe
        self.recipes = []
        self.recipe_index = {}

        # Vector dimensions
        self.text_embedding_dim = 384  # For text descriptions
        self.embedding_dim = 384       # For feature text
        self.categorical_dim = 100     # For categorical features
        self.feature_vector_dim = self.embedding_dim + self.categorical_dim

        # Binary patterns (existing)
        self.binary_patterns = {
            'no_prefix': (r'^no\s+(.+)$', r'^(.+)$'),
            'not_prefix': (r'^not\s+(.+)$', r'^(.+)$'),
            'without_with': (r'^without\s+(.+)$', r'^with\s+(.+)$'),
            'non_prefix': (r'^non[-\s](.+)$', r'^(.+)$'),
            'inactive_active': (r'^inactive$', r'^active$'),
        }

        self.negative_indicators = [
            'no ', 'not ', 'without', 'non-', 'inactive', 'absent', 'missing', 'none'
        ]
        self.positive_indicators = [
            'yes', 'active', 'present', 'true', 'with '
        ]

        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize the collection."""
        logger.info(
            f"Initialized enhanced two-step collection: {self.collection_name}")

    def _create_text_embedding(self, description: str) -> np.ndarray:
        """Create text embedding for recipe description."""
        if not description or pd.isna(description):
            # Return zero vector for missing descriptions
            return np.zeros(self.text_embedding_dim)

        # Clean and preprocess description
        clean_description = str(description).strip()
        text_embedding = self.embedding_model.encode(clean_description)

        return text_embedding

    def _create_feature_vector(self, features: List[str], values: List[Any], fit: bool = False) -> np.ndarray:
        """Create feature-based vector (existing functionality)."""
        # Text embedding from features
        feature_text = self._create_feature_text(features, values)
        feature_text_embedding = self.embedding_model.encode(feature_text)

        # Categorical/numerical encoding
        feature_categorical_vector = self._encode_mixed_features(
            features, values, fit=fit)

        # Combine
        feature_vector = np.concatenate(
            [feature_text_embedding, feature_categorical_vector])

        return feature_vector

    def update_recipes(self,
                       features_list: List[List[str]],
                       values_list: List[List[Any]],
                       descriptions_list: List[str],  # New parameter
                       recipe_ids: Optional[List[str]] = None,
                       metadata_list: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Update recipe store with text descriptions and features."""
        try:
            if len(features_list) != len(values_list) or len(features_list) != len(descriptions_list):
                raise ValueError(
                    "Features, values, and descriptions lists must have the same length")

            # Pre-analyze features (existing)
            logger.info("Analyzing feature patterns...")
            self._analyze_feature_values(features_list, values_list)

            # Clear existing recipes
            self.recipes = []
            self.recipe_index = {}

            logger.info(
                f"Processing {len(features_list)} recipes with descriptions...")

            for i, (features, values, description) in enumerate(zip(features_list, values_list, descriptions_list)):
                min_length = min(len(features), len(values))
                features = features[:min_length]
                values = values[:min_length]

                recipe_id = recipe_ids[i] if recipe_ids else str(uuid.uuid4())

                # Create multiple vectors
                text_vector = self._create_text_embedding(description)
                feature_vector = self._create_feature_vector(
                    features, values, fit=True)

                # Create payload
                payload = {
                    "features": features,
                    "values": [self._clean_value(v) for v in values],
                    "description": description or "",
                    "recipe_id": recipe_id,
                    "feature_text": self._create_feature_text(features, values)
                }

                if metadata_list and i < len(metadata_list):
                    payload.update(metadata_list[i])

                recipe_data = {
                    "id": recipe_id,
                    "text_vector": text_vector,           # For Step 1: text search
                    "feature_vector": feature_vector,     # For Step 2: feature refinement
                    "payload": payload
                }

                self.recipes.append(recipe_data)
                self.recipe_index[recipe_id] = len(self.recipes) - 1

                if (i + 1) % 50 == 0:
                    logger.info(
                        f"Processed {i + 1}/{len(features_list)} recipes")

            logger.info(
                f"Successfully updated {len(features_list)} recipes with two-step search capability")
            return True

        except Exception as e:
            logger.error(f"Error updating recipes: {e}")
            return False

    def search_by_text_description(self,
                                   text_description: str,
                                   top_k: int = 20) -> List[Dict[str, Any]]:
        """Step 1: Search recipes by text description similarity."""
        try:
            if len(self.recipes) == 0:
                logger.error("No recipes stored. Please update recipes first.")
                return []

            # Create text embedding for query
            query_text_vector = self._create_text_embedding(text_description)

            # Get all text vectors
            stored_text_vectors = np.array(
                [recipe["text_vector"] for recipe in self.recipes])

            # Calculate similarities
            text_similarities = cosine_similarity(
                [query_text_vector], stored_text_vectors)[0]

            # Get top results
            top_indices = np.argsort(text_similarities)[::-1][:top_k]

            text_results = []
            for idx in top_indices:
                recipe = self.recipes[idx]
                recipe_data = {
                    "id": recipe["id"],
                    "text_score": float(text_similarities[idx]),
                    "features": recipe["payload"].get("features", []),
                    "values": recipe["payload"].get("values", []),
                    "description": recipe["payload"].get("description", ""),
                    "feature_text": recipe["payload"].get("feature_text", ""),
                    "metadata": {k: v for k, v in recipe["payload"].items()
                                 if k not in ["features", "values", "recipe_id", "feature_text", "description"]}
                }
                text_results.append(recipe_data)

            logger.info(
                f"Text search found {len(text_results)} results for: '{text_description}'")
            return text_results

        except Exception as e:
            logger.error(f"Error in text-based search: {e}")
            return []

    def refine_by_features(self,
                           candidate_recipes: List[Dict[str, Any]],
                           query_features: List[str],
                           query_values: List[Any],
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """Step 2: Refine candidates using feature-based similarity."""
        try:
            if not candidate_recipes:
                return []

            # Create feature vector for query
            query_feature_vector = self._create_feature_vector(
                query_features, query_values, fit=False)

            # Get feature vectors for candidate recipes
            candidate_indices = [self.recipe_index[recipe["id"]] for recipe in candidate_recipes
                                 if recipe["id"] in self.recipe_index]

            if not candidate_indices:
                return candidate_recipes[:top_k]  # Fallback to text results

            candidate_feature_vectors = np.array([self.recipes[idx]["feature_vector"]
                                                  for idx in candidate_indices])

            # Calculate feature similarities
            feature_similarities = cosine_similarity(
                [query_feature_vector], candidate_feature_vectors)[0]

            # Combine text and feature scores (weighted average)
            text_weight = 0.3
            feature_weight = 0.7

            refined_results = []
            for i, (candidate_idx, recipe) in enumerate(zip(candidate_indices, candidate_recipes)):
                combined_score = (text_weight * recipe["text_score"] +
                                  feature_weight * feature_similarities[i])

                refined_recipe = recipe.copy()
                refined_recipe["feature_score"] = float(
                    feature_similarities[i])
                refined_recipe["combined_score"] = float(combined_score)
                refined_results.append(refined_recipe)

            # Sort by combined score
            refined_results.sort(
                key=lambda x: x["combined_score"], reverse=True)

            logger.info(
                f"Feature refinement completed, returning top {min(top_k, len(refined_results))} results")
            return refined_results[:top_k]

        except Exception as e:
            logger.error(f"Error in feature-based refinement: {e}")
            return candidate_recipes[:top_k]  # Fallback to text results

    def search_two_step(self,
                        text_description: str,
                        query_df: Optional[pd.DataFrame] = None,
                        text_top_k: int = 50,  # First step candidates
                        final_top_k: int = 10) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Two-step search: text description + optional feature refinement.

        Args:
            text_description: Recipe description (mandatory)
            query_df: DataFrame with features and values (optional)
            text_top_k: Number of candidates from text search
            final_top_k: Final number of results to return

        Returns:
            Tuple of (results, metadata)
        """
        search_metadata = {
            "search_type": "two_step",
            "text_description": text_description,
            "has_feature_refinement": query_df is not None,
            "text_candidates": text_top_k,
            "final_results": final_top_k
        }

        # Step 1: Text-based search
        logger.info(
            f"Step 1: Searching by text description: '{text_description}'")
        text_candidates = self.search_by_text_description(
            text_description, text_top_k)

        if not text_candidates:
            logger.warning("No candidates found in text search")
            return [], search_metadata

        search_metadata["text_results_found"] = len(text_candidates)

        # Step 2: Feature refinement (if query_df provided)
        if query_df is not None and not query_df.empty:
            logger.info("Step 2: Refining with feature-based similarity")

            # Extract features and values from DataFrame
            if 'charactDescr' in query_df.columns and 'valueCharLong' in query_df.columns:
                query_features = query_df['charactDescr'].tolist()
                query_values = query_df['valueCharLong'].tolist()
            else:
                query_features = query_df.iloc[:, 0].tolist()
                query_values = query_df.iloc[:, 1].tolist()

            search_metadata["query_features_count"] = len(query_features)

            # Refine candidates
            final_results = self.refine_by_features(
                text_candidates, query_features, query_values, final_top_k
            )
            search_metadata["refinement_completed"] = True

        else:
            logger.info("Step 2: Skipped (no feature data provided)")
            final_results = text_candidates[:final_top_k]
            search_metadata["refinement_completed"] = False

        search_metadata["final_results_count"] = len(final_results)

        return final_results, search_metadata

    # Include all existing methods for binary feature handling
    def _analyze_feature_values(self, features_list: List[List[str]], values_list: List[List[Any]]):
        """Analyze all feature values to detect patterns and binary oppositions."""
        # [Keep existing implementation]
        for features, values in zip(features_list, values_list):
            for feature, value in zip(features, values):
                if pd.notna(value) and str(value).strip():
                    clean_value = str(value).lower().strip()
                    self.feature_value_stats[feature][clean_value] += 1

        for feature, value_counts in self.feature_value_stats.items():
            unique_values = list(value_counts.keys())
            if self._is_likely_binary_feature(feature, unique_values):
                self.feature_types[feature] = 'binary'
                self._map_binary_oppositions(feature, unique_values)

    def _map_binary_oppositions(self, feature_name: str, values: List[str]) -> None:
        """Map binary oppositions for a specific feature."""
        if feature_name not in self.binary_features:
            self.binary_features[feature_name] = {
                'positive': set(), 'negative': set(), 'mappings': {}}

        negative_values = []
        positive_values = []

        for value in values:
            if any(neg in value for neg in self.negative_indicators):
                negative_values.append(value)
                self.binary_features[feature_name]['negative'].add(value)
            else:
                positive_values.append(value)
                self.binary_features[feature_name]['positive'].add(value)

        for neg_val in negative_values:
            core_concept = self._extract_core_concept(neg_val)
            if core_concept:
                for pos_val in positive_values:
                    if core_concept in pos_val or pos_val in core_concept:
                        self.binary_features[feature_name]['mappings'][neg_val] = pos_val
                        self.binary_features[feature_name]['mappings'][pos_val] = neg_val
                        break

    def _extract_core_concept(self, value: str):
        """Extract core concept from value."""
        value_lower = value.lower().strip()

        for pattern_name, (neg_pattern, pos_pattern) in self.binary_patterns.items():
            match = re.match(neg_pattern, value_lower, re.IGNORECASE)
            if match:
                if pattern_name == 'inactive_active':
                    if 'inactive' in value_lower:
                        return 'active'
                    elif 'active' in value_lower:
                        return 'inactive'
                else:
                    try:
                        return match.group(1)
                    except IndexError:
                        return value_lower.replace('no ', '').replace('not ', '').replace('without ', '').replace('non-', '').replace('non ', '')

        if value_lower.startswith('no '):
            return value_lower[3:]

        return None

    def _detect_value_type(self, value: Any) -> Tuple[str, float]:
        """Enhanced value type detection with binary awareness."""
        if pd.isna(value) or value is None or str(value).strip() == '' or str(value).strip().lower() == 'none':
            return 'missing', 0.0

        value_str = str(value).strip()

        # Handle European percentage format
        percentage_pattern = r'^(\d+(?:,\d+)?)\s*%?$'
        if value_str.endswith('%') or re.match(percentage_pattern, value_str):
            try:
                num_str = value_str.replace('%', '').replace(',', '.').strip()
                num_val = float(num_str)
                return 'percentage', num_val
            except ValueError:
                return 'categorical', 0.0

        # Handle European decimal format
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

        try:
            num_val = float(value_str)
            return 'numerical', num_val
        except ValueError:
            pass

        return 'binary', 0.0

    def _encode_binary_feature(self, feature_name: str, value: str, fit: bool = False) -> float:
        """Enhanced binary feature encoding with proper opposition handling."""
        if not value or pd.isna(value):
            return 0.0

        value_clean = str(value).lower().strip()

        if fit:
            if feature_name not in self.binary_features:
                self.binary_features[feature_name] = {
                    'positive': set(), 'negative': set(), 'mappings': {}}

        if feature_name in self.binary_features:
            if value_clean in self.binary_features[feature_name]['negative']:
                return -1.0
            elif value_clean in self.binary_features[feature_name]['positive']:
                return 1.0

        for neg_indicator in self.negative_indicators:
            if neg_indicator in value_clean:
                if fit and feature_name in self.binary_features:
                    self.binary_features[feature_name]['negative'].add(
                        value_clean)
                return -1.0

        for pos_indicator in self.positive_indicators:
            if pos_indicator in value_clean:
                if fit and feature_name in self.binary_features:
                    self.binary_features[feature_name]['positive'].add(
                        value_clean)
                return 1.0

        if feature_name.lower() in value_clean or any(word in value_clean for word in feature_name.lower().split()):
            if fit and feature_name in self.binary_features:
                self.binary_features[feature_name]['positive'].add(value_clean)
            return 1.0

        return 0.0

    def _is_likely_binary_feature(self, feature_name: str, values: List[str]) -> bool:
        """Enhanced binary feature detection."""
        if len(values) < 2:
            return False

        unique_values = set(v for v in values if v and pd.notna(v))
        if len(unique_values) > 10:
            return False

        has_negative = any(
            any(neg in val for neg in self.negative_indicators)
            for val in unique_values
        )

        has_positive = any(
            any(pos in val for pos in self.positive_indicators) or
            (not any(neg in val for neg in self.negative_indicators)
             and len(val.split()) <= 3)
            for val in unique_values
        )

        binary_feature_names = [
            'allergen', 'preserve', 'artificial', 'natural', 'gmo', 'organic',
            'kosher', 'halal', 'color', 'flavor', 'sweetener', 'starch',
            'pectin', 'blend', 'aspartame', 'additive', 'chemical'
        ]

        name_suggests_binary = any(name in feature_name.lower()
                                   for name in binary_feature_names)

        simple_binary_patterns = [
            ('yes', 'no'), ('true', 'false'), ('active', 'inactive'),
            ('present', 'absent'), ('with', 'without')
        ]

        has_simple_binary = any(
            all(pattern_val in ' '.join(unique_values).lower()
                for pattern_val in pattern)
            for pattern in simple_binary_patterns
        )

        return (has_negative and has_positive) or name_suggests_binary or has_simple_binary

    def _encode_mixed_features(self, features: List[str], values: List[Any], fit: bool = False) -> np.ndarray:
        """Enhanced encoding with improved binary feature handling."""
        feature_dict = {}
        for feature, value in zip(features, values):
            feature_dict[feature] = value

        if fit:
            self.all_feature_names.update(features)

        categorical_vector = np.zeros(self.categorical_dim)
        sorted_features = sorted(list(self.all_feature_names))

        for i, feature_name in enumerate(sorted_features[:self.categorical_dim]):
            if feature_name in feature_dict:
                value = feature_dict[feature_name]
                feature_type = self.feature_types.get(
                    feature_name, 'categorical')

                if feature_type == 'binary':
                    binary_val = self._encode_binary_feature(
                        feature_name, value, fit=fit)
                    categorical_vector[i] = binary_val

                elif feature_type in ['numerical', 'percentage']:
                    _, num_val = self._detect_value_type(value)
                    categorical_vector[i] = min(
                        max(num_val / 100.0, -1.0), 1.0)

                else:
                    clean_value = self._clean_value(value)

                    if fit and feature_name not in self.categorical_encoders:
                        self.categorical_encoders[feature_name] = LabelEncoder(
                        )

                    if feature_name in self.categorical_encoders:
                        try:
                            if fit:
                                if not hasattr(self.categorical_encoders[feature_name], 'classes_'):
                                    encoded_val = self.categorical_encoders[feature_name].fit_transform([clean_value])[
                                        0]
                                else:
                                    known_classes = set(
                                        self.categorical_encoders[feature_name].classes_)
                                    if clean_value not in known_classes:
                                        new_classes = list(
                                            known_classes) + [clean_value]
                                        self.categorical_encoders[feature_name].classes_ = np.array(
                                            new_classes)
                                    encoded_val = self.categorical_encoders[feature_name].transform([clean_value])[
                                        0]
                            else:
                                if hasattr(self.categorical_encoders[feature_name], 'classes_'):
                                    if clean_value in self.categorical_encoders[feature_name].classes_:
                                        encoded_val = self.categorical_encoders[feature_name].transform([clean_value])[
                                            0]
                                    else:
                                        encoded_val = -1
                                else:
                                    encoded_val = -1

                            categorical_vector[i] = encoded_val / 20.0

                        except Exception as e:
                            logger.warning(
                                f"Error encoding {feature_name}={clean_value}: {e}")
                            categorical_vector[i] = 0.0

        return categorical_vector

    def _clean_value(self, value: Any) -> str:
        """Clean and normalize a value."""
        if pd.isna(value) or value is None or str(value).strip() == '' or str(value).strip().lower() == 'none':
            return "MISSING"
        return str(value).strip()

    def _create_feature_text(self, features: List[str], values: List[Any]) -> str:
        """Create text representation with enhanced binary-aware formatting."""
        feature_texts = []
        for feature, value in zip(features, values):
            clean_value = self._clean_value(value)
            if clean_value != "MISSING":
                if feature in self.feature_types and self.feature_types[feature] == 'binary':
                    binary_val = self._encode_binary_feature(
                        feature, clean_value, fit=False)
                    if binary_val > 0:
                        feature_texts.append(f"{feature}: positive")
                    elif binary_val < 0:
                        feature_texts.append(f"{feature}: negative")
                    else:
                        feature_texts.append(f"{feature}: neutral")
                else:
                    feature_texts.append(f"{feature}: {clean_value}")

        return " | ".join(feature_texts)

    def get_feature_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis including binary feature detection."""
        type_counts = {}
        for feature_type in self.feature_types.values():
            type_counts[feature_type] = type_counts.get(feature_type, 0) + 1

        binary_analysis = {}
        for feature, mappings in self.binary_features.items():
            binary_analysis[feature] = {
                'positive_values': list(mappings['positive']),
                'negative_values': list(mappings['negative']),
                'opposition_mappings': mappings.get('mappings', {}),
                'total_unique_values': len(mappings['positive']) + len(mappings['negative'])
            }

        return {
            "total_features": len(self.all_feature_names),
            "feature_types": type_counts,
            "binary_features": binary_analysis,
            "numerical_features": [f for f, t in self.feature_types.items() if t in ['numerical', 'percentage']],
            "categorical_features": [f for f, t in self.feature_types.items() if t == 'categorical'],
            "binary_feature_names": [f for f, t in self.feature_types.items() if t == 'binary']
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the stored recipes."""
        analysis = self.get_feature_analysis()
        return {
            "total_recipes": len(self.recipes),
            "total_features": len(self.all_feature_names),
            "text_vector_dimension": self.text_embedding_dim,
            "feature_vector_dimension": self.feature_vector_dim,
            "search_capability": "two_step_text_and_features",
            "collection_name": self.collection_name,
            "feature_analysis": analysis
        }

# Usage Functions


def run_two_step_search(manager: EnhancedTwoStepRecipeManager,
                        text_description: str,
                        query_df: Optional[pd.DataFrame] = None,
                        text_top_k: int = 50,
                        final_top_k: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the two-step search and return formatted results.

    Args:
        manager: The two-step recipe manager
        text_description: Recipe description for text search
        query_df: Optional DataFrame with features for refinement
        text_top_k: Number of candidates from text search
        final_top_k: Final number of results

    Returns:
        Tuple of (detailed_results_df, summary_df)
    """

    print(f"RUNNING TWO-STEP RECIPE SEARCH")
    print(f"{'='*60}")
    print(f"Text Description: '{text_description}'")
    print(f"Feature Refinement: {'Yes' if query_df is not None else 'No'}")
    print(f"Text Candidates: {text_top_k}")
    print(f"Final Results: {final_top_k}")
    print(f"{'='*60}")

    # Run two-step search
    results, metadata = manager.search_two_step(
        text_description=text_description,
        query_df=query_df,
        text_top_k=text_top_k,
        final_top_k=final_top_k
    )

    if not results:
        print("No results found!")
        return pd.DataFrame(), pd.DataFrame()

    # Display results
    print(f"SEARCH COMPLETED")
    print(f"   Text candidates found: {metadata.get('text_results_found', 0)}")
    print(
        f"   Feature refinement: {'Applied' if metadata.get('refinement_completed') else 'Skipped'}")
    print(f"   Final results: {len(results)}")

    # Create results DataFrames
    detailed_data = []
    summary_data = []

    for i, result in enumerate(results, 1):
        # Summary data
        recipe_name = result['metadata'].get('recipe_name', f'Recipe_{i}')
        text_score = result.get('text_score', 0)
        feature_score = result.get('feature_score', 0)
        combined_score = result.get('combined_score', text_score)

        summary_data.append({
            'Rank': i,
            'Recipe_Name': recipe_name,
            'Text_Score': round(text_score, 4),
            'Feature_Score': round(feature_score, 4) if feature_score else 'N/A',
            'Combined_Score': round(combined_score, 4),
            'Description': result.get('description', '')[:300] + '...' if len(result.get('description', '')) > 300 else result.get('description', ''),
            'Features_Count': len(result.get('features', []))
        })

        # Detailed data (features and values)
        features = result.get('features', [])
        values = result.get('values', [])

        for j, (feature, value) in enumerate(zip(features, values)):
            detailed_data.append({
                'Rank': i,
                'Recipe_Name': recipe_name,
                'Feature_Index': j + 1,
                'Feature': feature,
                'Value': value,
                'Text_Score': round(text_score, 4),
                'Combined_Score': round(combined_score, 4)
            })

    detailed_df = pd.DataFrame(detailed_data)
    summary_df = pd.DataFrame(summary_data)

    print(f"TOP RESULTS SUMMARY:")
    print("=" * 120)
    print(summary_df.to_string(index=False))

    # Save to Excel
    try:
        with pd.ExcelWriter('two_step_search_results.xlsx', engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            detailed_df.to_excel(
                writer, sheet_name='Detailed_Features', index=False)

            # Metadata sheet
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_excel(
                writer, sheet_name='Search_Metadata', index=False)

        print(f"Results saved to 'two_step_search_results.xlsx'")
    except Exception as e:
        print(f"Could not save to Excel: {e}")

    return detailed_df, summary_df
