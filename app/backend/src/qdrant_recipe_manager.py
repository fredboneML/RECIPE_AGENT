#!/usr/bin/env python3
"""
Qdrant Recipe Manager

This manager actually uses Qdrant for persistent recipe storage and search,
replacing the in-memory approach of EnhancedTwoStepRecipeManager.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantRecipeManager:
    """
    Recipe manager that uses Qdrant for storage and search.
    Compatible with the existing agent interface but uses persistent storage.
    """

    def __init__(self,
                 collection_name: str = "food_recipes_two_step",
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 qdrant_host: str = "qdrant",
                 qdrant_port: int = 6333):
        """
        Initialize the Qdrant Recipe Manager

        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Sentence transformer model name
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
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

        # Check if collection exists
        self._validate_collection()

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

    def search_by_text_description(self,
                                   text_description: str,
                                   top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search recipes by text description using Qdrant vector search

        Args:
            text_description: Text description for search
            top_k: Number of results to return

        Returns:
            List of matching recipes with scores
        """
        try:
            # Create embedding for query using SentenceTransformer
            query_vector = self.embedding_model.encode(text_description)

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
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
                    "recipe_name": payload.get("recipe_name", ""),
                    "description": payload.get("description", ""),
                    "features": payload.get("features", []),
                    "values": payload.get("values", []),
                    "num_features": payload.get("num_features", 0),
                    "metadata": {
                        "recipe_name": payload.get("recipe_name", "")
                    }
                }
                results.append(recipe_data)

            logger.info(
                f"Found {len(results)} recipes for query: '{text_description[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"Error in Qdrant search: {e}")
            return []

    def search_two_step(self,
                        text_description: str,
                        query_df: Optional[pd.DataFrame] = None,
                        text_top_k: int = 50,
                        final_top_k: int = 10) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Two-step search: text description + optional feature refinement

        Args:
            text_description: Recipe description for search
            query_df: Optional DataFrame with features for refinement
            text_top_k: Number of candidates from text search
            final_top_k: Final number of results

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

        # Step 1: Text-based search using Qdrant
        logger.info(f"Step 1: Searching by text in Qdrant")
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

            # Refine candidates based on feature matching
            final_results = self._refine_by_features(
                text_candidates, query_features, query_values, final_top_k
            )
            search_metadata["refinement_completed"] = True

        else:
            logger.info("Step 2: Skipped (no feature data provided)")
            final_results = text_candidates[:final_top_k]
            search_metadata["refinement_completed"] = False

        search_metadata["final_results_count"] = len(final_results)

        return final_results, search_metadata

    def _refine_by_features(self,
                            candidates: List[Dict[str, Any]],
                            query_features: List[str],
                            query_values: List[Any],
                            top_k: int) -> List[Dict[str, Any]]:
        """
        Refine candidates using feature-based similarity

        Args:
            candidates: Candidate recipes from text search
            query_features: Feature names to match
            query_values: Feature values to match
            top_k: Number of results to return

        Returns:
            Refined and reranked results
        """
        try:
            # Create query feature set for matching
            query_feature_dict = {
                feat.lower(): str(val).lower()
                for feat, val in zip(query_features, query_values)
            }

            # Calculate feature similarity for each candidate
            for candidate in candidates:
                candidate_features = candidate.get("features", [])
                candidate_values = candidate.get("values", [])

                # Count matching features
                matching_count = 0
                total_features = len(query_features)

                for i, (query_feat, query_val) in enumerate(zip(query_features, query_values)):
                    query_feat_lower = query_feat.lower()
                    query_val_lower = str(query_val).lower()

                    # Check if this feature exists in candidate
                    for j, cand_feat in enumerate(candidate_features):
                        if cand_feat.lower() == query_feat_lower:
                            cand_val = str(
                                candidate_values[j]) if j < len(candidate_values) else ""
                            # Fuzzy match for values
                            if query_val_lower in cand_val.lower() or cand_val.lower() in query_val_lower:
                                matching_count += 1
                            break

                # Calculate feature similarity score
                feature_score = matching_count / \
                    total_features if total_features > 0 else 0

                # Combined score (weighted average)
                text_weight = 0.3
                feature_weight = 0.7
                combined_score = (text_weight * candidate["text_score"] +
                                  feature_weight * feature_score)

                candidate["feature_score"] = feature_score
                candidate["combined_score"] = combined_score

            # Sort by combined score
            refined_results = sorted(
                candidates, key=lambda x: x["combined_score"], reverse=True)

            logger.info(
                f"Feature refinement completed, returning top {min(top_k, len(refined_results))} results")
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
