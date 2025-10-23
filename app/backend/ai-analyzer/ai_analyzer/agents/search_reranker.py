#!/usr/bin/env python3
"""
Search & Reranker Agent

This agent is responsible for:
1. Executing the appropriate search (text-only or two-step)
2. Finding the top K similar recipes
3. Reranking results based on relevance
"""
from src.qdrant_recipe_manager import QdrantRecipeManager
from src.two_step_recipe_search import EnhancedTwoStepRecipeManager
import logging
import sys
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd

# Add path for importing modules
sys.path.insert(0, '/usr/src/app/src')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchRerankerAgent:
    """Agent for executing recipe search and reranking results"""

    def __init__(self,
                 recipe_manager: Union[EnhancedTwoStepRecipeManager, QdrantRecipeManager],
                 default_top_k: int = 3):
        """
        Initialize the Search & Reranker Agent

        Args:
            recipe_manager: Instance of EnhancedTwoStepRecipeManager or QdrantRecipeManager
            default_top_k: Default number of top results to return
        """
        self.recipe_manager = recipe_manager
        self.default_top_k = default_top_k

        logger.info(
            f"Initialized SearchRerankerAgent with top_k={default_top_k}")

    def search_recipes(self,
                       search_type: str,
                       text_description: str,
                       features_df: Optional[pd.DataFrame] = None,
                       top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute recipe search based on search type

        Args:
            search_type: 'text_only' or 'two_step'
            text_description: Text description for search
            features_df: Optional DataFrame with features for two-step search
            top_k: Number of results to return (defaults to self.default_top_k)

        Returns:
            Tuple of (results, metadata)
        """
        top_k = top_k or self.default_top_k

        try:
            if search_type == 'text_only':
                return self._execute_text_only_search(text_description, top_k)
            elif search_type == 'two_step':
                return self._execute_two_step_search(text_description, features_df, top_k)
            else:
                logger.warning(
                    f"Unknown search type: {search_type}. Defaulting to text-only.")
                return self._execute_text_only_search(text_description, top_k)

        except Exception as e:
            logger.error(f"Error executing recipe search: {e}")
            return [], {"error": str(e), "search_type": search_type}

    def _execute_text_only_search(self,
                                  text_description: str,
                                  top_k: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute text-only search

        Args:
            text_description: Text description for search
            top_k: Number of results to return

        Returns:
            Tuple of (results, metadata)
        """
        logger.info(f"Executing TEXT-ONLY search with top_k={top_k}")
        logger.info(f"Search query: {text_description[:100]}...")

        try:
            # Use two-step search without feature refinement
            results, metadata = self.recipe_manager.search_two_step(
                text_description=text_description,
                query_df=None,  # No feature refinement
                text_top_k=top_k * 2,  # Get more candidates for better results
                final_top_k=top_k
            )

            metadata['search_strategy'] = 'text_only'

            logger.info(
                f"Text-only search completed. Found {len(results)} results.")
            return results, metadata

        except Exception as e:
            logger.error(f"Error in text-only search: {e}")
            raise

    def _execute_two_step_search(self,
                                 text_description: str,
                                 features_df: Optional[pd.DataFrame],
                                 top_k: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute two-step search with feature refinement

        Args:
            text_description: Text description for search
            features_df: DataFrame with features for refinement
            top_k: Number of results to return

        Returns:
            Tuple of (results, metadata)
        """
        logger.info(f"Executing TWO-STEP search with top_k={top_k}")
        logger.info(f"Search query: {text_description[:100]}...")

        if features_df is not None and not features_df.empty:
            logger.info(
                f"Feature refinement enabled with {len(features_df)} features")
        else:
            logger.warning(
                "No features provided for two-step search. Falling back to text-only.")
            return self._execute_text_only_search(text_description, top_k)

        try:
            # Use two-step search with feature refinement
            results, metadata = self.recipe_manager.search_two_step(
                text_description=text_description,
                query_df=features_df,
                text_top_k=top_k * 5,  # Get more candidates for feature refinement
                final_top_k=top_k
            )

            metadata['search_strategy'] = 'two_step'

            logger.info(
                f"Two-step search completed. Found {len(results)} results.")
            return results, metadata

        except Exception as e:
            logger.error(f"Error in two-step search: {e}")
            raise

    def rerank_results(self,
                       results: List[Dict[str, Any]],
                       reranking_criteria: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Rerank search results based on custom criteria

        Args:
            results: List of search results
            reranking_criteria: Optional dictionary with reranking weights
                                e.g., {'text_score': 0.4, 'feature_score': 0.6}

        Returns:
            Reranked list of results
        """
        if not results:
            return results

        if reranking_criteria is None:
            # Return results as-is if no reranking criteria
            return results

        try:
            logger.info(
                f"Reranking {len(results)} results with custom criteria")

            # Calculate new scores based on criteria
            for result in results:
                new_score = 0.0

                if 'text_score' in reranking_criteria and 'text_score' in result:
                    new_score += reranking_criteria['text_score'] * \
                        result['text_score']

                if 'feature_score' in reranking_criteria and 'feature_score' in result:
                    new_score += reranking_criteria['feature_score'] * \
                        result.get('feature_score', 0)

                result['reranked_score'] = new_score

            # Sort by new score
            reranked_results = sorted(results, key=lambda x: x.get(
                'reranked_score', 0), reverse=True)

            logger.info("Reranking completed")
            return reranked_results

        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results  # Return original results if reranking fails

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "agent_name": "SearchRerankerAgent",
            "default_top_k": self.default_top_k,
            "recipe_manager_stats": self.recipe_manager.get_stats()
        }
