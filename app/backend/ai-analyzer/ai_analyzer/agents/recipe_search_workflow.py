#!/usr/bin/env python3
"""
Recipe Search Workflow

Workflow for agentic recipe search using the RecipeSearchManager
This replaces the non-agentic approach with an intelligent multi-agent system
"""
from src.qdrant_recipe_manager import QdrantRecipeManager
from src.two_step_recipe_search import EnhancedTwoStepRecipeManager
from ai_analyzer.utils.model_logger import ModelLogger, get_model_config_from_env
from ai_analyzer.agents.recipe_search_manager import RecipeSearchManager
from ai_analyzer.agents.factory import AgentFactory
import logging
import sys
import os
from typing import Dict, Any, Optional
import pandas as pd

# Add path for importing modules
sys.path.insert(0, '/usr/src/app/src')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecipeSearchWorkflow:
    """Workflow for agentic recipe search"""

    def __init__(self,
                 collection_name: str = "food_recipes_two_step",
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 max_features: int = 200,
                 default_top_k: int = 3):
        """
        Initialize the Recipe Search Workflow

        Args:
            collection_name: Name of the recipe collection in Qdrant
            embedding_model: Embedding model to use for vector search
            max_features: Maximum number of features (not used with QdrantRecipeManager)
            default_top_k: Default number of top results to return
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.max_features = max_features
        self.default_top_k = default_top_k

        # Get model configuration from environment
        model_config = get_model_config_from_env()
        self.model_provider = model_config["provider"]

        # Log initial model configuration
        logger.info(f"Initializing RecipeSearchWorkflow")
        logger.info(f"Model provider: {self.model_provider}")

        # Handle special case for Groq
        if self.model_provider == "groq":
            self.model_name = model_config["groq_model_name"]
            self.use_openai_compatibility = model_config["groq_use_openai_compatibility"]

            logger.info(f"Using Groq model: {self.model_name}")
            logger.info(
                f"Groq OpenAI compatibility mode: {self.use_openai_compatibility}")

            # When using OpenAI compatibility mode, set provider to OpenAI with Groq base URL
            if self.use_openai_compatibility:
                self.model_provider = "openai"
                self.base_url = "https://api.groq.com/openai/v1"
                self.api_key = model_config.get("groq_api_key", "")
                logger.info("Configured Groq with OpenAI compatibility mode")
            else:
                self.api_key = model_config.get("groq_api_key", "")
                logger.info("Configured Groq in native mode")
        else:
            self.model_name = model_config.get("model_name", "gpt-4")
            self.api_key = model_config.get("api_key", "")
            logger.info(
                f"Using {self.model_provider} model: {self.model_name}")

        # Log initialization
        ModelLogger.log_model_usage(
            agent_name="RecipeSearchWorkflow",
            model_provider=self.model_provider,
            model_name=self.model_name,
            params={
                "collection_name": collection_name,
                "embedding_model": embedding_model
            }
        )

        # Initialize the recipe manager (use Qdrant for persistent storage)
        logger.info("Initializing QdrantRecipeManager with Qdrant storage...")

        # Get Qdrant configuration
        qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')
        qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))

        self.recipe_manager = QdrantRecipeManager(
            collection_name=collection_name,
            embedding_model=embedding_model,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port
        )

        # Initialize the recipe search manager (with all agents)
        logger.info("Initializing RecipeSearchManager with all agents...")
        self.search_manager = AgentFactory.create_recipe_search_manager(
            recipe_manager=self.recipe_manager,
            model_provider=self.model_provider,
            model_name=self.model_name,
            api_key=self.api_key,
            default_top_k=default_top_k
        )

        if self.search_manager is None:
            raise RuntimeError("Failed to initialize RecipeSearchManager")

        logger.info("RecipeSearchWorkflow initialized successfully")

    def load_recipes(self,
                     features_list,
                     values_list,
                     descriptions_list,
                     recipe_ids=None,
                     metadata_list=None) -> bool:
        """
        Load recipes into the recipe manager

        NOTE: When using QdrantRecipeManager (default), recipes are automatically
        loaded from the Qdrant collection. This method is only needed for
        the in-memory EnhancedTwoStepRecipeManager.

        Args:
            features_list: List of feature lists
            values_list: List of value lists
            descriptions_list: List of descriptions
            recipe_ids: Optional list of recipe IDs
            metadata_list: Optional list of metadata dictionaries

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if using Qdrant manager (which loads from Qdrant automatically)
            if isinstance(self.recipe_manager, QdrantRecipeManager):
                logger.info(
                    "Using QdrantRecipeManager - recipes are loaded from Qdrant collection automatically")
                logger.info(
                    "No need to manually load recipes. They are persistent in Qdrant.")

                # Get current stats from Qdrant
                stats = self.recipe_manager.get_stats()
                logger.info(
                    f"Current Qdrant collection has {stats['total_recipes']} recipes")
                return True

            # For in-memory manager, load recipes as before
            logger.info(f"Loading {len(features_list)} recipes into memory...")

            success = self.recipe_manager.update_recipes(
                features_list=features_list,
                values_list=values_list,
                descriptions_list=descriptions_list,
                recipe_ids=recipe_ids,
                metadata_list=metadata_list
            )

            if success:
                logger.info("Recipes loaded successfully")

                # Display analysis
                feature_analysis = self.recipe_manager.get_feature_analysis()
                logger.info(f"Feature Analysis:")
                logger.info(
                    f"  Total Features: {feature_analysis['total_features']}")
                logger.info(
                    f"  Binary Features: {len(feature_analysis['binary_feature_names'])}")
                logger.info(
                    f"  Numerical Features: {len(feature_analysis['numerical_features'])}")
                logger.info(
                    f"  Categorical Features: {len(feature_analysis['categorical_features'])}")

                return True
            else:
                logger.error("Failed to load recipes")
                return False

        except Exception as e:
            logger.error(f"Error loading recipes: {e}")
            return False

    def search(self,
               supplier_brief: str,
               top_k: Optional[int] = None,
               custom_reranking: Optional[Dict[str, float]] = None,
               save_results: bool = False,
               output_file: str = "recipe_search_results.xlsx") -> Dict[str, Any]:
        """
        Execute agentic recipe search

        Args:
            supplier_brief: The supplier project brief text
            top_k: Number of top results to return
            custom_reranking: Optional custom reranking criteria
            save_results: Whether to save results to file
            output_file: Output filename for results

        Returns:
            Dictionary with search results and explanation
        """
        try:
            # Validate setup
            validation = self.search_manager.validate_setup()
            if not validation["all_valid"]:
                logger.error("Recipe search manager setup is invalid")
                return {
                    "error": "Recipe search manager setup is invalid",
                    "validation": validation
                }

            # Execute search
            results = self.search_manager.search_similar_recipes(
                supplier_brief=supplier_brief,
                top_k=top_k or self.default_top_k,
                custom_reranking=custom_reranking
            )

            # Save results if requested
            if save_results and 'results_table' in results:
                self._save_results(results, output_file)

            return results

        except Exception as e:
            logger.error(f"Error in recipe search: {e}")
            logger.exception("Detailed error:")
            return {
                "error": str(e),
                "results_table": pd.DataFrame(),
                "explanation": f"An error occurred: {str(e)}"
            }

    def _save_results(self, results: Dict[str, Any], output_file: str):
        """Save search results to Excel file"""
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Save results table
                if 'results_table' in results and not results['results_table'].empty:
                    results['results_table'].to_excel(
                        writer, sheet_name='Search_Results', index=False)

                # Save explanation
                if 'explanation' in results:
                    explanation_df = pd.DataFrame([{
                        'Explanation': results['explanation']
                    }])
                    explanation_df.to_excel(
                        writer, sheet_name='Explanation', index=False)

                # Save metadata
                if 'metadata' in results:
                    metadata_df = pd.DataFrame([results['metadata']])
                    metadata_df.to_excel(
                        writer, sheet_name='Metadata', index=False)

                # Save intermediate outputs
                if 'intermediate_outputs' in results:
                    # Flatten intermediate outputs for Excel
                    intermediate_data = []
                    for stage, data in results['intermediate_outputs'].items():
                        for key, value in data.items():
                            intermediate_data.append({
                                'Stage': stage,
                                'Key': key,
                                'Value': str(value)
                            })
                    if intermediate_data:
                        intermediate_df = pd.DataFrame(intermediate_data)
                        intermediate_df.to_excel(
                            writer, sheet_name='Intermediate_Outputs', index=False)

            logger.info(f"Results saved to {output_file}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        return {
            "workflow": {
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "max_features": self.max_features,
                "default_top_k": self.default_top_k,
                "model_provider": self.model_provider,
                "model_name": self.model_name
            },
            "recipe_manager": self.recipe_manager.get_stats(),
            "search_manager": self.search_manager.get_agent_stats()
        }


# Convenience function for quick recipe searches
def search_recipes(supplier_brief: str,
                   recipe_manager: EnhancedTwoStepRecipeManager,
                   top_k: int = 3,
                   model_provider: str = "openai",
                   model_name: str = "gpt-4",
                   api_key: Optional[str] = None,
                   save_results: bool = False) -> Dict[str, Any]:
    """
    Convenience function for quick recipe searches

    Args:
        supplier_brief: The supplier project brief text
        recipe_manager: Instance of EnhancedTwoStepRecipeManager
        top_k: Number of top results
        model_provider: Model provider
        model_name: Model name
        api_key: API key
        save_results: Whether to save results

    Returns:
        Search results dictionary
    """
    from ai_analyzer.agents.recipe_search_manager import run_agentic_recipe_search

    results = run_agentic_recipe_search(
        supplier_brief=supplier_brief,
        recipe_manager=recipe_manager,
        top_k=top_k,
        model_provider=model_provider,
        model_name=model_name,
        api_key=api_key
    )

    if save_results:
        workflow = RecipeSearchWorkflow()
        workflow._save_results(results, "recipe_search_results.xlsx")

    return results
