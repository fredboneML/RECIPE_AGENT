#!/usr/bin/env python3
"""
Recipe Search Manager Agent

This agent coordinates the entire agentic recipe search workflow:
1. Data Extractor & Router Agent - extracts info and decides search strategy
2. Search & Reranker Agent - executes search and finds top K recipes
3. Recipe Generator Agent - displays results and explains similarities
"""
from src.qdrant_recipe_manager import QdrantRecipeManager
from src.two_step_recipe_search import EnhancedTwoStepRecipeManager
from ai_analyzer.agents.recipe_generator import RecipeGeneratorAgent
from ai_analyzer.agents.search_reranker import SearchRerankerAgent
from ai_analyzer.agents.data_extractor_router import DataExtractorRouterAgent
import logging
import sys
from typing import Dict, Any, Optional, Union
import pandas as pd

# Add path for importing modules
sys.path.insert(0, '/usr/src/app/src')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecipeSearchManager:
    """Manager agent that coordinates the agentic recipe search workflow"""

    def __init__(self,
                 recipe_manager: Union[EnhancedTwoStepRecipeManager, QdrantRecipeManager],
                 model_provider: str = "openai",
                 model_name: str = "gpt-4",
                 api_key: Optional[str] = None,
                 default_top_k: int = 3):
        """
        Initialize the Recipe Search Manager

        Args:
            recipe_manager: Instance of EnhancedTwoStepRecipeManager or QdrantRecipeManager
            model_provider: Model provider (openai, groq, etc.)
            model_name: Model name to use
            api_key: API key for the model
            default_top_k: Default number of top results to return
        """
        self.recipe_manager = recipe_manager
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key
        self.default_top_k = default_top_k

        # Initialize the three specialized agents
        logger.info("Initializing specialized agents...")

        self.data_extractor = DataExtractorRouterAgent(
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key
        )

        self.search_reranker = SearchRerankerAgent(
            recipe_manager=recipe_manager,
            default_top_k=default_top_k
        )

        self.recipe_generator = RecipeGeneratorAgent(
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key
        )

        logger.info(
            "RecipeSearchManager initialized successfully with all agents")

    def search_similar_recipes(self,
                               supplier_brief: str,
                               top_k: Optional[int] = None,
                               custom_reranking: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Execute the complete agentic recipe search workflow

        Args:
            supplier_brief: The supplier project brief text
            top_k: Number of top results to return (defaults to self.default_top_k)
            custom_reranking: Optional custom reranking criteria

        Returns:
            Dictionary containing:
            - results_table: DataFrame with search results
            - explanation: Natural language explanation
            - metadata: Search metadata
            - summary: Summary statistics
            - intermediate_outputs: Outputs from each agent stage
        """
        top_k = top_k or self.default_top_k

        try:
            logger.info("=" * 80)
            logger.info("STARTING AGENTIC RECIPE SEARCH WORKFLOW")
            logger.info("=" * 80)
            logger.info(
                f"Supplier brief length: {len(supplier_brief)} characters")
            logger.info(f"Requested top K: {top_k}")
            logger.info("=" * 80)

            # STAGE 1: Data Extraction & Routing
            logger.info("\n[STAGE 1] Data Extraction & Routing")
            logger.info("-" * 80)

            extraction_result = self.data_extractor.extract_and_route(
                supplier_brief)

            search_type = extraction_result['search_type']
            text_description = extraction_result['text_description']
            features_df = extraction_result.get('features_df')
            routing_reasoning = extraction_result.get('reasoning', '')

            logger.info(f"✓ Extraction completed")
            logger.info(f"  Search Type: {search_type}")
            logger.info(
                f"  Text Description Length: {len(text_description)} chars")
            if features_df is not None:
                logger.info(
                    f"  Features Extracted: {len(features_df)} features")
            logger.info(f"  Routing Reasoning: {routing_reasoning}")

            # STAGE 2: Search & Reranking
            logger.info("\n[STAGE 2] Search & Reranking")
            logger.info("-" * 80)

            search_results, search_metadata = self.search_reranker.search_recipes(
                search_type=search_type,
                text_description=text_description,
                features_df=features_df,
                top_k=top_k
            )

            logger.info(f"✓ Search completed")
            logger.info(f"  Results Found: {len(search_results)}")
            logger.info(
                f"  Search Strategy: {search_metadata.get('search_strategy', 'unknown')}")

            # Apply custom reranking if provided
            if custom_reranking:
                logger.info("  Applying custom reranking...")
                search_results = self.search_reranker.rerank_results(
                    search_results,
                    custom_reranking
                )
                logger.info("  ✓ Reranking completed")

            # STAGE 3: Result Generation & Explanation
            logger.info("\n[STAGE 3] Result Generation & Explanation")
            logger.info("-" * 80)

            # Generate results table
            results_table = self.recipe_generator.generate_results_table(
                search_results)
            logger.info(
                f"✓ Results table generated with {len(results_table)} rows")

            # Generate natural language explanation
            explanation = self.recipe_generator.generate_explanation(
                supplier_brief=supplier_brief,
                search_type=search_type,
                results=search_results,
                metadata=search_metadata
            )
            logger.info("✓ Explanation generated")

            # Format complete output
            complete_output = self.recipe_generator.format_complete_output(
                table_df=results_table,
                explanation=explanation,
                metadata=search_metadata
            )

            # Add intermediate outputs for debugging/analysis
            complete_output['intermediate_outputs'] = {
                'stage_1_extraction': {
                    'search_type': search_type,
                    'text_description': text_description,
                    'features_extracted': len(features_df) if features_df is not None else 0,
                    'routing_reasoning': routing_reasoning
                },
                'stage_2_search': {
                    'results_count': len(search_results),
                    'search_metadata': search_metadata,
                    'reranking_applied': custom_reranking is not None
                },
                'stage_3_generation': {
                    'explanation_length': len(explanation),
                    'table_rows': len(results_table)
                }
            }

            logger.info("\n" + "=" * 80)
            logger.info(
                "AGENTIC RECIPE SEARCH WORKFLOW COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(
                f"Top Recipe: {complete_output['summary']['top_recipe']}")
            logger.info(
                f"Top Score: {complete_output['summary']['top_score']:.4f}")
            logger.info("=" * 80 + "\n")

            return complete_output

        except Exception as e:
            logger.error(f"Error in recipe search workflow: {e}")
            logger.exception("Detailed error:")

            # Return error response
            return {
                "results_table": pd.DataFrame(),
                "explanation": f"An error occurred during the recipe search: {str(e)}",
                "metadata": {"error": str(e)},
                "summary": {
                    "total_results": 0,
                    "search_type": "error",
                    "top_recipe": None,
                    "top_score": 0.0
                },
                "intermediate_outputs": {
                    "error": str(e)
                }
            }

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics from all agents"""
        return {
            "manager": {
                "model_provider": self.model_provider,
                "model_name": self.model_name,
                "default_top_k": self.default_top_k
            },
            "data_extractor": self.data_extractor.get_stats(),
            "search_reranker": self.search_reranker.get_stats(),
            "recipe_generator": self.recipe_generator.get_stats()
        }

    def validate_setup(self) -> Dict[str, bool]:
        """Validate that all agents and components are properly set up"""
        validation = {
            "recipe_manager_initialized": self.recipe_manager is not None,
            "data_extractor_initialized": self.data_extractor is not None,
            "search_reranker_initialized": self.search_reranker is not None,
            "recipe_generator_initialized": self.recipe_generator is not None,
            "recipe_data_available": len(self.recipe_manager.recipes) > 0 if self.recipe_manager else False
        }

        all_valid = all(validation.values())
        validation["all_valid"] = all_valid

        logger.info("Validation Results:")
        for key, value in validation.items():
            status = "✓" if value else "✗"
            logger.info(f"  {status} {key}: {value}")

        return validation


# Convenience function for quick searches
def run_agentic_recipe_search(supplier_brief: str,
                              recipe_manager: Union[EnhancedTwoStepRecipeManager, QdrantRecipeManager],
                              top_k: int = 3,
                              model_provider: str = "openai",
                              model_name: str = "gpt-4",
                              api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run agentic recipe search

    Args:
        supplier_brief: The supplier project brief text
        recipe_manager: Instance of EnhancedTwoStepRecipeManager or QdrantRecipeManager
        top_k: Number of top results to return
        model_provider: Model provider (openai, groq, etc.)
        model_name: Model name to use
        api_key: API key for the model

    Returns:
        Complete search results with explanation
    """
    manager = RecipeSearchManager(
        recipe_manager=recipe_manager,
        model_provider=model_provider,
        model_name=model_name,
        api_key=api_key,
        default_top_k=top_k
    )

    return manager.search_similar_recipes(supplier_brief, top_k=top_k)
