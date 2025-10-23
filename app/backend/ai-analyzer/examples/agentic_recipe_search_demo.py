#!/usr/bin/env python3
"""
Agentic Recipe Search Demo

This script demonstrates how to use the new agentic recipe search system
with the RecipeSearchManager and its coordinated agents.
"""
import pandas as pd
import json
from ai_analyzer.agents.recipe_search_workflow import RecipeSearchWorkflow
from src.two_step_recipe_search import EnhancedTwoStepRecipeManager
import os
import sys
import logging

# Add paths for imports
sys.path.insert(0, '/usr/src/app')
sys.path.insert(0, '/usr/src/app/src')
sys.path.insert(0, '/usr/src/app/ai-analyzer')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_recipes():
    """Load sample recipes from the data directory"""
    from pathlib import Path

    # Define data directory
    data_dir = Path('/usr/src/app/ai-analyzer/data')

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return None, None, None, None

    # Get all JSON files
    recipe_files = list(data_dir.glob('*.json'))

    if not recipe_files:
        logger.warning("No recipe files found in data directory")
        return None, None, None, None

    logger.info(f"Found {len(recipe_files)} recipe files")

    # Import functions from index_recipe.py
    sys.path.insert(0, '/usr/src/app/src')
    from src.index_recipe import (
        read_recipe_json,
        extract_recipe_description
    )

    features_list = []
    values_list = []
    descriptions_list = []
    metadata_list = []

    for recipe_file in recipe_files[:10]:  # Load first 10 for demo
        try:
            # Extract features and values
            recipe_data = read_recipe_json(str(recipe_file))

            # Extract description
            description = extract_recipe_description(str(recipe_file))

            # Check if we got valid data
            if recipe_data is not None and isinstance(recipe_data, pd.DataFrame):
                if 'charactDescr' in recipe_data.columns and 'valueCharLong' in recipe_data.columns:
                    recipe_temp = recipe_data[[
                        'charactDescr', 'valueCharLong']]
                    features_list.append(recipe_temp['charactDescr'].tolist())
                    values_list.append(recipe_temp['valueCharLong'].tolist())
                    descriptions_list.append(description)

                    recipe_filename = recipe_file.stem
                    metadata_list.append({"recipe_name": recipe_filename})

                    logger.info(f"Loaded: {recipe_file.name}")
        except Exception as e:
            logger.error(f"Error loading {recipe_file.name}: {e}")
            continue

    logger.info(f"Successfully loaded {len(features_list)} recipes")
    return features_list, values_list, descriptions_list, metadata_list


def demo_example_1_text_only():
    """Demo Example 1: Text-only search (from supplier brief)"""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 1: Text-Only Search with Supplier Brief")
    logger.info("="*80 + "\n")

    # Supplier brief (from the PDF image you provided)
    supplier_brief = """
    Fruit Prep Supplier Project Brief - Peach Apricot
    
    Flavour Inspiration:
    Flavour Concept: Peach Apricot
    
    Description:
    Fruit: Yellow peach particulates, max 12mm in size. Can also use a peach puree.
    Apricot Puree. Fruit content to be >30%.
    
    Flavour profile: Balanced peach and apricot flavours. Sweetness and slight tartness of
    the fruit. Dessert like poached peaches and apricots with compliment GD yogurt.
    
    Colour: Vibrant deep orange colour with visible particulates. Not artificial looking. All
    N1 colours are available for development.
    
    FRUIT PREP ATTRIBUTES:
    - Particles Size: 12mm max
    - Amount of Fruit (if applicable): >30%
    - pH/Acidity: <4.1
    - Brix: Fruit 30+/-5°, Syrup 50+/-5°
    - Viscosity (20°C 60 seconds): 6-9 +/- 2 cm
    - Key Product Claims: No Preservatives, Artificial Colours or Flavours
    - Allowed Stabilization System: Modified Starch (1442), Pectin (440), Guar (412), Xanthan (415), LBG (410)
    - Allowed Acidifying Agents: Citric Acid, Malic Acid, Lactic Acid or Ascorbic Acid
    - Allowed Flavouring Agents: Only Natural Flavours
    - Allowed Colouring Agents: Only Natural Colours. Natural Colours as per Natcol N1
    - Allowed Allergen: Milk containing products
    - Religious Certification: Halal & Kosher Preferred
    - Shelf Life: Min 3 months
    """

    try:
        # Initialize workflow
        workflow = RecipeSearchWorkflow(
            collection_name="food_recipes_two_step",
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
            max_features=200,
            default_top_k=3
        )

        # Load recipes
        features_list, values_list, descriptions_list, metadata_list = load_sample_recipes()

        if features_list is None or len(features_list) == 0:
            logger.error("No recipes loaded. Exiting demo.")
            return

        workflow.load_recipes(
            features_list=features_list,
            values_list=values_list,
            descriptions_list=descriptions_list,
            metadata_list=metadata_list
        )

        # Execute search
        results = workflow.search(
            supplier_brief=supplier_brief,
            top_k=3,
            save_results=True,
            output_file="example_1_text_only_results.xlsx"
        )

        # Display results
        print("\n" + "="*80)
        print("SEARCH RESULTS - Example 1: Text-Only Search")
        print("="*80 + "\n")

        if 'results_table' in results and not results['results_table'].empty:
            print("Top Matching Recipes:")
            print(results['results_table'].to_string(index=False))
            print("\n")

        if 'explanation' in results:
            print("Natural Language Explanation:")
            print("-" * 80)
            print(results['explanation'])
            print("\n")

        if 'summary' in results:
            print("Search Summary:")
            print(f"  Total Results: {results['summary']['total_results']}")
            print(f"  Search Type: {results['summary']['search_type']}")
            print(f"  Top Recipe: {results['summary']['top_recipe']}")
            print(f"  Top Score: {results['summary']['top_score']:.4f}")
            print("\n")

        logger.info("Example 1 completed successfully")

    except Exception as e:
        logger.error(f"Error in Example 1: {e}")
        logger.exception("Detailed error:")


def demo_example_2_two_step():
    """Demo Example 2: Two-step search with feature refinement"""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 2: Two-Step Search with Feature Refinement")
    logger.info("="*80 + "\n")

    # Supplier brief with specific features
    supplier_brief = """
    Recipe Name: 521082 FIT BANANA
    MaterialMasterShorttext: DELFP BANANA H FLIP FIT CHO
    Colour Yellow, Flavour Banana, Stabilizer StarchLBGPectin, SMEAR 3M TPC
    
    Product Requirements:
    - Industry: Dairy
    - Standard product with Halal and Kosher certification
    - No saccharose, No preservatives
    - No artificial colors, No nature identical flavour
    - Natural flavour required
    - Not gene manipulated
    - No allergens
    - Stabilizers: Starch, Pectin, LBG
    - No blend, No other stabilizer
    - Color required
    - No Aspartame, No Xanthan
    - Pasteurization type: pasteurized
    - Launch Type: New Product
    - Non-Plant-based
    - Produktsegment: Spoonable Yoghurt
    """

    try:
        # Initialize workflow
        workflow = RecipeSearchWorkflow(
            collection_name="food_recipes_two_step",
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
            max_features=200,
            default_top_k=3
        )

        # Load recipes
        features_list, values_list, descriptions_list, metadata_list = load_sample_recipes()

        if features_list is None or len(features_list) == 0:
            logger.error("No recipes loaded. Exiting demo.")
            return

        workflow.load_recipes(
            features_list=features_list,
            values_list=values_list,
            descriptions_list=descriptions_list,
            metadata_list=metadata_list
        )

        # Execute search
        # The agent will automatically detect that this requires two-step search
        results = workflow.search(
            supplier_brief=supplier_brief,
            top_k=3,
            save_results=True,
            output_file="example_2_two_step_results.xlsx"
        )

        # Display results
        print("\n" + "="*80)
        print("SEARCH RESULTS - Example 2: Two-Step Search")
        print("="*80 + "\n")

        if 'results_table' in results and not results['results_table'].empty:
            print("Top Matching Recipes:")
            print(results['results_table'].to_string(index=False))
            print("\n")

        if 'explanation' in results:
            print("Natural Language Explanation:")
            print("-" * 80)
            print(results['explanation'])
            print("\n")

        if 'summary' in results:
            print("Search Summary:")
            print(f"  Total Results: {results['summary']['total_results']}")
            print(f"  Search Type: {results['summary']['search_type']}")
            print(f"  Top Recipe: {results['summary']['top_recipe']}")
            print(f"  Top Score: {results['summary']['top_score']:.4f}")
            print("\n")

        # Show intermediate outputs
        if 'intermediate_outputs' in results:
            print("Intermediate Agent Outputs:")
            print("-" * 80)
            for stage, data in results['intermediate_outputs'].items():
                print(f"\n{stage}:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
            print("\n")

        logger.info("Example 2 completed successfully")

    except Exception as e:
        logger.error(f"Error in Example 2: {e}")
        logger.exception("Detailed error:")


def main():
    """Main function to run demos"""
    print("\n" + "="*80)
    print("AGENTIC RECIPE SEARCH DEMO")
    print("="*80 + "\n")

    print("This demo showcases the new agentic recipe search system")
    print("with intelligent routing, search, and natural language explanations.")
    print("\n")

    # Run Example 1: Text-only search
    demo_example_1_text_only()

    print("\n" + "="*80 + "\n")

    # Run Example 2: Two-step search with feature refinement
    demo_example_2_two_step()

    print("\n" + "="*80)
    print("DEMO COMPLETED")
    print("="*80 + "\n")

    print("Results have been saved to Excel files:")
    print("  - example_1_text_only_results.xlsx")
    print("  - example_2_two_step_results.xlsx")
    print("\n")


if __name__ == "__main__":
    main()
