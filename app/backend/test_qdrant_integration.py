#!/usr/bin/env python3
"""
Test Qdrant Integration

This script tests the new QdrantRecipeManager to ensure it's working correctly
with the indexed recipes in Qdrant.
"""
from src.qdrant_recipe_manager import QdrantRecipeManager
import sys
import os
import logging

# Add path for imports
sys.path.insert(0, '/usr/src/app/src')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('qdrant_integration_test')


def test_qdrant_connection():
    """Test 1: Verify connection to Qdrant"""
    logger.info("=" * 80)
    logger.info("TEST 1: Qdrant Connection")
    logger.info("=" * 80)

    try:
        manager = QdrantRecipeManager(
            collection_name="food_recipes_two_step",
            embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
            qdrant_host=os.getenv('QDRANT_HOST', 'qdrant'),
            qdrant_port=int(os.getenv('QDRANT_PORT', '6333'))
        )

        logger.info("✅ Successfully connected to Qdrant")
        return manager
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        return None


def test_collection_stats(manager):
    """Test 2: Get collection statistics"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Collection Statistics")
    logger.info("=" * 80)

    try:
        stats = manager.get_stats()
        logger.info(f"Collection Name: {stats['collection_name']}")
        logger.info(f"Total Recipes: {stats['total_recipes']}")
        logger.info(f"Embedding Model: {stats['embedding_model']}")
        logger.info(
            f"Qdrant Host: {stats['qdrant_host']}:{stats['qdrant_port']}")
        logger.info(f"Storage Type: {stats['storage']}")

        if stats['total_recipes'] > 0:
            logger.info(
                f"✅ Collection has {stats['total_recipes']} recipes indexed")
            return True
        else:
            logger.warning(
                "⚠️  Collection is empty! Run init_vector_index.py to index recipes.")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        return False


def test_text_search(manager):
    """Test 3: Text-based search"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Text-Based Search")
    logger.info("=" * 80)

    query = "banana yogurt with chocolate"
    logger.info(f"Query: '{query}'")

    try:
        results = manager.search_by_text_description(query, top_k=3)

        if results:
            logger.info(f"✅ Found {len(results)} results")
            for i, result in enumerate(results, 1):
                logger.info(f"\n  Result {i}:")
                logger.info(f"    Recipe Name: {result['recipe_name']}")
                logger.info(f"    Score: {result['text_score']:.4f}")
                logger.info(
                    f"    Description: {result['description'][:100]}...")
            return True
        else:
            logger.warning("⚠️  No results found")
            return False
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return False


def test_two_step_search(manager):
    """Test 4: Two-step search (text + features)"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Two-Step Search")
    logger.info("=" * 80)

    import pandas as pd

    query_text = "peach yogurt with natural flavour"
    query_df = pd.DataFrame({
        'charactDescr': ['Flavour', 'Allergens', 'Industry (SD Reporting)'],
        'valueCharLong': ['Peach', 'No allergens', 'Dairy']
    })

    logger.info(f"Query Text: '{query_text}'")
    logger.info(f"Query Features:\n{query_df.to_string()}")

    try:
        results, metadata = manager.search_two_step(
            text_description=query_text,
            query_df=query_df,
            text_top_k=20,
            final_top_k=3
        )

        if results:
            logger.info(f"\n✅ Found {len(results)} results")
            logger.info(f"Metadata:")
            logger.info(f"  Search Type: {metadata.get('search_type')}")
            logger.info(
                f"  Text Candidates: {metadata.get('text_results_found')}")
            logger.info(
                f"  Feature Refinement: {metadata.get('has_feature_refinement')}")
            logger.info(
                f"  Final Results: {metadata.get('final_results_count')}")

            for i, result in enumerate(results, 1):
                logger.info(f"\n  Result {i}:")
                logger.info(f"    Recipe Name: {result['recipe_name']}")
                logger.info(
                    f"    Text Score: {result.get('text_score', 0):.4f}")
                logger.info(
                    f"    Feature Score: {result.get('feature_score', 0):.4f}")
                logger.info(
                    f"    Combined Score: {result.get('combined_score', 0):.4f}")
            return True
        else:
            logger.warning("⚠️  No results found")
            return False
    except Exception as e:
        logger.error(f"❌ Two-step search failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 80)
    logger.info("QDRANT INTEGRATION TEST SUITE")
    logger.info("=" * 80 + "\n")

    # Test 1: Connection
    manager = test_qdrant_connection()
    if not manager:
        logger.error("\n❌ TESTS FAILED: Cannot connect to Qdrant")
        return False

    # Test 2: Stats
    has_recipes = test_collection_stats(manager)
    if not has_recipes:
        logger.warning(
            "\n⚠️  Collection is empty. Run init_vector_index.py first.")
        return False

    # Test 3: Text search
    text_search_success = test_text_search(manager)

    # Test 4: Two-step search
    two_step_success = test_two_step_search(manager)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Connection: ✅")
    logger.info(f"Collection Stats: {'✅' if has_recipes else '⚠️'}")
    logger.info(f"Text Search: {'✅' if text_search_success else '❌'}")
    logger.info(f"Two-Step Search: {'✅' if two_step_success else '❌'}")
    logger.info("=" * 80 + "\n")

    if text_search_success and two_step_success:
        logger.info(
            "🎉 ALL TESTS PASSED! Qdrant integration is working correctly.")
        return True
    else:
        logger.error("❌ SOME TESTS FAILED. Check logs above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
