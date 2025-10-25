#!/usr/bin/env python3
from ai_analyzer.utils.model_logger import query_llm
from qdrant_recipe_manager import QdrantRecipeManager
import os
import logging
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sys

# Add the src directory to the path to import recipe search modules
sys.path.append(os.path.join(os.path.dirname(
    __file__), '..', '..', '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_language_with_ai(text: str) -> str:
    """Use AI to detect the language of the input text"""
    if not text or not text.strip():
        return "en"

    try:
        # Create a prompt for language detection focused on recipe context
        prompt = f"""Analyze the following text and determine its language. This text is a recipe description or search query for finding similar recipes in a food product database.

Text: "{text}"

The text may contain:
- Recipe names and descriptions
- Food ingredients and characteristics
- Product specifications (Color, Flavour, Stabilizer, etc.)
- Industry terms (Dairy, Halal, Kosher, etc.)
- Technical specifications (Brix, Pasteurization, etc.)

Please identify the language and respond with ONLY the language code from this list:
- "en" for English
- "it" for Italian  
- "fr" for French
- "de" for German
- "es" for Spanish
- "pt" for Portuguese
- "nl" for Dutch
- "da" for Danish

If the text contains multiple languages or is unclear, default to "en" (English).

Respond with only the language code, nothing else."""

        # Make AI call for language detection
        response = query_llm(prompt, provider="openai")

        if response:
            # Clean the response and extract language code
            language_code = response.strip().lower()

            # Validate the response
            valid_codes = ["en", "it", "fr", "de", "es", "pt", "nl", "da"]
            if language_code in valid_codes:
                logger.info(
                    f"AI detected language: {language_code} for query: '{text[:100]}...'")
                return language_code
            else:
                logger.warning(
                    f"AI returned invalid language code: {language_code}, defaulting to English")
                return "en"
        else:
            logger.warning(
                "AI language detection failed, defaulting to English")
            return "en"

    except Exception as e:
        logger.error(f"Error in AI language detection: {e}")
        logger.exception("Detailed error:")
        return "en"


def format_response_in_language(results: List[Dict[str, Any]], language: str) -> str:
    """Format the response in the specified language"""
    if not results:
        if language == "nl":
            return "Geen recepten gevonden die overeenkomen met uw beschrijving. Probeer een andere zoekterm of geef meer details over het recept dat u zoekt."
        elif language == "fr":
            return "Aucune recette trouvée correspondant à votre description. Essayez un terme de recherche différent ou fournissez plus de détails sur la recette que vous recherchez."
        elif language == "de":
            return "Keine Rezepte gefunden, die Ihrer Beschreibung entsprechen. Versuchen Sie einen anderen Suchbegriff oder geben Sie mehr Details über das Rezept an, das Sie suchen."
        else:
            return "No recipes found matching your description. Please try a different search term or provide more details about the recipe you're looking for."

    # Create language-specific response parts
    if language == "nl":
        response_parts = [
            f"Gevonden {len(results)} vergelijkbare recepten:\n\n"]
        desc_prefix = "   Beschrijving: "
        score_prefix = "   Overeenkomst Score: "
        feature_prefix = "   Feature Score: "
    elif language == "fr":
        response_parts = [f"Trouvé {len(results)} recettes similaires:\n\n"]
        desc_prefix = "   Description: "
        score_prefix = "   Score de Correspondance: "
        feature_prefix = "   Score de Caractéristique: "
    elif language == "de":
        response_parts = [f"Gefunden {len(results)} ähnliche Rezepte:\n\n"]
        desc_prefix = "   Beschreibung: "
        score_prefix = "   Ähnlichkeits-Score: "
        feature_prefix = "   Feature-Score: "
    else:  # English
        response_parts = [f"Found {len(results)} similar recipes:\n\n"]
        desc_prefix = "   Description: "
        score_prefix = "   Similarity Score: "
        feature_prefix = "   Feature Score: "

    for i, result in enumerate(results, 1):
        recipe_id = result.get("id", f"recipe_{i}")
        description = result.get("description", "")
        text_score = result.get("text_score", 0)
        feature_score = result.get("feature_score")
        combined_score = result.get("combined_score", text_score)

        response_parts.append(f"{i}. Recipe ID: {recipe_id}")
        response_parts.append(
            f"{desc_prefix}{description[:200]}{'...' if len(description) > 200 else ''}")
        response_parts.append(f"{score_prefix}{combined_score:.3f}")

        if feature_score is not None:
            response_parts.append(f"{feature_prefix}{feature_score:.3f}")

        response_parts.append("")  # Empty line for readability

    return "\n".join(response_parts)


def format_response_in_language_with_ai(results: List[Dict[str, Any]], language: str, original_query: str) -> str:
    """Use AI to format the response in the specified language"""
    try:
        # Prepare the results data for AI formatting
        results_data = []
        for i, result in enumerate(results, 1):
            results_data.append({
                "rank": i,
                "id": result.get("id", f"recipe_{i}"),
                "description": result.get("description", ""),
                "similarity_score": result.get("combined_score", result.get("text_score", 0)),
                "feature_score": result.get("feature_score")
            })

        # Create language-specific instructions
        language_instructions = {
            "en": "in English",
            "nl": "in Dutch (Nederlands)",
            "fr": "in French (Français)",
            "de": "in German (Deutsch)",
            "it": "in Italian (Italiano)",
            "es": "in Spanish (Español)",
            "pt": "in Portuguese (Português)",
            "da": "in Danish (Dansk)"
        }

        lang_instruction = language_instructions.get(language, "in English")

        # Create AI prompt for response formatting focused on recipe similarity
        prompt = f"""You are a helpful recipe search assistant specializing in finding similar food products and recipes. Format the following recipe search results {lang_instruction}.

Original user query: "{original_query}"
Number of similar recipes found: {len(results)}

Recipe Results:
{json.dumps(results_data, indent=2)}

Please format this as a natural, helpful response that:
1. Announces how many similar recipes were found
2. Lists each recipe with its description and similarity score
3. Uses appropriate language and tone for {lang_instruction}
4. Keeps descriptions concise but informative
5. Includes similarity scores to help the user understand relevance
6. Focuses on recipe similarity and food product characteristics
7. Mentions key features like Color, Flavour, Stabilizer, Industry, etc. when relevant

Format the response in a clear, readable way with proper numbering and structure. Emphasize that these are similar recipes found in the database."""

        # Make AI call for response formatting
        response = query_llm(prompt, provider="openai")

        if response:
            logger.info(
                f"AI formatted response in {language} for {len(results)} recipes")
            return response
        else:
            logger.warning("AI response formatting failed, using fallback")
            return format_response_in_language(results, language)

    except Exception as e:
        logger.error(f"Error in AI response formatting: {e}")
        logger.exception("Detailed error:")
        return format_response_in_language(results, language)


class RecipeSearchAgent:
    """Agent for handling recipe search functionality"""

    def __init__(self, collection_name: str = "food_recipes_two_step"):
        """Initialize the recipe search agent"""
        self.recipe_manager = None
        self.collection_name = collection_name
        self._initialize_recipe_manager()

    def _initialize_recipe_manager(self):
        """Initialize the recipe search manager using Qdrant"""
        try:
            logger.info("Initializing Qdrant recipe search manager...")
            self.recipe_manager = QdrantRecipeManager(
                collection_name=self.collection_name,
                embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
                qdrant_host="qdrant",
                qdrant_port=6333
            )
            logger.info(
                "Qdrant recipe search manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Qdrant recipe manager: {e}")
            logger.exception("Detailed error:")

    def search_recipes(self,
                       description: str,
                       features: Optional[List[Dict[str, str]]] = None,
                       text_top_k: int = 20,
                       final_top_k: int = 3) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, str]:
        """
        Search for similar recipes based on description and optional features

        Args:
            description: Recipe description (mandatory)
            features: Optional list of feature dictionaries with 'charactDescr' and 'valueCharLong'
            text_top_k: Number of candidates from text search
            final_top_k: Final number of results to return

        Returns:
            Tuple of (results, metadata, formatted_response, detected_language)
        """
        try:
            if not self.recipe_manager:
                logger.error("Recipe manager not initialized")
                return [], {"error": "Recipe service not available"}, "Recipe service not available"

            # Validate input
            if not description.strip():
                logger.warning("Empty description provided")
                return [], {"error": "Recipe description is required"}, "Recipe description is required"

            # Detect language of the user's query using AI
            detected_language = detect_language_with_ai(description)
            logger.info(
                f"AI detected language: {detected_language} for query: '{description[:100]}...'")

            logger.info(
                f"Searching recipes for description: '{description[:100]}...'")

            # Prepare query DataFrame if features are provided
            query_df = None
            if features:
                try:
                    # Convert features to DataFrame format
                    features_data = []
                    for feature in features:
                        if 'charactDescr' in feature and 'valueCharLong' in feature:
                            features_data.append({
                                'charactDescr': feature['charactDescr'],
                                'valueCharLong': feature['valueCharLong']
                            })

                    if features_data:
                        query_df = pd.DataFrame(features_data)
                        logger.info(
                            f"Using {len(features_data)} features for refinement")
                except Exception as e:
                    logger.warning(f"Error processing features: {e}")
                    query_df = None

            # Run two-step search using Qdrant
            results, metadata = self.recipe_manager.search_two_step(
                text_description=description,
                query_df=query_df,
                text_top_k=text_top_k,
                final_top_k=final_top_k
            )

            # Format response in the detected language using AI
            formatted_response = format_response_in_language_with_ai(
                results, detected_language, description)

            logger.info(f"Found {len(results)} recipes")
            return results, metadata, formatted_response, detected_language

        except Exception as e:
            logger.error(f"Error in recipe search: {e}")
            logger.exception("Detailed error:")
            return [], {"error": f"Error searching recipes: {str(e)}"}, f"Error searching recipes: {str(e)}", "en"

    def get_service_status(self) -> Dict[str, Any]:
        """Get the status of the recipe search service"""
        if not self.recipe_manager:
            return {
                "status": "unavailable",
                "message": "Recipe service not initialized",
                "total_recipes": 0
            }

        try:
            # Get collection info from Qdrant
            collection_info = self.recipe_manager.qdrant_client.get_collection(
                self.collection_name)
            total_recipes = collection_info.points_count

            return {
                "status": "available",
                "message": "Recipe service is running",
                "total_recipes": total_recipes,
                "collection_name": self.collection_name,
                "search_capability": "two_step_search",
                "qdrant_connection": "active"
            }
        except Exception as e:
            logger.error(f"Error getting recipe service status: {e}")
            return {
                "status": "error",
                "message": f"Error getting service status: {str(e)}",
                "total_recipes": 0
            }

    def generate_followup_questions(self, search_results: List[Dict[str, Any]], original_query: str = "", language: str = "en") -> List[str]:
        """Generate recipe-specific follow-up questions to help users find more similar recipes in the detected language"""
        try:
            # Language-specific instructions for AI
            language_instructions = {
                "en": "in English",
                "nl": "in Dutch (Nederlands)",
                "fr": "in French (Français)",
                "de": "in German (Deutsch)",
                "it": "in Italian (Italiano)",
                "es": "in Spanish (Español)",
                "pt": "in Portuguese (Português)",
                "da": "in Danish (Dansk)"
            }

            lang_instruction = language_instructions.get(
                language, "in English")

            # Add AI-generated contextual questions based on the original query
            if original_query:
                try:
                    ai_prompt = f"""Based on this recipe search query and the results found, generate 3-5 helpful follow-up questions {lang_instruction} that would help the user find more similar recipes.

Original Query: "{original_query}"
Number of Results Found: {len(search_results)}
Has Results: {"Yes" if search_results else "No"}

The user is looking for similar recipes in a food product database. Generate questions {lang_instruction} that help them:
1. Refine their search with specific features
2. Explore related product categories
3. Specify technical requirements
4. Find alternatives with similar characteristics

Focus on food industry terms like: Color, Flavour, Stabilizer, Industry, Product Line, Dietary requirements, Processing methods, etc.

IMPORTANT: Generate the questions {lang_instruction}. Use natural, conversational language appropriate for {lang_instruction}.

Generate 3-5 specific, actionable questions that would help find more similar recipes."""

                    ai_response = query_llm(
                        ai_prompt, provider="openai")

                    if ai_response:
                        # Parse AI response into individual questions
                        ai_questions = [q.strip() for q in ai_response.split(
                            '\n') if q.strip() and '?' in q]
                        # Return AI-generated questions
                        return ai_questions[:5]

                except Exception as e:
                    logger.warning(
                        f"Error generating AI follow-up questions: {e}")

            # Fallback questions in English if no query provided
            return [
                "Would you like to refine your search with specific features?",
                "Are you looking for recipes with similar characteristics?",
                "Would you like to specify dietary requirements?",
                "Do you want to explore different product categories?",
                "Would you like to search by flavor profile?"
            ]

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            logger.exception("Detailed error:")
            return [
                "Would you like to refine your search with specific features?",
                "Are you looking for recipes with similar characteristics?",
                "Would you like to specify dietary requirements?",
                "Do you want to explore different product categories?",
                "Would you like to search by flavor profile?"
            ]
