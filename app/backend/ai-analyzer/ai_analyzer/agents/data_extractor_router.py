#!/usr/bin/env python3
"""
Data Extractor & Search Router Agent

This agent is responsible for:
1. Extracting meaningful information from supplier project briefs
2. Preparing data for two-step feature-based search
3. Structuring the search query with features and text description
"""
import logging
import re
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from ai_analyzer.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractorRouterAgent:
    """Agent for extracting data from supplier briefs for two-step feature-based search"""

    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4", api_key: Optional[str] = None):
        """Initialize the Data Extractor & Router Agent"""
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key or config.get('AI_ANALYZER_OPENAI_API_KEY')

        # Create the agent
        self.agent = Agent(
            name="Data Extractor & Search Router",
            role="Extract recipe information from supplier briefs for feature-based search",
            model=OpenAIChat(
                api_key=self.api_key
            ),
            instructions=[
                "You are an expert recipe analyzer who extracts structured information from supplier project briefs.",
                "Your task is to:",
                "1. Extract key recipe characteristics and map them to database field names (charactDescr)",
                "2. Create a concise text description focusing on key features: Product type, Flavor/Flavour, Color, Application, and Attributes",
                "3. Structure the extracted information for two-step feature-based search",
                "",
                "DATABASE FIELD NAMES (charactDescr) - USE THESE EXACT NAMES:",
                "Core Features:",
                "- Flavour OR Flavor (use based on context)",
                "- Color OR Farbe",
                "- Product Line OR Produktlinie",
                "- Application (Fruit filling) OR Fruit Prep Application",
                "- Customer Brand OR Kunden Marke",
                "- Industry (SD Reporting) OR Produktsegment (SD Reporting)",
                "",
                "Ingredients & Composition:",
                "- Starch OR Stärke (for starch stabilizer)",
                "- Pectin OR Pektin",
                "- Xanthan",
                "- LBG (Locust Bean Gum)",
                "- Guarkernmehl (Guar gum)",
                "- Other stabilizer OR Andere Stabilisatoren",
                "- Sweetening system OR Süßstoff",
                "- Natural flavor OR Natürliche Aromen",
                "",
                "Product Attributes:",
                "- HALAL OR Halal",
                "- KOSHER OR Kosher",
                "- VEGAN",
                "- VEGETARISCH OR Type vegan / vegetarian",
                "- ORGANIC OR Bio zertifiziert OR Auslobung BIO",
                "- GMO-free DE OR GMO frei EU Reg. 1829/1830 OR GMO-free AT",
                "- Clean label - No additives",
                "- Glutenfrei <20ppm OR Gluten/glutenhaltige Getreide",
                "- Artificial colors (yes/no)",
                "- Allergene OR Allergens",
                "",
                "Technical Parameters:",
                "- pH range",
                "- Brix range",
                "- Viskositaet 30S 20°C OR Thickness",
                "- Fruit restants OR Puree/Pieces",
                "",
                "TEXT DESCRIPTION GUIDELINES:",
                "Create a concise description (2-3 sentences max) that includes:",
                "- Product type and application (e.g., 'Fruit preparation for yogurt', 'Skyr flavor')",
                "- Main flavor/ingredient (e.g., 'Matcha tea', 'Peach apricot')",
                "- Key differentiators (e.g., 'organic', 'no added sugar', 'with particulates')",
                "- Target application if specified",
                "Example: 'Matcha tea fruit preparation for Skyr yogurt application, natural flavor, low sugar content, starch stabilized'",
                "",
                "FEATURE EXTRACTION RULES:",
                "1. Map user terms to exact database field names listed above",
                "2. For stabilizers: if 'starch' mentioned → use 'Starch' OR 'Stärke'",
                "3. For certifications: use uppercase versions (HALAL, KOSHER, VEGAN, ORGANIC)",
                "4. For boolean attributes: use 'yes'/'no' or 'allowed'/'not allowed'",
                "5. Extract specific values when mentioned (e.g., pH <4.1, Brix 30±5)",
                "6. Prioritize features that are explicitly stated in the brief",
                "",
                "OUTPUT FORMAT:",
                "Your response must be a JSON object with the following structure:",
                "{",
                "  'search_type': 'two_step',",
                "  'text_description': 'concise product description with key features',",
                "  'features': [",
                "    {'feature_name': 'Exact charactDescr name from database', 'feature_value': 'Specific value'},",
                "    ...",
                "  ],",
                "  'reasoning': 'brief explanation of mapping choices'",
                "}",
                "",
                "IMPORTANT:",
                "- Always use EXACT database field names (charactDescr) listed above",
                "- Keep text_description concise and searchable (focus on product type, flavor, key attributes)",
                "- Only extract features that are clearly mentioned or strongly implied",
                "- Always provide valid JSON output with search_type set to 'two_step'"
            ],
            markdown=False
        )

        logger.info(
            f"Initialized DataExtractorRouterAgent with model {self.model_name}")

    def extract_and_route(self, supplier_brief: str, document_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract information from supplier brief for two-step feature-based search

        Args:
            supplier_brief: The supplier project brief text (user input)
            document_text: Optional extracted text from uploaded document

        Returns:
            Dictionary containing:
            - search_type: 'two_step' (always)
            - text_description: Extracted text for search
            - features_df: DataFrame with features for two-step search
            - reasoning: Explanation of extracted features
        """
        try:
            # Combine user input with document text if available
            if document_text:
                logger.info("Processing supplier brief with uploaded document")
                combined_brief = f"""
USER DESCRIPTION:
{supplier_brief}

EXTRACTED DOCUMENT CONTENT:
{document_text}
"""
            else:
                logger.info("Processing supplier brief from text input only")
                combined_brief = supplier_brief

            # Prepare the prompt
            prompt = f"""
Analyze the following supplier project brief and extract relevant recipe information:

SUPPLIER BRIEF:
{combined_brief}

Extract the key information and decide on the appropriate search strategy.
Provide your response as a JSON object following the specified format.
"""

            # Get response from agent
            logger.info("Extracting information from supplier brief...")
            response = self.agent.run(prompt)

            # Parse the response
            response_content = str(response.content) if hasattr(
                response, 'content') else str(response)
            result = self._parse_agent_response(response_content)

            # If two-step search, convert features to DataFrame
            if result['search_type'] == 'two_step' and result.get('features'):
                features_df = pd.DataFrame(result['features'])
                # Rename columns to match expected format
                features_df = features_df.rename(columns={
                    'feature_name': 'charactDescr',
                    'feature_value': 'valueCharLong'
                })
                result['features_df'] = features_df
            else:
                result['features_df'] = None

            logger.info(
                f"Extraction completed. Search type: {result['search_type']}")
            logger.info(f"Reasoning: {result.get('reasoning', 'N/A')}")

            return result

        except Exception as e:
            logger.error(f"Error in data extraction and routing: {e}")
            # Fallback to two-step search with empty features
            return {
                'search_type': 'two_step',
                'text_description': document_text if document_text else supplier_brief,
                'features_df': None,
                'reasoning': f'Error in extraction: {str(e)}. Falling back to two-step search with text only.'
            }

    def _parse_agent_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the agent's JSON response"""
        import json

        try:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(
                r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

            # Parse JSON
            result = json.loads(json_str)

            # Validate required fields
            if 'search_type' not in result:
                result['search_type'] = 'two_step'
            if 'text_description' not in result:
                result['text_description'] = response_text
            if 'features' not in result:
                result['features'] = []
            if 'reasoning' not in result:
                result['reasoning'] = 'No reasoning provided'

            return result

        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return two-step fallback
            return {
                'search_type': 'two_step',
                'text_description': response_text,
                'features': [],
                'reasoning': 'Failed to parse structured response. Using two-step search with text only.'
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "agent_name": "DataExtractorRouterAgent",
            "model_provider": self.model_provider,
            "model_name": self.model_name
        }
