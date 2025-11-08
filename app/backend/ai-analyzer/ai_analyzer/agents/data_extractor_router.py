#!/usr/bin/env python3
"""
Data Extractor & Search Router Agent

This agent is responsible for:
1. Extracting meaningful information from supplier project briefs
2. Deciding whether to use text-only search or two-step search with feature refinement
3. Preparing the search query structure
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
    """Agent for extracting data from supplier briefs and routing search strategy"""

    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4", api_key: Optional[str] = None):
        """Initialize the Data Extractor & Router Agent"""
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key or config.get('AI_ANALYZER_OPENAI_API_KEY')

        # Create the agent
        self.agent = Agent(
            name="Data Extractor & Search Router",
            role="Extract recipe information from supplier briefs and determine optimal search strategy",
            model=OpenAIChat(
                api_key=self.api_key
            ),
            instructions=[
                "You are an expert recipe analyzer who extracts structured information from supplier project briefs.",
                "Your task is to:",
                "1. Extract key recipe characteristics (flavors, colors, ingredients, attributes)",
                "2. Identify whether the brief contains specific feature requirements",
                "3. Decide if text-only search or two-step feature-based search is appropriate",
                "4. Structure the extracted information for recipe search",
                "",
                "EXTRACTION GUIDELINES:",
                "- Extract flavor/flavour information (e.g., 'peach', 'apricot', 'banana')",
                "- Extract color/colour information (e.g., 'yellow', 'orange', 'vibrant')",
                "- Extract ingredient details (fruit content, puree, particulates)",
                "- Extract stabilizer information (starch, pectin, LBG, guar, xanthan)",
                "- Extract product attributes (organic, natural, halal, kosher, GMO-free)",
                "- Extract texture/consistency information",
                "- Extract any specific technical requirements",
                "",
                "ROUTING DECISION:",
                "- Use TEXT-ONLY search when:",
                "  * The brief contains primarily descriptive text without specific feature requirements",
                "  * The brief focuses on flavor profiles, sensory descriptions, or general product concepts",
                "  * No structured attributes (like 'No preservatives', 'Halal', 'Organic') are mentioned",
                "",
                "- Use TWO-STEP search when:",
                "  * The brief contains specific feature requirements (e.g., 'No artificial colors', 'Halal')",
                "  * Structured attributes are clearly defined (e.g., 'Natural flavour', 'No preservatives')",
                "  * The brief includes technical specifications or product classifications",
                "",
                "OUTPUT FORMAT:",
                "Your response must be a JSON object with the following structure:",
                "{",
                "  'search_type': 'text_only' or 'two_step',",
                "  'text_description': 'extracted descriptive text for search',",
                "  'features': [",
                "    {'feature_name': 'Feature name', 'feature_value': 'Feature value'},",
                "    ...",
                "  ],",
                "  'reasoning': 'brief explanation of why this search type was chosen'",
                "}",
                "",
                "IMPORTANT: Always provide valid JSON output."
            ],
            markdown=False
        )

        logger.info(
            f"Initialized DataExtractorRouterAgent with model {self.model_name}")

    def extract_and_route(self, supplier_brief: str, document_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract information from supplier brief and determine search strategy

        Args:
            supplier_brief: The supplier project brief text (user input)
            document_text: Optional extracted text from uploaded document

        Returns:
            Dictionary containing:
            - search_type: 'text_only' or 'two_step'
            - text_description: Extracted text for search
            - features_df: Optional DataFrame with features (for two-step search)
            - reasoning: Explanation of routing decision
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
            # Fallback to text-only search
            return {
                'search_type': 'text_only',
                'text_description': document_text if document_text else supplier_brief,
                'features_df': None,
                'reasoning': f'Error in extraction: {str(e)}. Falling back to text-only search.'
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
                result['search_type'] = 'text_only'
            if 'text_description' not in result:
                result['text_description'] = response_text
            if 'features' not in result:
                result['features'] = []
            if 'reasoning' not in result:
                result['reasoning'] = 'No reasoning provided'

            return result

        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return text-only fallback
            return {
                'search_type': 'text_only',
                'text_description': response_text,
                'features': [],
                'reasoning': 'Failed to parse structured response. Using text-only search.'
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "agent_name": "DataExtractorRouterAgent",
            "model_provider": self.model_provider,
            "model_name": self.model_name
        }
