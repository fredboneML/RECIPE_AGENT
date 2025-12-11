#!/usr/bin/env python3
"""
Data Extractor & Search Router Agent

This agent is responsible for:
1. Extracting meaningful information from supplier project briefs
2. Preparing data for two-step feature-based search
3. Structuring the search query with features and text description
4. Using intelligent feature mappings for multilingual/synonym support
"""
import logging
import re
import json
import os
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
        
        # Load feature mappings for intelligent extraction
        self.feature_mappings = self._load_feature_mappings()
        
        # Generate dynamic instructions with mappings
        instructions = self._generate_instructions()

        # Create the agent
        self.agent = Agent(
            name="Data Extractor & Search Router",
            role="Extract recipe information from supplier briefs for feature-based search",
            model=OpenAIChat(
                api_key=self.api_key
            ),
            instructions=instructions
        )
        
        logger.info(f"DataExtractorRouterAgent initialized with {len(self.feature_mappings.get('feature_name_mappings', {}))} feature name mappings")
    
    def _load_feature_mappings(self) -> Dict[str, Any]:
        """Load feature mappings for intelligent extraction"""
        mappings_path = '/usr/src/app/data/feature_extraction_mappings.json'
        
        try:
            if os.path.exists(mappings_path):
                with open(mappings_path, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                logger.info(f"Loaded feature mappings: {mappings.get('stats', {})}")
                return mappings
            else:
                logger.warning(f"Feature mappings not found at {mappings_path}")
                logger.warning("Proceeding with default mappings")
                return {'feature_name_mappings': {}, 'value_mappings': {}}
        except Exception as e:
            logger.error(f"Error loading feature mappings: {e}")
            return {'feature_name_mappings': {}, 'value_mappings': {}}
    
    def _generate_instructions(self) -> List[str]:
        """Generate instructions with feature mappings"""
        base_instructions = [
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
                "2. Extract SINGLE, SPECIFIC values that match database format:",
                "   - For YES/NO features: use 'Yes' or 'No' (e.g., Starch: Yes, Artificial colors: No)",
                "   - For certifications: use 'Yes' not 'allowed' or 'Preferred' (e.g., HALAL: Yes, KOSHER: Yes)",
                "   - For pH range: MUST be MIN-MAX format (e.g., '3.0-4.1', '3.2-4.5', '2.8-4.0')",
                "     * NEVER use '<4.1' or '>3.0' - always convert to range format",
                "     * If only max given (pH <4.1), use '3.0-4.1' as default range",
                "     * If only min given (pH >3.0), use '3.0-4.5' as default range",
                "   - For Brix range: MUST be MIN-MAX format (e.g., '25-35', '30-40', '20-30')",
                "     * Extract from brief carefully: 'Fruit 30±5' means '25-35', 'Syrup 50±5' means '45-55'",
                "     * Use the Fruit Brix range, not Syrup",
                "   - For flavor/color: use PRIMARY flavor only (e.g., Flavour: Peach, Color: Orange)",
                "     * If 'Peach Apricot', extract 'Peach' as primary",
                "     * If 'Strawberry Vanilla', extract 'Strawberry' as primary",
                "   - For application: use general category (e.g., 'Yogurt', 'Ice Cream', 'Bakery')",
                "3. ONE value per feature - if multiple options, choose the PRIMARY one",
                "4. Avoid compound values - choose the MAIN one",
                "5. Format validation checklist:",
                "   ✓ pH range contains '-' (e.g., '3.0-4.1')",
                "   ✓ Brix range contains '-' (e.g., '25-35')",
                "   ✓ Boolean values are exactly 'Yes' or 'No'",
                "   ✓ Flavour is single word or two words max",
                "6. Prioritize features explicitly stated in brief",
                "",
                "OUTPUT FORMAT:",
                "Your response must be a JSON object with the following structure:",
                "{",
                "  'search_type': 'two_step',",
                "  'text_description': 'concise product description with key features',",
                "  'features': [",
                "    {'feature_name': 'Flavour', 'feature_value': 'Peach'},",
                "    {'feature_name': 'Color', 'feature_value': 'Orange'},",
                "    {'feature_name': 'Application (Fruit filling)', 'feature_value': 'Yogurt'},",
                "    {'feature_name': 'Starch', 'feature_value': 'Yes'},",
                "    {'feature_name': 'Pectin', 'feature_value': 'Yes'},",
                "    {'feature_name': 'Xanthan', 'feature_value': 'Yes'},",
                "    {'feature_name': 'HALAL', 'feature_value': 'Yes'},",
                "    {'feature_name': 'KOSHER', 'feature_value': 'Yes'},",
                "    {'feature_name': 'Artificial colors', 'feature_value': 'No'},",
                "    {'feature_name': 'pH range', 'feature_value': '3.0-4.1'},",
                "    {'feature_name': 'Brix range', 'feature_value': '25-35'}",
                "  ],",
                "  'reasoning': 'brief explanation of mapping choices'",
                "}",
                "",
                "CRITICAL: For pH and Brix range, ALWAYS use MIN-MAX format with dash '-'",
                "Example: If brief says 'pH <4.1', convert to '3.0-4.1'",
                "Example: If brief says 'Brix Fruit 30±5', convert to '25-35'",
                "",
                "IMPORTANT EXAMPLES:",
                "Boolean Features:",
                "- ✅ CORRECT: {'feature_name': 'Starch', 'feature_value': 'Yes'}",
                "- ❌ WRONG: {'feature_name': 'Starch', 'feature_value': 'allowed'} or 'Modified Starch (1442)'",
                "",
                "- ✅ CORRECT: {'feature_name': 'HALAL', 'feature_value': 'Yes'}",
                "- ❌ WRONG: {'feature_name': 'HALAL', 'feature_value': 'Preferred'} or 'allowed'",
                "",
                "- ✅ CORRECT: {'feature_name': 'Artificial colors', 'feature_value': 'No'}",
                "- ❌ WRONG: {'feature_name': 'Artificial colors', 'feature_value': 'not allowed'}",
                "",
                "Numeric Ranges (MUST contain dash '-'):",
                "- ✅ CORRECT: {'feature_name': 'pH range', 'feature_value': '3.0-4.1'}",
                "- ❌ WRONG: {'feature_name': 'pH range', 'feature_value': '<4.1'} (missing min)",
                "- ❌ WRONG: {'feature_name': 'pH range', 'feature_value': '4.1'} (not a range)",
                "",
                "- ✅ CORRECT: {'feature_name': 'Brix range', 'feature_value': '25-35'}",
                "- ❌ WRONG: {'feature_name': 'Brix range', 'feature_value': '30±5'} (convert to 25-35)",
                "- ❌ WRONG: {'feature_name': 'Brix range', 'feature_value': '30-55'} (too wide, use fruit brix)",
                "",
                "Text Values:",
                "- ✅ CORRECT: {'feature_name': 'Flavour', 'feature_value': 'Peach'}",
                "- ❌ WRONG: {'feature_name': 'Flavour', 'feature_value': 'Peach Apricot'} (use primary only)",
                "",
                "- ✅ CORRECT: {'feature_name': 'Application (Fruit filling)', 'feature_value': 'Yogurt'}",
                "- ❌ WRONG: {'feature_name': 'Application', 'feature_value': 'Mixed with white mass, On top, At bottom'}",
                "",
                "IMPORTANT:",
                "- Always use EXACT database field names (charactDescr) listed above",
                "- Keep text_description concise and searchable (focus on product type, flavor, key attributes)",
                "- Only extract features that are clearly mentioned or strongly implied",
                "- For multi-flavor products (e.g., 'Peach Apricot'), extract PRIMARY flavor only: 'Peach'",
                "- For pH/Brix ranges, ALWAYS convert to MIN-MAX format with dash: '3.0-4.1', '25-35'",
                "- All boolean features must be exactly 'Yes' or 'No', never 'allowed', 'Preferred', etc.",
                "- Always provide valid JSON output with search_type set to 'two_step'",
                "",
                "=" * 80,
                "INTELLIGENT FEATURE MAPPING (USE THIS TO UNDERSTAND USER TERMINOLOGY):",
                "=" * 80,
                self._format_feature_mappings_guide(),
                "=" * 80
            ]
        
        return base_instructions
    
    def _format_feature_mappings_guide(self) -> str:
        """Format feature mappings as a guide for the LLM"""
        guide_lines = []
        
        feature_map = self.feature_mappings.get('feature_name_mappings', {})
        value_map = self.feature_mappings.get('value_mappings', {})
        
        if feature_map:
            guide_lines.append("\nFEATURE NAME SYNONYMS (User term → Database field):")
            # Group by target feature
            reverse_map = {}
            for user_term, db_field in feature_map.items():
                if db_field not in reverse_map:
                    reverse_map[db_field] = []
                reverse_map[db_field].append(user_term)
            
            # Show important mappings
            important_features = [
                'Flavour', 'Color', 'Application (Fruit filling)',
                'Starch', 'Pectin', 'HALAL', 'KOSHER', 'VEGAN',
                'pH range', 'Brix range', 'Artificial colors'
            ]
            
            for db_field in important_features:
                if db_field in reverse_map:
                    synonyms = sorted(set(reverse_map[db_field]))[:8]  # Limit to 8 examples
                    guide_lines.append(f"  {db_field}: {', '.join(synonyms)}")
        
        if value_map:
            guide_lines.append("\nVALUE NORMALIZATION (User value → Database value):")
            # Show key value mappings
            important_value_features = ['HALAL', 'KOSHER', 'VEGAN', 'Starch', 'Pectin', 'Artificial colors']
            for feature in important_value_features:
                if feature in value_map:
                    examples = list(value_map[feature].items())[:6]
                    if examples:
                        guide_lines.append(f"  {feature}:")
                        for user_val, db_val in examples:
                            guide_lines.append(f"    '{user_val}' → '{db_val}'")
        
        guide_lines.append("\nUSE THESE MAPPINGS TO:")
        guide_lines.append("1. Map user terminology (any language/synonym) to exact database field names")
        guide_lines.append("2. Normalize user values to match database values")
        guide_lines.append("3. Handle case variations (lowercase/uppercase/mixed)")
        guide_lines.append("4. Support multilingual inputs (English, German, French, etc.)")
        
        return "\n".join(guide_lines)

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

            # ============================================================
            # DEBUG LOGGING: Full extraction details for comparison
            # ============================================================
            logger.info("=" * 80)
            logger.info("DEBUG: EXTRACTION DETAILS FOR DEBUGGING")
            logger.info("=" * 80)
            
            # Log the input (combined brief)
            logger.info("INPUT (Combined Brief):")
            logger.info("-" * 40)
            for line in combined_brief.strip().split('\n'):
                logger.info(f"  {line}")
            logger.info("-" * 40)
            
            # Log the extracted text description (FULL, not truncated)
            logger.info("EXTRACTED TEXT DESCRIPTION:")
            logger.info("-" * 40)
            text_desc = result.get('text_description', 'N/A')
            logger.info(f"  {text_desc}")
            logger.info("-" * 40)
            
            # Log ALL extracted features with their values
            logger.info("EXTRACTED FEATURES AND VALUES:")
            logger.info("-" * 40)
            if result.get('features'):
                for i, feature in enumerate(result['features'], 1):
                    feature_name = feature.get('feature_name', 'N/A')
                    feature_value = feature.get('feature_value', 'N/A')
                    logger.info(f"  {i:2d}. {feature_name}: {feature_value}")
            else:
                logger.info("  No features extracted")
            logger.info("-" * 40)
            
            # Log reasoning
            logger.info("EXTRACTION REASONING:")
            logger.info("-" * 40)
            logger.info(f"  {result.get('reasoning', 'N/A')}")
            logger.info("=" * 80)
            # ============================================================

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
