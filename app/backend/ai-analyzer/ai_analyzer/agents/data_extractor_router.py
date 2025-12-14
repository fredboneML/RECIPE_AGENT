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

        # Create the agent with low temperature for consistent extraction
        # Temperature 0.0 = deterministic output (same input → same output)
        self.agent = Agent(
            name="Data Extractor & Search Router",
            role="Extract recipe information from supplier briefs for feature-based search",
            model=OpenAIChat(
                api_key=self.api_key,
                temperature=0.0  # Deterministic extraction for consistent results
            ),
            instructions=instructions
        )

        logger.info(
            f"DataExtractorRouterAgent initialized with {len(self.feature_mappings.get('feature_name_mappings', {}))} feature name mappings")

    def _load_feature_mappings(self) -> Dict[str, Any]:
        """Load feature mappings for intelligent extraction"""
        mappings_path = '/usr/src/app/data/feature_extraction_mappings.json'

        try:
            if os.path.exists(mappings_path):
                with open(mappings_path, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                logger.info(
                    f"Loaded feature mappings: {mappings.get('stats', {})}")
                return mappings
            else:
                logger.warning(
                    f"Feature mappings not found at {mappings_path}")
                logger.warning("Proceeding with default mappings")
                return {'feature_name_mappings': {}, 'value_mappings': {}}
        except Exception as e:
            logger.error(f"Error loading feature mappings: {e}")
            return {'feature_name_mappings': {}, 'value_mappings': {}}

    def _generate_instructions(self) -> List[str]:
        """Generate instructions with feature mappings"""
        base_instructions = [
            "You are an expert recipe analyzer for a fruit preparation/ingredient database.",
            "Your task is to extract structured information from supplier project briefs to search 600K+ recipes.",
            "",
            "=" * 80,
            "TEXT DESCRIPTION (CRITICAL FOR SEARCH):",
            "=" * 80,
            "Create a RICH, SEARCHABLE description (3-5 sentences) that includes:",
            "- Product type: fruit preparation, fruit filling, compound, puree, etc.",
            "- Application: yogurt, ice cream, bakery, beverage, dairy, quark/skyr",
            "- Main flavor(s): Matcha, Peach, Strawberry, Vanilla, etc.",
            "- Texture: with pieces, smooth, chunky, liquid (Flüssig/Stückig)",
            "- Key attributes: organic, natural, no artificial colors, low sugar, etc.",
            "",
            "EXAMPLE TEXT DESCRIPTIONS:",
            "- 'Matcha tea fruit preparation for skyr and quark application, natural flavor, starch stabilized, no artificial colors, with pieces'",
            "- 'Peach apricot fruit filling for yogurt, contains pectin stabilizer, halal and kosher certified, smooth texture'",
            "- 'Strawberry fruit preparation for ice cream application, organic certified, no added sugar, with fruit pieces'",
            "",
            "=" * 80,
            "FEATURE EXTRACTION - DATABASE FIELD NAMES (charactDescr):",
            "=" * 80,
            "",
            "CORE PRODUCT FEATURES:",
            "- Flavour: The main flavor (Matcha, Peach, Strawberry, Vanilla, Mango, etc.)",
            "- Farbe / Color: Product color or 'Keine Farbe enthalten' (no color)",
            "- Produktsegment (SD Reporting): Target segment - Extract EXACT value from brief (e.g., Käse, Quark/Topfen, Joghurt, Eiscreme, Backwaren, Molkerei). DO NOT infer - use the EXACT value shown in the brief.",
            "- Industrie (SD Reporting): Industry (Molkerei/Dairy, Backwaren/Bakery, Getränke/Beverage) - Different from Produktsegment!",
            "- Fruit Prep Application / Application (Fruit filling): Usage context",
            "- Produktkategorien: Product category (Fruchtzubereitung, Fruchtpüree, Compound)",
            "- Flüssig/Stückig: Texture - Flüssig (liquid/smooth) or Stückig (with pieces/chunks)",
            "",
            "STABILIZERS & INGREDIENTS (use database values, NOT Yes/No):",
            "- Stärke: 'Stärke enthalten' (contains starch) OR 'Keine Stärke' (no starch)",
            "- Pektin: 'Pektin enthalten' (contains pectin) OR 'Kein Pektin' (no pectin)",
            "- Xanthan: 'Xanthan enthalten' OR 'Kein Xanthan'",
            "- Guarkernmehl: Guar gum - 'Guarkernmehl enthalten' OR 'Kein Guarkernmehl'",
            "- LBG: Locust bean gum presence",
            "- Andere Stabilisatoren: 'Keine anderen Stabil enthalten' OR stabilizer name",
            "- Natürliche Aromen: 'Natürliches Aroma' (natural) OR 'Kein naturidentes Aroma'",
            "- Süßstoff: 'keine Süsstoffe' (no sweeteners) OR sweetener type",
            "- Saccharose: Sucrose - 'Saccharose' OR 'keine Saccharose'",
            "- Konservierung: 'Nicht konserviert' (no preservatives) OR preservative type",
            "",
            "CERTIFICATIONS (use database values):",
            "- HALAL: 'suitable HALAL' OR 'not suitable HALAL'",
            "- KOSHER: 'suitable KOSHER' OR 'certified KOSHER' OR 'Not suitable for kosher'",
            "- VEGAN: 'suitable VEGAN' OR 'not suitable VEGAN'",
            "- Bio zertifiziert / Auslobung BIO: 'Bio' (organic) OR 'Nicht Bio'",
            "- GMO-free DE: 'GMO-frei' OR 'Nicht GMO-frei'",
            "- GMO frei EU Reg. 1829/1830: EU GMO-free regulation",
            "- Clean label - No additives: 'Clean Label' OR 'Nicht Clean Label'",
            "",
            "ALLERGENS & DIETARY:",
            "- Künstliche Farben: 'keine künstl. Farbe' (no artificial colors) OR 'mit künstl. Farbe'",
            "- Allergene: 'Allergenfrei' (allergen-free) OR specific allergens",
            "- Glutenfrei <20ppm: 'Glutenfrei' OR 'Nicht Glutenfrei'",
            "- Laktosefrei: 'Laktosefrei' OR 'nicht laktosefrei'",
            "",
            "TECHNICAL PARAMETERS:",
            "- pH range: Use MIN-MAX format (e.g., '3.5-4.2')",
            "- Brix range: Use MIN-MAX format (e.g., '25-35')",
            "- BRIX AFM: Measured Brix value",
            "- Wasseraktivität AFM: Water activity value",
            "- Viskositaet 30S 20°C: Viscosity measurement",
            "",
            "=" * 80,
            "OUTPUT FORMAT:",
            "=" * 80,
            "{",
            "  'search_type': 'two_step',",
            "  'text_description': 'Rich searchable description with product type, flavor, application, texture, key attributes',",
            "  'features': [",
            "    {'feature_name': 'Flavour', 'feature_value': 'Matcha'},",
            "    {'feature_name': 'Produktsegment (SD Reporting)', 'feature_value': 'Quark/Topfen'},",
            "    {'feature_name': 'Stärke', 'feature_value': 'Stärke enthalten'},",
            "    {'feature_name': 'Künstliche Farben', 'feature_value': 'keine künstl. Farbe'},",
            "    {'feature_name': 'Natürliche Aromen', 'feature_value': 'Natürliches Aroma'},",
            "    {'feature_name': 'HALAL', 'feature_value': 'suitable HALAL'},",
            "    {'feature_name': 'KOSHER', 'feature_value': 'suitable KOSHER'},",
            "    {'feature_name': 'VEGAN', 'feature_value': 'suitable VEGAN'},",
            "    {'feature_name': 'Flüssig/Stückig', 'feature_value': 'Stückig'},",
            "    {'feature_name': 'Allergene', 'feature_value': 'Allergenfrei'}",
            "  ],",
            "  'reasoning': 'Brief explanation of extraction choices'",
            "}",
            "",
            "=" * 80,
            "EXTRACTION RULES:",
            "=" * 80,
            "1. FLAVOR: Extract ALL distinct flavors mentioned in the brief, separated by commas.",
            "   - If multiple flavor options are listed, include ALL of them (e.g., 'Gyros, Honey BBQ, Lime-Mint').",
            "   - This is CRITICAL for matching against the 600K recipe database.",
            "   - Example: Brief mentions 'Gyros, BBQ, Pumpkin Spice' → Flavour: 'Gyros, Honey BBQ, Pumpkin Spice'",
            "2. STABILIZERS: Use database format values, NOT 'Yes'/'No':",
            "   - Starch allowed → {'feature_name': 'Stärke', 'feature_value': 'Stärke enthalten'}",
            "   - No starch → {'feature_name': 'Stärke', 'feature_value': 'Keine Stärke'}",
            "3. CERTIFICATIONS: Use database format values:",
            "   - Halal required → {'feature_name': 'HALAL', 'feature_value': 'suitable HALAL'}",
            "   - Kosher certified → {'feature_name': 'KOSHER', 'feature_value': 'certified KOSHER'}",
            "   - Vegan → {'feature_name': 'VEGAN', 'feature_value': 'suitable VEGAN'}",
            "4. COLORS: Use German format:",
            "   - No artificial colors → {'feature_name': 'Künstliche Farben', 'feature_value': 'keine künstl. Farbe'}",
            "   - No coloring agent → {'feature_name': 'Farbe', 'feature_value': 'Keine Farbe enthalten'}",
            "5. pH/BRIX: Always use MIN-MAX format with dash (e.g., '3.0-4.1', '25-35')",
            "   - If '<4.1' given, use '3.0-4.1'",
            "   - If '30±5' given, convert to '25-35'",
            "6. PRODUKTSEGMENT (SD Reporting): Extract the EXACT value from the brief:",
            "   - Look for 'Produktsegment (SD Reporting): [VALUE]' in the brief",
            "   - Use the EXACT value shown (e.g., 'Käse', 'Quark/Topfen', 'Joghurt', 'Eiscreme', 'Backwaren', 'Molkerei')",
            "   - DO NOT infer from context - if the brief shows 'Käse', use 'Käse', not 'Molkerei'",
            "   - Produktsegment and Industrie are DIFFERENT fields - do not confuse them",
            "7. TEXTURE: If 'with pieces' or 'chunky' → Flüssig/Stückig: 'Stückig'",
            "   If 'smooth' or 'puree' → Flüssig/Stückig: 'Flüssig'",
            "",
            "IMPORTANT NOTES:",
            "- Extract features ONLY when clearly mentioned or strongly implied",
            "- Use GERMAN feature names when they match database better",
            "- The text_description is used for semantic search - make it descriptive and keyword-rich",
            "- Focus on features that help narrow down from 600K recipes",
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
            guide_lines.append(
                "\nFEATURE NAME SYNONYMS (User term → Database field):")
            # Group by target feature
            reverse_map = {}
            for user_term, db_field in feature_map.items():
                if db_field not in reverse_map:
                    reverse_map[db_field] = []
                reverse_map[db_field].append(user_term)

            # Show important mappings - expanded list with German features
            important_features = [
                'Flavour', 'Farbe', 'Color',
                'Produktsegment (SD Reporting)', 'Application (Fruit filling)',
                'Stärke', 'Starch', 'Pektin', 'Pectin', 'Xanthan',
                'HALAL', 'KOSHER', 'VEGAN',
                'Künstliche Farben', 'Artificial colors',
                'Natürliche Aromen', 'Bio zertifiziert',
                'Allergene', 'Glutenfrei <20ppm',
                'Flüssig/Stückig', 'Süßstoff'
            ]

            for db_field in important_features:
                if db_field in reverse_map:
                    synonyms = sorted(set(reverse_map[db_field]))[:6]
                    guide_lines.append(f"  {db_field}: {', '.join(synonyms)}")

        # Add key database values - these are CRITICAL for matching
        guide_lines.append("\n" + "=" * 60)
        guide_lines.append("DATABASE VALUE EXAMPLES (use these EXACT values):")
        guide_lines.append("=" * 60)
        guide_lines.append("\nSTABILIZERS (German format):")
        guide_lines.append(
            "  Stärke: 'Stärke enthalten' (yes) | 'Keine Stärke' (no)")
        guide_lines.append(
            "  Pektin: 'Pektin enthalten' (yes) | 'Kein Pektin' (no)")
        guide_lines.append(
            "  Xanthan: 'Xanthan enthalten' (yes) | 'Kein Xanthan' (no)")
        guide_lines.append(
            "  Andere Stabilisatoren: 'Keine anderen Stabil enthalten' (none)")

        guide_lines.append("\nCOLORS & AROMA (German format):")
        guide_lines.append(
            "  Künstliche Farben: 'keine künstl. Farbe' (no artificial) | 'mit künstl. Farbe' (yes)")
        guide_lines.append(
            "  Farbe: 'Keine Farbe enthalten' (no color) | color name")
        guide_lines.append(
            "  Natürliche Aromen: 'Natürliches Aroma' (natural) | 'Kein naturidentes Aroma'")

        guide_lines.append("\nCERTIFICATIONS:")
        guide_lines.append(
            "  HALAL: 'suitable HALAL' | 'not suitable HALAL' | 'suitable used in c. HALAL rec.'")
        guide_lines.append(
            "  KOSHER: 'suitable KOSHER' | 'certified KOSHER' | 'Not suitable for kosher'")
        guide_lines.append("  VEGAN: 'suitable VEGAN' | 'not suitable VEGAN'")

        guide_lines.append("\nPRODUCT SEGMENTS (Produktsegment):")
        guide_lines.append("  Yogurt/Skyr: 'Quark/Topfen', 'Joghurt'")
        guide_lines.append("  Ice Cream: 'Eiscreme'")
        guide_lines.append("  Bakery: 'Backwaren'")
        guide_lines.append("  Dairy: 'Molkerei'")

        guide_lines.append("\nTEXTURE (Flüssig/Stückig):")
        guide_lines.append("  With pieces/chunks: 'Stückig'")
        guide_lines.append("  Smooth/liquid: 'Flüssig'")

        guide_lines.append("\nALLERGENS & DIETARY:")
        guide_lines.append(
            "  Allergene: 'Allergenfrei' (allergen-free) | specific allergen")
        guide_lines.append(
            "  Laktosefrei: 'Laktosefrei' | 'nicht laktosefrei'")
        guide_lines.append(
            "  Konservierung: 'Nicht konserviert' (no preservatives)")
        guide_lines.append("  Süßstoff: 'keine Süsstoffe' (no sweeteners)")

        guide_lines.append("\nUSE THESE MAPPINGS TO:")
        guide_lines.append(
            "1. Map user terminology (any language/synonym) to exact database field names")
        guide_lines.append(
            "2. Use EXACT database values shown above - NOT simplified Yes/No")
        guide_lines.append(
            "3. Handle multilingual inputs (English, German, French)")
        guide_lines.append(
            "4. Prioritize German feature names when they match database better")

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
