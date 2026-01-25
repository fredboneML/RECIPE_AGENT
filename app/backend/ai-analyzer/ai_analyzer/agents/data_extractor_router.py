#!/usr/bin/env python3
"""
Data Extractor & Search Router Agent

This agent is responsible for:
1. Extracting meaningful information from supplier project briefs
2. Preparing data for two-step feature-based search
3. Structuring the search query with features and text description
4. Using intelligent feature mappings for multilingual/synonym support
5. Extracting numerical constraints (Brix >40, pH <4.1, etc.) for range filtering
"""
import logging
import re
import json
import os
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from openai import OpenAI
from ai_analyzer.config import config

# Import numerical constraint parser for range queries
try:
    from ai_analyzer.utils.numerical_constraint_parser import (
        parse_numerical_constraints_from_brief,
        constraints_to_qdrant_filters,
        NumericalConstraint,
        FIELD_CODE_INFO
    )
    NUMERICAL_PARSER_AVAILABLE = True
except ImportError:
    NUMERICAL_PARSER_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractorRouterAgent:
    """Agent for extracting data from supplier briefs for two-step feature-based search"""

    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """Initialize the Data Extractor & Router Agent
        
        Default model is gpt-4o-mini which:
        - Supports temperature=0.0 for deterministic output
        - Has strong JSON instruction following
        - Is fast and cost-effective
        - Supports structured outputs (though we don't use it for compatibility)
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key or config.get('AI_ANALYZER_OPENAI_API_KEY')

        # Load feature mappings for intelligent extraction
        self.feature_mappings = self._load_feature_mappings()

        # Generate dynamic instructions with mappings
        self.instructions = self._generate_instructions()

        # Create OpenAI client directly for deterministic extraction
        # Using direct API to ensure temperature=0.0 is properly respected
        self.client = OpenAI(api_key=self.api_key)
        
        # Store system prompt from instructions
        self.system_prompt = "\n".join(self.instructions)

        logger.info(
            f"DataExtractorRouterAgent initialized with {len(self.feature_mappings.get('feature_name_mappings', {}))} feature name mappings")
        logger.info(f"Using OpenAI API directly with temperature=0.0 for deterministic extraction")

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
            "CRITICAL: OUTPUT MUST BE VALID JSON",
            "=" * 80,
            "Your ENTIRE response must be a single, valid JSON object with double quotes.",
            "Do NOT include any text before or after the JSON object.",
            "Do NOT use single quotes - use double quotes only.",
            "Do NOT include explanatory text outside the JSON structure.",
            "",
            "=" * 80,
            "TEXT DESCRIPTION (CRITICAL FOR SEARCH):",
            "=" * 80,
            "⚠️  CRITICAL: DO NOT HALLUCINATE OR ADD INFORMATION NOT PRESENT IN THE BRIEF!",
            "⚠️  ONLY include details that are EXPLICITLY mentioned or clearly implied in the input.",
            "⚠️  For short queries (just a recipe name), keep description SHORT and factual.",
            "",
            "Guidelines for creating text_description:",
            "- If brief is DETAILED (multiple paragraphs): Create a rich 3-5 sentence description",
            "- If brief is SHORT (just a name/flavor): Create a SHORT 1-sentence description using ONLY what's given",
            "- NEVER invent details like 'rich in flavor', 'smooth texture', 'no artificial colors' unless EXPLICITLY stated",
            "- Include ONLY information from the brief:",
            "  * Product type: ONLY if mentioned (fruit preparation, fruit filling, compound, puree, etc.)",
            "  * Application: ONLY if mentioned (yogurt, ice cream, bakery, beverage, dairy, quark/skyr, cheese)",
            "  * Main flavor(s): ALWAYS extract (this is usually present)",
            "  * Texture: ONLY if explicitly stated (with pieces, smooth, chunky, liquid)",
            "  * Key attributes: ONLY if explicitly stated (organic, natural, no artificial colors, low sugar, etc.)",
            "",
            "EXAMPLE TEXT DESCRIPTIONS:",
            "- DETAILED BRIEF → 'Matcha tea fruit preparation for skyr and quark application, natural flavor, starch stabilized, no artificial colors, with pieces'",
            "- SHORT QUERY ('Mango Chutney für Ofenkäse') → 'Mango chutney for baked cheese'  ← KEEP IT SHORT!",
            "- SHORT QUERY ('Strawberry organic') → 'Strawberry, organic certified'  ← DO NOT ADD texture, colors, etc.",
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
            "  * If brief says 'Without if possible' or 'No flavoring' → 'Kein naturidentes Aroma'",
            "- Süßstoff: 'keine Süsstoffe' (no sweeteners) OR sweetener type",
            "- Saccharose: Sucrose - 'Saccharose' OR 'keine Saccharose'",
            "  * If brief specifies sugar percentage constraint (e.g., 'Max 5%'), still extract 'Saccharose'",
            "  * Note: Percentage constraints are captured in text_description for search context",
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
            "OUTPUT FORMAT (VALID JSON - USE DOUBLE QUOTES, NOT SINGLE QUOTES):",
            "=" * 80,
            "{",
            '  "search_type": "two_step",',
            '  "text_description": "Rich searchable description with product type, flavor, application, texture, key attributes",',
            '  "features": [',
            '    {"feature_name": "Flavour", "feature_value": "Matcha"},',
            '    {"feature_name": "Produktsegment (SD Reporting)", "feature_value": "Quark/Topfen"},',
            '    {"feature_name": "Stärke", "feature_value": "Stärke enthalten"},',
            '    {"feature_name": "Künstliche Farben", "feature_value": "keine künstl. Farbe"},',
            '    {"feature_name": "Natürliche Aromen", "feature_value": "Natürliches Aroma"},',
            '    {"feature_name": "HALAL", "feature_value": "suitable HALAL"},',
            '    {"feature_name": "KOSHER", "feature_value": "suitable KOSHER"},',
            '    {"feature_name": "VEGAN", "feature_value": "suitable VEGAN"},',
            '    {"feature_name": "Flüssig/Stückig", "feature_value": "Stückig"},',
            '    {"feature_name": "Allergene", "feature_value": "Allergenfrei"}',
            "  ],",
            '  "numerical_constraints": [',
            '    {"field": "Brix", "constraint": ">40"},',
            '    {"field": "pH", "constraint": "<4.1"},',
            '    {"field": "Fruit content", "constraint": ">30%"},',
            '    {"field": "Fruit in finished product", "constraint": "max 5%"},',
            '    {"field": "Sugar in finished product", "constraint": "max 5%"},',
            '    {"field": "Viscosity", "constraint": "6-9"},',
            '    {"field": "Fat content", "constraint": "58±2%"}',
            "  ],",
            '  "reasoning": "Brief explanation of extraction choices"',
            "}",
            "",
            "CRITICAL: Your response MUST be valid JSON with double quotes, not Python dict syntax with single quotes.",
            "",
            "=" * 80,
            "EXTRACTION RULES:",
            "=" * 80,
            "0. ⚠️  ANTI-HALLUCINATION RULE (MOST IMPORTANT!):",
            "   - DO NOT invent, assume, or infer ANY information not explicitly in the brief",
            "   - DO NOT add common-sense defaults ('smooth texture', 'no artificial colors', etc.)",
            "   - ONLY extract features that are CLEARLY STATED in the input text",
            "   - For short queries (< 20 words), extract MINIMAL features and keep description SHORT",
            "   - If a feature is not mentioned, DO NOT extract it - leave it out entirely",
            "   - Example: 'Mango Chutney für Ofenkäse' → ONLY extract Flavour and Produktsegment, nothing else!",
            "",
            "1. FLAVOR: Extract ALL distinct flavors mentioned in the brief, separated by commas.",
            "   - If multiple flavor options are listed, include ALL of them (e.g., 'Gyros, Honey BBQ, Lime-Mint').",
            "   - This is CRITICAL for matching against the 600K recipe database.",
            "   - Example: Brief mentions 'Gyros, BBQ, Pumpkin Spice' → Flavour: 'Gyros, Honey BBQ, Pumpkin Spice'",
            "2. STABILIZERS: Use database format values, NOT 'Yes'/'No':",
            '   - Starch allowed → {"feature_name": "Stärke", "feature_value": "Stärke enthalten"}',
            '   - No starch → {"feature_name": "Stärke", "feature_value": "Keine Stärke"}',
            "3. CERTIFICATIONS: Use database format values:",
            '   - Halal required → {"feature_name": "HALAL", "feature_value": "suitable HALAL"}',
            '   - Kosher certified → {"feature_name": "KOSHER", "feature_value": "certified KOSHER"}',
            '   - Vegan → {"feature_name": "VEGAN", "feature_value": "suitable VEGAN"}',
            "4. COLORS: Use German format:",
            '   - No artificial colors → {"feature_name": "Künstliche Farben", "feature_value": "keine künstl. Farbe"}',
            '   - No coloring agent → {"feature_name": "Farbe", "feature_value": "Keine Farbe enthalten"}',
            "5. pH/BRIX/NUMERICAL VALUES: Report EXACT constraints found in brief - DO NOT normalize to ranges!",
            "   - For '<4.1', report as '<4.1' (we will parse this separately)",
            "   - For '>30%', report as '>30%'",
            "   - For '30±5' or '30+/-5', report as '30±5'",
            "   - For '6-9', report as '6-9'",
            "   - For 'max 12mm', report as 'max 12'",
            "   - For 'min 3 months', report as 'min 3'",
            "   - CRITICAL: We need the ORIGINAL constraint format for Qdrant range filtering!",
            "6. PRODUKTSEGMENT (SD Reporting): Extract the EXACT value from the brief:",
            "   - Look for 'Produktsegment (SD Reporting): [VALUE]' in the brief",
            "   - Use the EXACT value shown (e.g., 'Käse', 'Quark/Topfen', 'Joghurt', 'Eiscreme', 'Backwaren', 'Molkerei')",
            "   - If brief mentions 'Skyr', map to 'Quark/Topfen' or 'Joghurt' (Skyr is similar to these)",
            "   - DO NOT infer from context - if the brief shows 'Käse', use 'Käse', not 'Molkerei'",
            "   - Produktsegment and Industrie are DIFFERENT fields - do not confuse them",
            "7. TEXTURE: If 'with pieces' or 'chunky' → Flüssig/Stückig: 'Stückig'",
            "   If 'smooth' or 'puree' → Flüssig/Stückig: 'Flüssig'",
            "",
            "7. SUGAR & FRUIT CONTENT CONSTRAINTS:",
            "   - If brief specifies 'Sugar in finished product (%) Max X%', extract as numerical_constraint: {\"field\": \"Sugar content\", \"constraint\": \"max X%\"}",
            "   - If brief specifies 'Fruit in finished product (%) X%' or 'Fruit in finished product Max X%', extract as numerical_constraint: {\"field\": \"Fruit content\", \"constraint\": \"max X%\"}",
            "   - CRITICAL: In form-style briefs, BOTH 'Fruit in finished product' AND 'Sugar in finished product' often have separate % values - extract BOTH!",
            "   - ⚠️  NUMERICAL CONSTRAINTS MUST CONTAIN ACTUAL NUMBERS! Valid examples: 'max 5%', '>30%', '5%', '5-10%'",
            "   - ⚠️  NEVER put descriptive text like 'any fruit possible' as a constraint value - ONLY numbers!",
            "   - If you see 'Fruit in finished product (%) 5%' in the same row/line, the constraint is '5%' or 'max 5%'",
            "   - 📋 FORM TABLE HEURISTIC: In form-style documents, 'Fruit in finished product (%)' and 'Sugar in finished product (%)' are usually adjacent rows with similar percentage limits.",
            "   - If OCR merged the rows (e.g., 'Fruit in finished product (%) Sugar in finished product (%) Max 5%'), extract BOTH with the same max value:",
            "     → {\"field\": \"Fruit content\", \"constraint\": \"max 5%\"} AND {\"field\": \"Sugar content\", \"constraint\": \"max 5%\"}",
            "   - Extract 'Saccharose: Saccharose' if sucrose is allowed, regardless of percentage constraint",
            "   - If 'No added sugar' is checked, ensure this is mentioned in text_description",
            "8. FLAVOURING PREFERENCES:",
            "   - If brief says 'Without if possible' or 'No flavoring' → 'Natürliche Aromen: Kein naturidentes Aroma'",
            "   - If brief says 'Natural flavor' or 'With natural flavor' → 'Natürliche Aromen: Natürliches Aroma'",
            "",
            "IMPORTANT NOTES:",
            "- Extract features ONLY when clearly mentioned or strongly implied",
            "- Use GERMAN feature names when they match database better",
            "- The text_description is used for semantic search - make it descriptive and keyword-rich",
            "- Include percentage constraints and 'no added sugar' requirements in text_description even if not extractable as features",
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

            # Detect if this is a short query (likely just a recipe name)
            is_short_query = len(combined_brief.strip()) < 50 and len(combined_brief.split()) < 10
            
            if is_short_query:
                logger.info(f"Detected SHORT QUERY (length={len(combined_brief)}, words={len(combined_brief.split())})")
                short_query_instruction = """
⚠️  SPECIAL MODE: SHORT QUERY DETECTED
This is likely just a recipe name or simple search term.
DO NOT elaborate or add details!
- text_description: Keep it SHORT - just translate/normalize the input (1 sentence maximum)
- features: Extract ONLY what's obvious from the name (usually just Flavour and maybe Produktsegment)
- DO NOT infer texture, colors, stabilizers, or other attributes
Example: "Mango Chutney für Ofenkäse" → text_description: "Mango chutney for baked cheese"
"""
            else:
                short_query_instruction = ""

            # Prepare the prompt
            prompt = f"""
Analyze the following supplier project brief and extract relevant recipe information:

SUPPLIER BRIEF:
{combined_brief}
{short_query_instruction}
Extract the key information and decide on the appropriate search strategy.
Provide your response as a JSON object following the specified format.
"""

            # Get response from OpenAI API directly with temperature=0.0 + seed for full determinism
            logger.info("Extracting information from supplier brief...")
            logger.info(f"Using model: {self.model_name} with temperature=0.0 and seed=42 for fully deterministic extraction")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic extraction
                seed=42,  # Fixed seed for reproducible results (required for full determinism)
                max_tokens=4000
                # Note: response_format json_object only works with gpt-4o, gpt-4-turbo, gpt-3.5-turbo-0125+
                # We rely on explicit prompt instructions for JSON format instead
            )

            # Parse the response
            response_content = response.choices[0].message.content
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
            
            # ============================================================
            # NUMERICAL CONSTRAINTS: Parse and convert to Qdrant filters
            # ============================================================
            numerical_filters = {}
            
            if NUMERICAL_PARSER_AVAILABLE:
                # Method 1: Parse from raw brief text (catches structured tables)
                try:
                    text_constraints = parse_numerical_constraints_from_brief(combined_brief)
                    if text_constraints:
                        numerical_filters.update(constraints_to_qdrant_filters(text_constraints))
                        logger.info(f"Extracted {len(text_constraints)} numerical constraints from brief text")
                except Exception as e:
                    logger.warning(f"Error parsing numerical constraints from text: {e}")
                
                # Method 2: Parse from LLM-extracted numerical_constraints
                llm_constraints = result.get('numerical_constraints', [])
                if llm_constraints:
                    try:
                        parsed_llm = self._parse_llm_numerical_constraints(llm_constraints)
                        numerical_filters.update(parsed_llm)
                        logger.info(f"Parsed {len(parsed_llm)} numerical constraints from LLM output")
                    except Exception as e:
                        logger.warning(f"Error parsing LLM numerical constraints: {e}")
            
            result['numerical_filters'] = numerical_filters

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

            # Log numerical constraints
            logger.info("NUMERICAL CONSTRAINTS (for Qdrant range filtering):")
            logger.info("-" * 40)
            if result.get('numerical_filters'):
                for field_code, qdrant_filter in result['numerical_filters'].items():
                    logger.info(f"  {field_code}: {qdrant_filter}")
            else:
                logger.info("  No numerical constraints extracted")
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
                'numerical_filters': {},
                'reasoning': f'Error in extraction: {str(e)}. Falling back to two-step search with text only.'
            }

    def _parse_llm_numerical_constraints(self, llm_constraints: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
        """
        Parse numerical constraints from LLM output format to Qdrant filter format.
        
        LLM output format:
            [{"field": "Brix", "constraint": ">40"}, ...]
        
        Returns:
            Dict mapping field_code to Qdrant range filter
            {"Z_BRIX": {"gt": 40}, ...}
        """
        if not NUMERICAL_PARSER_AVAILABLE:
            return {}
        
        from ai_analyzer.utils.numerical_constraint_parser import (
            parse_constraint_text,
            BRIEF_FIELD_TO_CODE,
            NumericalConstraint,
            FIELD_CODE_INFO
        )
        
        filters = {}
        
        for item in llm_constraints:
            field_name = item.get('field', '').lower().strip()
            constraint_text = item.get('constraint', '')
            
            if not field_name or not constraint_text:
                continue
            
            # Early validation: constraint must contain at least one digit
            # This catches LLM hallucinations like "any fruit possible except those already included"
            if not any(char.isdigit() for char in constraint_text):
                logger.warning(
                    f"Skipping non-numerical constraint for '{field_name}': '{constraint_text}' "
                    f"(constraint must contain numbers like '5%', 'max 10', '>30')"
                )
                continue
            
            # Map field name to Z_* code
            field_code = BRIEF_FIELD_TO_CODE.get(field_name)
            
            # Try partial matching if exact match not found
            if not field_code:
                for brief_name, code in BRIEF_FIELD_TO_CODE.items():
                    if brief_name in field_name or field_name in brief_name:
                        field_code = code
                        break
            
            if not field_code:
                # Common unmapped fields with explanations
                unmapped_explanations = {
                    'shelf life': 'Shelf life is not in the 60 specified fields schema - use text_description for semantic search',
                    'mindesthaltbarkeit': 'Shelf life (Mindesthaltbarkeit) is not in the 60 specified fields schema',
                    'haltbarkeit': 'Shelf life (Haltbarkeit) is not in the 60 specified fields schema',
                    'best before': 'Shelf life is not in the 60 specified fields schema',
                    'expiry': 'Expiry/shelf life is not in the 60 specified fields schema',
                }
                explanation = unmapped_explanations.get(field_name, None)
                if explanation:
                    logger.info(f"Field '{field_name}' cannot be used for numerical filtering: {explanation}")
                else:
                    logger.warning(f"Could not map field '{field_name}' to Z_* code")
                continue
            
            # Parse the constraint
            operator, val1, val2 = parse_constraint_text(constraint_text)
            
            if operator == 'unknown':
                logger.warning(f"Could not parse constraint '{constraint_text}' for field '{field_name}'")
                continue
            
            # Build Qdrant filter
            if operator == 'gt':
                filters[field_code] = {"gt": val1}
            elif operator == 'gte':
                filters[field_code] = {"gte": val1}
            elif operator == 'lt':
                filters[field_code] = {"lt": val1}
            elif operator == 'lte':
                filters[field_code] = {"lte": val1}
            elif operator == 'range':
                filters[field_code] = {"gte": val1, "lte": val2}
            elif operator == 'eq':
                # For exact match, use small range
                filters[field_code] = {"gte": val1 - 0.01, "lte": val1 + 0.01}
            
            logger.info(f"Parsed constraint: {field_name} ({field_code}) {constraint_text} → {filters[field_code]}")
        
        return filters

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
            if 'numerical_constraints' not in result:
                result['numerical_constraints'] = []
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
                'numerical_constraints': [],
                'reasoning': 'Failed to parse structured response. Using two-step search with text only.'
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "agent_name": "DataExtractorRouterAgent",
            "model_provider": self.model_provider,
            "model_name": self.model_name
        }
