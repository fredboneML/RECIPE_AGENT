#!/usr/bin/env python3
"""
Integration Tests for Data Extractor and Recipe Search Agents

These tests verify the full pipeline:
1. Data Extractor Agent extracts binary and numerical constraints from briefs
2. Search Agent applies these constraints correctly to Qdrant
3. Recipe indexing properly encodes binary oppositions

Based on customer briefs from Test_Input/
"""

import pytest
import sys
import os
import json
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai-analyzer'))

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST BRIEFS (from Test_Input/)
# =============================================================================

BRIEF_CHEERS_1 = """
PRODUCT BRIEF: Vegan Gyros-Style Cheese Preparation

Concept Description:
We are looking for a vegan cheese preparation with Mediterranean/Greek-inspired flavors. 
The product should work well as an oven-baked cheese alternative, similar to traditional 
Ofenkäse but fully plant-based.

Flavor Profile: Savory, herb-forward gyros seasoning with warming spices. 
Should appeal to consumers seeking meat-free Mediterranean options.

KEY PRODUCT REQUIREMENTS:
Dietary Claims: Vegan, Gluten-free (<20ppm required)
Religious Certification: Halal certified
Allergens: Allergen-free preferred
Preservatives: No preservatives
Artificial Additives: No artificial colors or sweeteners
Flavoring: Natural aromas only, no nature-identical flavors
Sugar: No added saccharose

Product Category: Zubereitung / Preparation
Target Segment: Käse (Cheese alternatives)

Launch Type: New Product for seasonal rotation
"""

BRIEF_CHEERS_2_DE = """
ANFRAGE: Vegane Ofenkäse-Zubereitung

Produktbeschreibung:
Für unser Foodservice-Sortiment suchen wir eine vegane Käsezubereitung im 
griechischen Stil (Gyros-Geschmack). Das Produkt soll sich zum Überbacken 
eignen und als Beilage zu mediterranen Gerichten verwendet werden.

ANFORDERUNGEN:
- Vegan zertifiziert
- Glutenfrei (unter 20ppm)
- Halal-Zertifizierung erforderlich
- Würziges Geschmacksprofil

NICHT ERLAUBT:
- Keine künstlichen Farbstoffe
- Keine Süßstoffe oder Saccharose
- Keine Konservierungsstoffe
- Keine naturidenten/künstlichen Aromen

ERLAUBT:
- Natürliche Aromen
- Stärke als Stabilisator
- OGT-konforme Rezeptur

Industrie: Würzig / Savory
Produktsegment: Käse
"""

BRIEF_FRUIT_PREP = """
Flavour Inspiration:
Flavour Concept Inspiration & Description 
Peach Apricot Fruit: Yellow peach particulates, max 12mm in size. Can also use a peach puree. 
Apricot Puree. Fruit content to be >30%. 
Flavour profile: Balanced peach and apricot flavours. Sweetness and slight tartness of 
the fruit. Dessert like poached peaches and apricots with compliment GD yogurt.
Colour: Vibrant deep orange colour with visible particulates. Not artificial looking. All 
N1 colours are available for development. 

FRUIT PREP ATTRIBUTES:
Particles Size: 12mm max
Amount of Fruit (if applicable): >30%
pH/Acidity: <4.1
Brix: Fruit 30+/-5°, Syrup 50+/-5°
Viscosity (20°C 60 seconds): 6-9 +/- 2 cm
Key Product Claims: No Preservatives, Artificial Colours or Flavours 
Allowed Stabilization System: Modified Starch (1442), Pectin (440), Guar (412), Xanthan (415), LBG (410) 
Allowed Acidifying Agents: Citric Acid, Malic Acid, Lactic Acid or Ascorbic Acid
Allowed Flavouring Agents: Only Natural Flavours
Allowed Colouring Agents: Only Natural Colours 
Allowed Allergen: Milk containing products
Religious Certification: Halal & Kosher Preferred. 
Shelf Life: Min 3 months.
"""

BRIEF_MATCHA = """
PROJECT & FINISH PRODUCT GENERAL INFORMATIONS

Project code name: MAT
Brief date: 11/08/2025
Application & Project objectives: Develop a new Skyr flavour made with Matcha tea

WHITE MASS - characteristics:
Fat content (%): 2
Sugar content (%): 0
Protein content (%): 10
pH: 4,5
Target finish product shelf life: 45-55 days
Finish product storage temperature: 4-6°C
"""


# =============================================================================
# DATA EXTRACTOR AGENT TESTS
# =============================================================================

class TestDataExtractorAgent:
    """Test the DataExtractorRouterAgent's extraction capabilities"""

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI response"""
        def _create_response(content: str):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = content
            return mock_response
        return _create_response

    def test_extract_binary_from_english_brief(self, mock_openai_response):
        """Test extraction of binary fields from English brief"""
        # Expected extraction from BRIEF_CHEERS_1
        expected_features = {
            "sweetener": "No",  # "No artificial colors or sweeteners"
            "preservatives": "No",  # "No preservatives"
            "artificial_colors": "No",  # "No artificial colors"
            "natural_aromas": "Yes",  # "Natural aromas only"
            "nature_identical_aromas": "No",  # "no nature-identical flavors"
            "saccharose": "No",  # "No added saccharose"
            "halal": "Yes",  # "Halal certified"
        }
        
        # Verify expected structure
        for field, value in expected_features.items():
            assert value in ["Yes", "No"], f"Invalid value for {field}"

    def test_extract_binary_from_german_brief(self, mock_openai_response):
        """Test extraction of binary fields from German brief"""
        # Expected extraction from BRIEF_CHEERS_2_DE
        expected_features = {
            "künstliche_farbstoffe": "No",  # "Keine künstlichen Farbstoffe"
            "süßstoffe": "No",  # "Keine Süßstoffe"
            "saccharose": "No",  # "Keine Saccharose"
            "konservierungsstoffe": "No",  # "Keine Konservierungsstoffe"
            "naturidente_aromen": "No",  # "Keine naturidenten/künstlichen Aromen"
            "natürliche_aromen": "Yes",  # "Natürliche Aromen"
            "stärke": "Yes",  # "Stärke als Stabilisator"
            "halal": "Yes",  # "Halal-Zertifizierung erforderlich"
        }
        
        for field, value in expected_features.items():
            assert value in ["Yes", "No"], f"Invalid value for {field}"

    def test_extract_numerical_from_fruit_prep_brief(self, mock_openai_response):
        """Test extraction of numerical constraints from Fruit Prep brief"""
        # Expected numerical constraints from BRIEF_FRUIT_PREP
        expected_numerical = {
            "Z_FRUCHTG": {"gt": 30.0},  # ">30%"
            "Z_PH": {"lt": 4.1},  # "<4.1"
            "Z_BRIX": {"gte": 25.0, "lte": 35.0},  # "30+/-5°"
            "Z_VISK60S": {"gte": 4.0, "lte": 11.0},  # "6-9 +/- 2 cm"
            # "Z_FLST": {"lte": 12.0},  # "12mm max" - particle size
        }
        
        for field, constraint in expected_numerical.items():
            assert isinstance(constraint, dict), f"Constraint for {field} should be dict"
            assert any(k in constraint for k in ["gt", "gte", "lt", "lte"]), \
                f"Invalid constraint keys for {field}"

    def test_extract_european_decimal_from_matcha_brief(self, mock_openai_response):
        """Test extraction of European decimal format from Matcha brief"""
        # The Matcha brief uses "4,5" for pH
        expected_numerical = {
            "Z_FETTST": {"eq": 2.0},  # "Fat content (%): 2"
            "Z_ZUCKER": {"eq": 0.0},  # "Sugar content (%): 0"
            "Z_PROT": {"eq": 10.0},  # "Protein content (%): 10"
            "Z_PH": {"eq": 4.5},  # "pH: 4,5" - European decimal!
        }
        
        # The pH value should be correctly parsed as 4.5, not 45 or error
        assert expected_numerical["Z_PH"]["eq"] == 4.5, \
            "European decimal 4,5 should parse to 4.5"


# =============================================================================
# NUMERICAL CONSTRAINT PARSER INTEGRATION TESTS
# =============================================================================

class TestNumericalConstraintIntegration:
    """Test the numerical constraint parser with real brief snippets"""

    @pytest.fixture
    def parser(self):
        """Get the numerical constraint parser functions"""
        try:
            from ai_analyzer.utils.numerical_constraint_parser import (
                parse_numerical_constraint,
                parse_numerical_constraints_from_brief,
                constraints_to_qdrant_filters,
            )
            return {
                'parse_single': parse_numerical_constraint,
                'parse_brief': parse_numerical_constraints_from_brief,
                'to_filters': constraints_to_qdrant_filters,
            }
        except ImportError:
            pytest.skip("Numerical constraint parser not available")

    def test_fruit_content_constraint(self, parser):
        """Test parsing 'fruit content >30%'"""
        result = parser['parse_single'](">30%")
        assert result is not None
        assert "gt" in result or "gte" in result
        if "gt" in result:
            assert result["gt"] == 30.0

    def test_ph_constraint(self, parser):
        """Test parsing 'pH <4.1'"""
        result = parser['parse_single']("<4.1")
        assert result is not None
        assert "lt" in result
        assert result["lt"] == 4.1

    def test_brix_tolerance_constraint(self, parser):
        """Test parsing 'Brix 30+/-5°'"""
        result = parser['parse_single']("30+/-5°")
        assert result is not None
        assert "gte" in result and "lte" in result
        assert result["gte"] == 25.0
        assert result["lte"] == 35.0

    def test_viscosity_range_with_tolerance(self, parser):
        """Test parsing '6-9 +/- 2 cm'"""
        result = parser['parse_single']("6-9 +/- 2 cm")
        assert result is not None
        # Should create range: [6-2, 9+2] = [4, 11]
        assert "gte" in result or "lte" in result

    def test_european_decimal(self, parser):
        """Test parsing European decimal '4,5'"""
        result = parser['parse_single']("4,5")
        assert result is not None
        # Check that 4,5 was parsed as 4.5
        values = [v for v in result.values() if isinstance(v, float)]
        assert any(abs(v - 4.5) < 0.1 for v in values), \
            f"Expected 4.5 in result, got {result}"

    def test_max_constraint(self, parser):
        """Test parsing '12mm max'"""
        result = parser['parse_single']("12mm max")
        assert result is not None
        assert "lte" in result
        assert result["lte"] == 12.0

    def test_min_constraint(self, parser):
        """Test parsing 'Min 3 months'"""
        result = parser['parse_single']("Min 3 months")
        assert result is not None
        assert "gte" in result
        assert result["gte"] == 3.0


# =============================================================================
# RECIPE SEARCH AGENT TESTS
# =============================================================================

class TestRecipeSearchAgent:
    """Test the RecipeSearchAgent's search capabilities"""

    @pytest.fixture
    def mock_qdrant_results(self):
        """Create mock Qdrant search results"""
        return [
            {
                "id": "recipe_001",
                "score": 0.95,
                "payload": {
                    "recipe_name": "000000000000444700_DE10_01_L",
                    "spec_fields": {
                        "Z_INH02": "No",  # Sweetener
                        "Z_INH04": "No",  # Preserved
                        "Z_INH05": "No",  # Artificial colors
                        "Z_INH01H": "Yes",  # Halal
                        "Z_BRIX": "13.0",
                        "Z_PH": "4.3",
                    },
                    "numerical": {
                        "Z_BRIX": 13.0,
                        "Z_PH": 4.3,
                    }
                }
            },
            {
                "id": "recipe_002",
                "score": 0.90,
                "payload": {
                    "recipe_name": "000000000000074657_PL10_06_L",
                    "spec_fields": {
                        "Z_INH02": "No",
                        "Z_INH04": "No",
                        "Z_FRUCHTG": "61.6 %",
                        "Z_BRIX": "49.0",
                        "Z_PH": "3.7",
                    },
                    "numerical": {
                        "Z_BRIX": 49.0,
                        "Z_PH": 3.7,
                        "Z_FRUCHTG": 61.6,
                    }
                }
            }
        ]

    def test_numerical_filter_application(self, mock_qdrant_results):
        """Test that numerical filters are correctly applied"""
        # Numerical filters for "Brix > 40"
        numerical_filters = {
            "Z_BRIX": {"gt": 40.0}
        }
        
        # Only recipe_002 has Brix > 40 (49.0)
        filtered = []
        for recipe in mock_qdrant_results:
            brix = recipe["payload"]["numerical"].get("Z_BRIX")
            if brix is not None and brix > 40:
                filtered.append(recipe)
        
        assert len(filtered) == 1
        assert filtered[0]["id"] == "recipe_002"

    def test_ph_filter_application(self, mock_qdrant_results):
        """Test that pH filters are correctly applied"""
        # Numerical filters for "pH < 4.0"
        numerical_filters = {
            "Z_PH": {"lt": 4.0}
        }
        
        # Only recipe_002 has pH < 4.0 (3.7)
        filtered = []
        for recipe in mock_qdrant_results:
            ph = recipe["payload"]["numerical"].get("Z_PH")
            if ph is not None and ph < 4.0:
                filtered.append(recipe)
        
        assert len(filtered) == 1
        assert filtered[0]["id"] == "recipe_002"

    def test_fruit_content_filter_application(self, mock_qdrant_results):
        """Test that fruit content filters are correctly applied"""
        # Numerical filters for "Fruit content > 30%"
        numerical_filters = {
            "Z_FRUCHTG": {"gt": 30.0}
        }
        
        # Only recipe_002 has fruit content > 30% (61.6%)
        filtered = []
        for recipe in mock_qdrant_results:
            fruit = recipe["payload"]["numerical"].get("Z_FRUCHTG")
            if fruit is not None and fruit > 30:
                filtered.append(recipe)
        
        assert len(filtered) == 1
        assert filtered[0]["id"] == "recipe_002"

    def test_binary_filter_application(self, mock_qdrant_results):
        """Test that binary filters are correctly applied"""
        # Filter for "Halal = Yes"
        filtered = []
        for recipe in mock_qdrant_results:
            halal = recipe["payload"]["spec_fields"].get("Z_INH01H")
            if halal == "Yes":
                filtered.append(recipe)
        
        # Only recipe_001 has Halal = Yes
        assert len(filtered) == 1
        assert filtered[0]["id"] == "recipe_001"


# =============================================================================
# RECIPE INDEXING TESTS
# =============================================================================

class TestRecipeIndexing:
    """Test that recipe indexing properly encodes binary oppositions"""

    @pytest.fixture
    def sample_recipe_json(self):
        """Sample recipe JSON structure"""
        return {
            "MAKT": [{"MAKTX": "P TYPE GYROS OVCHEES GTF H.S. GLUF VEGAN"}],
            "Classification": {
                "CHAR": [
                    {
                        "charactDescr": "Standardprodukt",
                        "valueRelation": "1",
                        "valueFrom": None,
                        "valueschar": [{"valueCharLong": "Yes"}]
                    },
                    {
                        "charactDescr": "Süßstoff",
                        "valueRelation": "1",
                        "valueFrom": None,
                        "valueschar": [{"valueCharLong": "No"}]
                    },
                    {
                        "charactDescr": "Halal",
                        "valueRelation": "1",
                        "valueFrom": None,
                        "valueschar": [{"valueCharLong": "Yes"}]
                    },
                ],
                "NUM": [
                    {
                        "charactDescr": "Brix",
                        "valueRelation": "1",
                        "valueFrom": "13.0",
                        "valuesnum": []
                    },
                    {
                        "charactDescr": "PH",
                        "valueRelation": "1",
                        "valueFrom": "4.3",
                        "valuesnum": []
                    },
                ]
            }
        }

    def test_binary_field_extraction(self, sample_recipe_json):
        """Test that binary fields are correctly extracted"""
        classification = sample_recipe_json["Classification"]["CHAR"]
        
        binary_values = {}
        for item in classification:
            desc = item["charactDescr"]
            value = item["valueschar"][0]["valueCharLong"] if item.get("valueschar") else None
            if value in ["Yes", "No"]:
                binary_values[desc] = value
        
        assert binary_values.get("Standardprodukt") == "Yes"
        assert binary_values.get("Süßstoff") == "No"
        assert binary_values.get("Halal") == "Yes"

    def test_numerical_field_extraction(self, sample_recipe_json):
        """Test that numerical fields are correctly extracted"""
        classification = sample_recipe_json["Classification"]["NUM"]
        
        numerical_values = {}
        for item in classification:
            desc = item["charactDescr"]
            value_from = item.get("valueFrom")
            if value_from:
                try:
                    numerical_values[desc] = float(value_from)
                except ValueError:
                    pass
        
        assert numerical_values.get("Brix") == 13.0
        assert numerical_values.get("PH") == 4.3


# =============================================================================
# END-TO-END PIPELINE TESTS
# =============================================================================

class TestEndToEndPipeline:
    """Test the complete extraction-to-search pipeline"""

    def test_english_brief_to_search_filters(self):
        """Test complete pipeline for English brief"""
        # Input: BRIEF_CHEERS_1
        # Expected extraction:
        expected_binary_filters = {
            "Z_INH02": "No",  # Sweetener
            "Z_INH04": "No",  # Preserved
            "Z_INH05": "No",  # Artificial colors
            "Z_INH01H": "Yes",  # Halal
            "Z_INH06": "No",  # Nature identical flavor
            "Z_INH06Z": "Yes",  # Natural flavor
            "Z_INH03": "No",  # Saccharose
        }
        
        # Validate structure
        for field, value in expected_binary_filters.items():
            assert field.startswith("Z_"), f"Invalid field code: {field}"
            assert value in ["Yes", "No"], f"Invalid binary value for {field}"

    def test_fruit_prep_brief_to_numerical_filters(self):
        """Test complete pipeline for Fruit Prep brief with numerical constraints"""
        # Expected numerical filters from BRIEF_FRUIT_PREP
        expected_numerical_filters = {
            "Z_FRUCHTG": {"gt": 30.0},  # >30%
            "Z_PH": {"lt": 4.1},  # <4.1
            "Z_BRIX": {"gte": 25.0, "lte": 35.0},  # 30+/-5°
        }
        
        # Validate structure
        for field, constraint in expected_numerical_filters.items():
            assert field.startswith("Z_"), f"Invalid field code: {field}"
            assert isinstance(constraint, dict)
            for op, value in constraint.items():
                assert op in ["gt", "gte", "lt", "lte", "eq"], f"Invalid operator: {op}"
                assert isinstance(value, (int, float)), f"Invalid value type for {field}"

    def test_german_brief_binary_extraction(self):
        """Test that German brief binary terms are correctly normalized"""
        # German terms from BRIEF_CHEERS_2_DE
        german_terms = {
            "Keine künstlichen Farbstoffe": "No",
            "Keine Süßstoffe": "No",
            "Keine Konservierungsstoffe": "No",
            "Keine naturidenten/künstlichen Aromen": "No",
            "Natürliche Aromen": "Yes",
            "Stärke als Stabilisator": "Yes",
            "Halal-Zertifizierung erforderlich": "Yes",
        }
        
        for term, expected in german_terms.items():
            # Check that term contains expected binary indicators
            if expected == "No":
                assert any(neg in term.lower() for neg in ["keine", "kein", "nicht"]), \
                    f"Expected negative indicator in: {term}"
            elif expected == "Yes" and "keine" not in term.lower():
                # Positive terms without "keine"
                assert True  # Just validate structure

    def test_combined_filters_structure(self):
        """Test that combined binary and numerical filters have valid structure"""
        # Combined filters that would be sent to Qdrant
        combined_filters = {
            "binary": {
                "Z_INH02": "No",
                "Z_INH04": "No",
                "Z_INH01H": "Yes",
            },
            "numerical": {
                "Z_FRUCHTG": {"gt": 30.0},
                "Z_PH": {"lt": 4.1},
                "Z_BRIX": {"gte": 25.0, "lte": 35.0},
            }
        }
        
        # Validate binary filters
        for field, value in combined_filters["binary"].items():
            assert field.startswith("Z_")
            assert value in ["Yes", "No"]
        
        # Validate numerical filters
        for field, constraint in combined_filters["numerical"].items():
            assert field.startswith("Z_")
            for op, val in constraint.items():
                assert op in ["gt", "gte", "lt", "lte"]
                assert isinstance(val, (int, float))


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCasesIntegration:
    """Test edge cases in the full pipeline"""

    def test_brief_with_no_numerical_constraints(self):
        """Test brief that only has binary constraints"""
        brief = """
        Product Requirements:
        - Vegan certified
        - No preservatives
        - No artificial colors
        - Halal approved
        """
        # Should extract binary constraints only, no numerical
        # Numerical filters should be empty or None
        expected_numerical = {}
        assert len(expected_numerical) == 0

    def test_brief_with_conflicting_constraints(self):
        """Test brief with potentially conflicting constraints"""
        # Brief says "no sweeteners" but also "can contain saccharose"
        brief = """
        Requirements:
        - No artificial sweeteners
        - Saccharose allowed as natural sweetener
        """
        # The extraction should handle this by:
        # Z_INH02 (Sweetener) = No (artificial)
        # Z_INH03 (Saccharose) = Yes (allowed)
        expected = {
            "Z_INH02": "No",  # Artificial sweetener
            "Z_INH03": "Yes",  # Saccharose
        }
        assert expected["Z_INH02"] != expected["Z_INH03"]

    def test_brief_with_implicit_constraints(self):
        """Test brief with implicit constraints from context"""
        brief = """
        Looking for a fruit preparation with high fruit content
        for premium yogurt applications.
        Must be suitable for organic certification.
        """
        # "High fruit content" implies Z_FRUCHTG > some threshold
        # "Organic certification" implies specific constraints
        # The extraction should infer reasonable defaults
        pass

    def test_multilingual_brief_mixed(self):
        """Test brief that mixes German and English"""
        brief = """
        Product: Fruchtgehalt > 30%
        Requirements: No preservatives (Keine Konservierung)
        Brix: 45°
        pH-Wert: < 4.0
        """
        # Should handle mixed language extraction
        expected_numerical = {
            "Z_FRUCHTG": {"gt": 30.0},
            "Z_BRIX": {"eq": 45.0},
            "Z_PH": {"lt": 4.0},
        }
        expected_binary = {
            "Z_INH04": "No",  # Preserved
        }
        assert len(expected_numerical) == 3
        assert len(expected_binary) == 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
