from src.two_step_recipe_search import EnhancedTwoStepRecipeManager, run_two_step_search
import pandas as pd
import json
import os
import re


def read_recipe_json(recipe_json):
    """Read recipe JSON file and extract features/values."""
    result = None
    try:
        with open(os.path.join(data_dir, recipe_json), 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(os.path.join(data_dir, recipe_json)[-50:])

        if not data:
            print(f" The JSON file {recipe_json} is empty")
        elif 'Classification' in data.keys() and data['Classification'] and 'valueschar' in data['Classification'] and data['Classification']['valueschar']:
            result = pd.DataFrame(data['Classification']['valueschar'])
        else:
            print(f"Classification/valueschar not found in {recipe_json}")
    except Exception as e:
        print(f"Error reading {recipe_json}: {e}")

    return result


def clean_tdline(tdline: str) -> str:
    """
    If 'Colour', 'Flavour', or 'Stabilizer' exist in tdline,
    keep it as is. Otherwise, return an empty string.
    """
    keywords = ["Colour", "Flavour", "Stabilizer"]

    if any(word in tdline for word in keywords):
        return tdline
    else:
        return ""


def extract_recipe_name(filename: str) -> str:
    """
    Extracts the recipe name from a given filename.
    Logic:
    - Split the filename by "_"
    - Take everything starting from the 3rd element (index 2)
    - Remove ".json" if present
    - Check if the remaining string has at least 3 alphabetic letters:
        * Replace "_" with spaces
        * Return as "Recipe Name: <string>"
      Otherwise:
        * Return "Recipe Name: "
    """

    # Split the filename by "_"
    parts = filename.split("_")

    # Take everything after the second "_" (index 2 onward)
    if len(parts) > 2:
        rest = "_".join(parts[2:])
    else:
        rest = ""

    # Remove ".json" extension if present
    rest = rest.replace(".json", "")

    # Count alphabetic letters
    alpha_count = sum(1 for c in rest if c.isalpha())

    # Check if at least 3 alphabetic letters are present
    if alpha_count >= 3:
        # Replace underscores with spaces
        rest = rest.replace("_", " ")
        return f"Recipe Name: {rest.strip()}"
    else:
        return "Recipe Name: "


def extract_recipe_description(recipe_json):
    """
    Extract comprehensive recipe description based on our data structure.
    Combines recipe name, MaterialMasterShorttext, and processed text data.

    Example output for '0000017883_000000000000286789_521082_FIT BANANA.json':
    "Recipe Name: 521082 FIT BANANA, MaterialMasterShorttext: DELFP CHOCOLATE FLIP FIT CHO, Colour Yellow, Flavour Banana, Stabilizer StarchLBGPectin, SMEAR 3M TPC"
    """
    description_parts = []

    # Dictionary with Prefix → replacement
    replacements = {
        r"(?i)\bfl_": "Flavour ",
        r"(?i)\bco_": "Colour ",
        r"(?i)\bstab_": "Stabilizer "
    }

    try:
        # Extract recipe name from filename
        recipe_name = extract_recipe_name(recipe_json)
        if recipe_name.strip() != "Recipe Name:":  # Only add if we found a valid name
            description_parts.append(recipe_name)

        # Load and process JSON data
        with open(os.path.join(data_dir, recipe_json), 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract MaterialMasterShorttext
        if 'MaterialMasterShorttext' in data.keys() and data['MaterialMasterShorttext']:
            try:
                material_master_short_text = data['MaterialMasterShorttext'][0]['maktx']
                if material_master_short_text and material_master_short_text.strip():
                    description_parts.append(
                        f"MaterialMasterShorttext: {material_master_short_text}")
            except (IndexError, KeyError, TypeError) as e:
                print(
                    f"Error extracting MaterialMasterShorttext from {recipe_json}: {e}")

        # Extract and process Texts data
        if 'Texts' in data.keys() and data['Texts']:
            try:
                texts_df = pd.DataFrame(data['Texts'])
                if 'lines' in texts_df.columns and not texts_df.empty:
                    # Get the tdline from the nested structure
                    lines_data = texts_df['lines'].iloc[0]
                    if lines_data and len(lines_data) > 0 and 'tdline' in lines_data[0]:
                        tdline = lines_data[0]['tdline']

                        if tdline and tdline.strip():
                            # Apply prefix replacements
                            for pattern, repl in replacements.items():
                                tdline = re.sub(pattern, repl, tdline)

                            # Clean the tdline (keep only if contains keywords)
                            cleaned_tdline = clean_tdline(tdline)
                            if cleaned_tdline:
                                description_parts.append(cleaned_tdline)

            except (IndexError, KeyError, TypeError) as e:
                print(f"Error extracting Texts from {recipe_json}: {e}")

        # If we don't have enough descriptive content, try to extract from Classification
        if len(description_parts) <= 1:  # Only filename, need more content
            if 'Classification' in data and 'valueschar' in data['Classification']:
                try:
                    classification_df = pd.DataFrame(
                        data['Classification']['valueschar'])

                    if 'charactDescr' in classification_df.columns and 'valueCharLong' in classification_df.columns:
                        # Extract key descriptive features
                        key_features = [
                            'Product Line', 'Customer Brand', 'Project title',
                            'Color', 'Flavor', 'Flavour', 'Fruit content', 'Brix',
                            'Produktsegment (SD Reporting)', 'Industry (SD Reporting)'
                        ]

                        classification_parts = []
                        for _, row in classification_df.iterrows():
                            feature = row['charactDescr']
                            value = row['valueCharLong']

                            if (feature in key_features and
                                pd.notna(value) and
                                str(value).strip() and
                                    str(value).strip().lower() not in ['none', '', 'null']):
                                classification_parts.append(
                                    f"{feature}: {value}")

                        if classification_parts:
                            # Limit to top 3 most important characteristics
                            description_parts.extend(classification_parts[:3])

                except Exception as e:
                    print(
                        f"Error extracting Classification from {recipe_json}: {e}")

        # Combine all parts
        if description_parts:
            final_description = ", ".join(description_parts)
            return final_description
        else:
            # Fallback to simplified filename
            base_name = recipe_json.replace(
                '.json', '').replace('_', ' ').replace('-', ' ')
            return f"Recipe {base_name}"

    except Exception as e:
        print(f"Error extracting description from {recipe_json}: {e}")
        # Fallback to filename
        base_name = recipe_json.replace(
            '.json', '').replace('_', ' ').replace('-', ' ')
        return f"Recipe {base_name}"

########################################################################
# MAIN EXECUTION
########################################################################


# Initialize the enhanced two-step manager
manager = EnhancedTwoStepRecipeManager(
    collection_name="food_recipes_two_step",
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    max_features=200
)

# Prepare data for two-step system
features_list = []
values_list = []
descriptions_list = []  # Descriptions for text search
metadata_list = []
recipe_json_list = sorted(recipe_json_list)

print(f"Processing {len(recipe_json_list)} recipes for two-step search...")

for i in range(len(recipe_json_list)):
    # Extract features and values (existing)
    recipe_data = read_recipe_json(recipe_json_list[i])

    # Extract description (NEW)
    description = extract_recipe_description(recipe_json_list[i])

    # Check if we got valid data
    if recipe_data is not None and isinstance(recipe_data, pd.DataFrame):
        if 'charactDescr' in recipe_data.columns and 'valueCharLong' in recipe_data.columns:
            recipe_temp = recipe_data[['charactDescr', 'valueCharLong']]
            features_list.append(recipe_temp['charactDescr'].tolist())
            values_list.append(recipe_temp['valueCharLong'].tolist())
            descriptions_list.append(description)  # NEW
            metadata_list.append(
                {"recipe_name": recipe_json_list[i].split('.')[0]})
            print(
                f"Added recipe {i+1}/{len(recipe_json_list)}: {recipe_json_list[i]}")
            print(f"   Description: {description[:100]}...")
        else:
            print(f"Missing required columns in {recipe_json_list[i]}")
    else:
        print(
            f"Skipping invalid recipe {i+1}/{len(recipe_json_list)}: {recipe_json_list[i]}")

print(
    f"\nSuccessfully processed {len(features_list)} recipes out of {len(recipe_json_list)}")

# Update recipes with descriptions
if len(features_list) > 0:
    success = manager.update_recipes(
        features_list=features_list,
        values_list=values_list,
        descriptions_list=descriptions_list,
        metadata_list=metadata_list
    )

    if success:
        print("Recipes updated successfully with two-step search capability!")

        # Display analysis
        print(f"Feature Analysis Results:")
        feature_analysis = manager.get_feature_analysis()

        print(f"    Total Features: {feature_analysis['total_features']}")
        print(
            f"    Binary Features: {len(feature_analysis['binary_feature_names'])}")
        print(
            f"    Numerical Features: {len(feature_analysis['numerical_features'])}")
        print(
            f"    Categorical Features: {len(feature_analysis['categorical_features'])}")

        print(f"Manager Stats: {manager.get_stats()}")


# Example 1: Text-only search
print(f"EXAMPLE 1: Text-only search")
text_description_1 = """Fruit: Yellow peach particulates, max 12mm in size. Can also use a peach puree. 
Apricot Puree. Fruit content to be >30%. 
Flavour profile: Balanced peach and apricot flavours. Sweetness and slight tartness of 
the fruit. Dessert like poached peaches and apricots with compliment GD yogurt.
Colour: Vibrant deep orange colour with visible particulates. Not artificial looking. All 
N1 colours are available for development."""

detailed_results_1, summary_results_1 = run_two_step_search(
    manager=manager,
    text_description=text_description_1,
    query_df=None,  # No feature refinement
    text_top_k=3,
    final_top_k=3
)


# Example 2: Two-step search with feature refinement
print(f"EXAMPLE 2: Two-step search with feature refinement")
text_description_2 = """
                    Recipe Name: 521082 FIT BANANA
                    MaterialMasterShorttext: DELFP BANANA H FLIP FIT CHO
                    Colour Yellow, Flavour Banana, Stabilizer StarchLBGPectin, SMEAR  3M  TPC
"""

# Your existing query DataFrame
query_df = pd.DataFrame({
    'charactDescr': ['Puree/with pieces',
                     'Industry (SD Reporting)',
                     'Standard product',
                     'Halal',
                     'Kosher',
                     'Sweetener',
                     'Saccharose',
                     'Preserved',
                     'Artificial colors',
                     'Nature identical flavor',
                     'Natural flavor',
                     'Contains GMO',
                     'Flavour',
                     'Allergens',
                     'Starch',
                     'Pectin',
                     'LBG',
                     'Blend',
                     'Other stabilizer',
                     'Color',
                     'Aspartame',
                     'Xanthan',
                     'Pasteurization type',
                     'Material short text',
                     'Launch Type',
                     'Plant-based',
                     'Produktsegment (SD Reporting)'],
    'valueCharLong': ['puree',
                      'Dairy',
                      'Standard product',
                      'Halal',
                      'Kosher',
                      'Sweetener',
                      'No saccarose',
                      'No preservative',
                      'No artificial colors',
                      'No nature identical flavour',
                      'Natural flavour',
                      'Not gene manipulated',
                      'Flavour',
                      'No allergens',
                      'Starch',
                      'Pectin',
                      'LBG',
                      'No blend',
                      'No other stabilizer',
                      'Color',
                      'No Aspartame',
                      'No Xanthan',
                      'pasteurized',
                      'DELFP BANANA H FLIP FIT CHO',
                      'New Product',
                      'Non-Plant-based',
                      'Spoonable Yoghurt']
})


detailed_results_2, summary_results_2 = run_two_step_search(
    manager=manager,
    text_description=text_description_2,
    query_df=query_df,  # With feature refinement
    text_top_k=3,
    final_top_k=3
)
