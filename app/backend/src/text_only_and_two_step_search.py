from src.two_step_recipe_search import run_two_step_search
import pandas as pd


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
