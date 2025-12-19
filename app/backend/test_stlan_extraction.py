"""
Test script to extract stlan values from recipe JSON files.
Returns 'L', 'P', or 'Missing' based on the stlan field value.

Business Logic:
- The stlan field is located in the BillOfMaterialSTB parent field
- When both P and L versions exist for the same recipe, P is prioritized because
  it contains all maintained data (P versions are production versions with complete data)
- If only L exists, return 'L'
- If stlan field is missing or BillOfMaterialSTB is missing, return 'Missing'
"""

import json
import os
from pathlib import Path
from typing import Union, Literal


def extract_stlan(recipe_data: Union[dict, str]) -> Literal['L', 'P', 'Missing']:
    """
    Extract stlan value from a recipe JSON.

    The stlan field indicates the recipe version:
    - 'P': Production version (has all maintained data, prioritized when both P and L exist)
    - 'L': Lab/development version
    - 'Missing': stlan field not found (BillOfMaterialSTB may be missing)

    Business Logic:
    - When both P and L versions exist, P is returned (P has all data maintained)
    - When only L exists, L is returned
    - When stlan is missing or BillOfMaterialSTB is missing, 'Missing' is returned

    Args:
        recipe_data: Either a dictionary (parsed JSON) or a file path to a JSON file

    Returns:
        'P' if stlan field exists with value 'P' (prioritized even if L also exists)
        'L' if stlan field exists with value 'L' (and no P found)
        'Missing' if stlan doesn't exist or BillOfMaterialSTB is missing
    """
    # Load JSON if file path is provided
    if isinstance(recipe_data, str):
        try:
            with open(recipe_data, 'r', encoding='utf-8') as f:
                recipe_data = json.load(f)
        except Exception as e:
            print(f"Error loading file {recipe_data}: {e}")
            return 'Missing'

    # Recursively search for stlan field
    # We'll collect all stlan values and prioritize: P > L > Missing
    # P is prioritized because it contains all maintained data
    found_values = set()

    def find_stlan(obj):
        """Recursively search for stlan field in nested structures."""
        if isinstance(obj, dict):
            # Check if stlan key exists
            if 'stlan' in obj:
                value = obj['stlan']
                if value == 'L' or value == 'P':
                    found_values.add(value)

            # Recursively search in all values
            for value in obj.values():
                find_stlan(value)

        elif isinstance(obj, list):
            # Recursively search in all list items
            for item in obj:
                find_stlan(item)

    find_stlan(recipe_data)

    # Return priority: P > L > Missing
    # P is prioritized because production versions have all data maintained
    if 'P' in found_values:
        return 'P'
    elif 'L' in found_values:
        return 'L'
    else:
        return 'Missing'


def test_stlan_extraction(data_dir: str = None):
    """
    Test stlan extraction on all JSON files in the data directory.

    Args:
        data_dir: Path to data directory. Defaults to app/data/
    """
    if data_dir is None:
        # Get the project root (parent of app/)
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / 'app' / 'data'
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return

    # Get all JSON files
    json_files = sorted(data_dir.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print(
        f"Testing stlan extraction on {len(json_files)} JSON files in {data_dir}\n")
    print(f"{'File Name':<60} {'stlan Value':<15}")
    print("-" * 75)

    results = {'L': 0, 'P': 0, 'Missing': 0}

    for json_file in json_files:
        stlan_value = extract_stlan(str(json_file))
        results[stlan_value] += 1

        # Truncate filename if too long
        filename = json_file.name
        if len(filename) > 58:
            filename = filename[:55] + "..."

        print(f"{filename:<60} {stlan_value:<15}")

    print("-" * 75)
    print(f"\nSummary:")
    print(f"  Files with stlan='L': {results['L']}")
    print(f"  Files with stlan='P': {results['P']}")
    print(
        f"  Files with stlan='Missing' (no BillOfMaterialSTB or stlan field): {results['Missing']}")
    print(f"  Total files: {len(json_files)}")


if __name__ == '__main__':
    test_stlan_extraction()
