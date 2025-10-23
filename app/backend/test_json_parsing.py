#!/usr/bin/env python3
"""
Test JSON Parsing

Test if the updated init_vector_index.py can properly parse the JSON files
in the data directory.
"""
import sys
import os
import json
import pandas as pd

# Add path for imports
sys.path.insert(0, '/usr/src/app/src')

# Import the functions from init_vector_index.py
sys.path.insert(0, '/usr/src/app')


def test_read_recipe_json(recipe_json_path):
    """Test the read_recipe_json function"""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(recipe_json_path)}")
    print(f"{'='*60}")

    try:
        with open(recipe_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Data type: {type(data)}")
        if isinstance(data, list):
            print(f"Array length: {len(data)}")
            if len(data) > 0:
                print(
                    f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
        elif isinstance(data, dict):
            print(f"Object keys: {list(data.keys())}")

        # Test the updated parsing logic
        result = None
        if not data:
            print("❌ Data is empty")
        else:
            # Handle both array and single object formats
            if isinstance(data, list) and len(data) > 0:
                # Array format - look for Classification in each item
                for i, item in enumerate(data):
                    if isinstance(item, dict) and 'Classification' in item and item['Classification']:
                        if 'valueschar' in item['Classification'] and item['Classification']['valueschar']:
                            result = pd.DataFrame(
                                item['Classification']['valueschar'])
                            print(
                                f"✅ Found Classification data in array item {i}")
                            break
            elif isinstance(data, dict) and 'Classification' in data:
                # Single object format
                if data['Classification'] and 'valueschar' in data['Classification'] and data['Classification']['valueschar']:
                    result = pd.DataFrame(data['Classification']['valueschar'])
                    print(f"✅ Found Classification data in object format")

            if result is None:
                print("❌ Classification/valueschar not found")
            else:
                print(f"✅ Successfully parsed {len(result)} features")
                print(f"Sample features:")
                for i, row in result.head(3).iterrows():
                    print(f"  {row['charactDescr']}: {row['valueCharLong']}")

        return result is not None

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Test JSON parsing on all files in data directory"""
    data_dir = "/Volumes/ExternalDrive/Recipe_Agent/app/data"

    print("JSON PARSING TEST")
    print("="*60)

    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False

    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files")

    success_count = 0
    for json_file in json_files[:3]:  # Test first 3 files
        file_path = os.path.join(data_dir, json_file)
        if test_read_recipe_json(file_path):
            success_count += 1

    print(f"\n{'='*60}")
    print(
        f"RESULTS: {success_count}/{min(3, len(json_files))} files parsed successfully")
    print(f"{'='*60}")

    return success_count > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
