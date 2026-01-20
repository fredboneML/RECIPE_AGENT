#!/usr/bin/env python3
"""
Extract 60 specified fields from recipe JSON and export to Excel in a specific order.

Extracts exactly 60 fields in the specified order:
- If a field is available, extracts its value
- If a field is missing, treats it as NaN

Outputs to a multi-sheet Excel file for easy filtering and navigation.

Usage:
    python extract_specified_recipe_fields_sorted.py <input_json_file> [output_excel_file]
    
    If output_excel_file is not provided, it will be generated from the input filename.

Output Excel Structure:
    - Specific_Fields: Contains the 60 specified fields with their values (NaN if missing)
    - Classification_Char: All character classification fields (charactDescr + valueschar)
    - Classification_Num: All numeric classification fields (charactDescr + valuesnum)
    - Summary: Metadata about the extraction (recipe ID, field counts, etc.)
    - Consolidated: Unified view with Field_Type, Characteristic_Code, Description, Value, Sort
      (only contains the 60 specified fields in the specified order)

Example:
    python extract_specified_recipe_fields_sorted.py app/data/000000000000442939_AT10_01_L.json
"""
import json
import pandas as pd
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Define the 60 fields in the exact order specified
SPECIFIED_FIELDS_ORDERED = [
    'Z_MAKTX', 'Z_INH01', 'Z_WEIM', 'Z_KUNPROGRU', 'Z_PRODK', 'Z_INH07', 'Z_KOCHART', 'Z_KNOGM',
    'Z_INH08', 'Z_INH12', 'ZMX_TIPOALERG', 'Z_INH02', 'Z_INH03', 'Z_INH19', 'Z_INH04', 'Z_INH18',
    'Z_INH05', 'Z_INH09', 'Z_INH06', 'Z_INH06Z', 'Z_FSTAT', 'Z_INH21', 'Z_INH13', 'Z_INH14',
    'Z_INH15', 'Z_INH16', 'Z_INH20', 'Z_STABGU', 'Z_STABCAR', 'Z_STAGEL', 'Z_STANO', 'Z_INH17',
    'Z_BRIX', 'Z_PH', 'ZM_PH', 'Z_VISK20S', 'Z_VISK20S_7C', 'Z_VISK30S', 'Z_VISK60S', 'Z_VISKHAAKE',
    'ZMX_DD103', 'ZMX_DD102', 'ZM_AW', 'Z_FGAW', 'Z_FRUCHTG', 'ZMX_DD108', 'Z_AW', 'Z_FLST',
    'Z_PP', 'ZMX_DD109', 'Z_DOSIER', 'Z_ZUCKER', 'Z_FETTST', 'ZMX_DD104', 'Z_PROT', 'Z_SALZ',
    'Z_INH01K', 'Z_INH01H', 'Z_DAIRY', 'Z_BFS'
]


def extract_specified_fields_ordered(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract the 60 specified fields in the exact order, treating missing fields as NaN.
    
    Args:
        data: The recipe JSON data
        
    Returns:
        DataFrame with all 60 fields in the specified order (NaN for missing fields)
    """
    results = []
    
    # Create a lookup dictionary for faster access
    char_lookup = {}
    num_lookup = {}
    
    if 'Classification' in data and data['Classification']:
        classification = data['Classification']
        
        # Build lookup for character fields
        if 'valueschar' in classification and classification['valueschar']:
            for item in classification['valueschar']:
                charact = item.get('charact', '')
                if charact:
                    char_lookup[charact] = {
                        'description': item.get('charactDescr', ''),
                        'value': item.get('valueCharLong', item.get('valueChar', '')),
                        'value_neutral': item.get('valueNeutralLong', item.get('valueNeutral', '')),
                        'type': 'Character'
                    }
        
        # Build lookup for numeric fields
        if 'valuesnum' in classification and classification['valuesnum']:
            for item in classification['valuesnum']:
                charact = item.get('charact', '')
                if charact:
                    value_from = item.get('valueFrom', '')
                    unit = item.get('unitFrom', '')
                    
                    # Format the value for display
                    if value_from:
                        if unit:
                            value_str = f"{value_from} {unit}"
                        else:
                            value_str = str(value_from)
                    else:
                        value_str = np.nan
                    
                    num_lookup[charact] = {
                        'description': item.get('charactDescr', ''),
                        'value': value_str,
                        'value_neutral': value_from,
                        'unit': unit if unit else None,
                        'type': 'Numeric'
                    }
    
    # Extract fields in the specified order
    for field_name in SPECIFIED_FIELDS_ORDERED:
        # Check character fields first
        if field_name in char_lookup:
            field_data = char_lookup[field_name]
            results.append({
                'Field': field_name,
                'Description': field_data['description'],
                'Type': field_data['type'],
                'Value': field_data['value'],
                'Value_Neutral': field_data['value_neutral'],
                'Unit': None
            })
        # Check numeric fields
        elif field_name in num_lookup:
            field_data = num_lookup[field_name]
            results.append({
                'Field': field_name,
                'Description': field_data['description'],
                'Type': field_data['type'],
                'Value': field_data['value'],
                'Value_Neutral': field_data['value_neutral'],
                'Unit': field_data['unit']
            })
        # Field not found - add with NaN values
        else:
            results.append({
                'Field': field_name,
                'Description': np.nan,
                'Type': np.nan,
                'Value': np.nan,
                'Value_Neutral': np.nan,
                'Unit': np.nan
            })
    
    return pd.DataFrame(results)


def extract_classification_char(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract all Classification valueschar data.
    
    Args:
        data: The recipe JSON data
        
    Returns:
        DataFrame with all valueschar data
    """
    if 'Classification' not in data or not data['Classification']:
        return pd.DataFrame()
    
    classification = data['Classification']
    
    if 'valueschar' in classification and classification['valueschar']:
        df = pd.DataFrame(classification['valueschar'])
        # Select and rename relevant columns for clarity
        columns_to_keep = ['charact', 'charactDescr', 'valueChar', 'valueCharLong', 
                          'valueNeutral', 'valueNeutralLong']
        available_columns = [col for col in columns_to_keep if col in df.columns]
        
        if available_columns:
            df = df[available_columns]
            # Rename for clarity
            rename_map = {
                'charact': 'Characteristic_Code',
                'charactDescr': 'Description',
                'valueChar': 'Value_Short',
                'valueCharLong': 'Value',
                'valueNeutral': 'Value_Neutral_Short',
                'valueNeutralLong': 'Value_Neutral'
            }
            df = df.rename(columns=rename_map)
            return df
    
    return pd.DataFrame()


def extract_classification_num(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract all Classification valuesnum data.
    
    Args:
        data: The recipe JSON data
        
    Returns:
        DataFrame with all valuesnum data
    """
    if 'Classification' not in data or not data['Classification']:
        return pd.DataFrame()
    
    classification = data['Classification']
    
    if 'valuesnum' in classification and classification['valuesnum']:
        df = pd.DataFrame(classification['valuesnum'])
        # Select and rename relevant columns
        columns_to_keep = ['charact', 'charactDescr', 'valueFrom', 'valueRelation', 
                          'unitFrom', 'unitFromIso']
        available_columns = [col for col in columns_to_keep if col in df.columns]
        
        if available_columns:
            df = df[available_columns]
            # Rename for clarity
            rename_map = {
                'charact': 'Characteristic_Code',
                'charactDescr': 'Description',
                'valueFrom': 'Value',
                'valueRelation': 'Value_Relation',
                'unitFrom': 'Unit',
                'unitFromIso': 'Unit_ISO'
            }
            df = df.rename(columns=rename_map)
            return df
    
    return pd.DataFrame()


def create_consolidated_sheet(specific_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a consolidated sheet with the 60 specified fields in the specified order.
    
    Args:
        specific_df: DataFrame with the 60 specified fields
        
    Returns:
        Consolidated DataFrame with columns: Field_Type, Characteristic_Code, Description, Value, Sort
    """
    consolidated_rows = []
    
    # Process fields in the specified order (already ordered in specific_df)
    if not specific_df.empty:
        for _, row in specific_df.iterrows():
            field_name = row['Field']
            value = row['Value']
            
            # Use field name as Field_Type for the 60 specified fields
            consolidated_rows.append({
                'Field_Type': field_name,
                'Characteristic_Code': field_name,
                'Description': row['Description'] if pd.notna(row['Description']) else '',
                'Value': value if pd.notna(value) else '',
                'Sort': ''  # Empty for user to fill
            })
    
    df = pd.DataFrame(consolidated_rows)
    # Fill Sort column with empty strings instead of NaN
    if 'Sort' in df.columns:
        df['Sort'] = df['Sort'].fillna('')
    return df


def extract_recipe_to_excel(recipe_json_path: str, output_excel_path: str) -> bool:
    """
    Extract 60 specified fields from recipe JSON and save to Excel.
    
    Args:
        recipe_json_path: Path to input JSON file
        output_excel_path: Path to output Excel file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read JSON file
        with open(recipe_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract recipe ID from filename
        recipe_id = Path(recipe_json_path).stem
        
        # Extract the 60 specified fields in order
        specific_df = extract_specified_fields_ordered(data)
        
        # Extract Classification data (for other sheets)
        classification_char_df = extract_classification_char(data)
        classification_num_df = extract_classification_num(data)
        
        # Create consolidated sheet (only contains the 60 specified fields)
        consolidated_df = create_consolidated_sheet(specific_df)
        
        # Count how many fields were found (non-NaN)
        fields_found = specific_df['Value'].notna().sum()
        
        # Create Excel writer
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            # Sheet 1: Specific Fields (60 fields in order)
            specific_df.to_excel(writer, sheet_name='Specific_Fields', index=False)
            
            # Sheet 2: All Classification Character Values
            if not classification_char_df.empty:
                classification_char_df.to_excel(writer, sheet_name='Classification_Char', index=False)
            else:
                empty_df = pd.DataFrame(columns=['Characteristic_Code', 'Description', 'Value'])
                empty_df.to_excel(writer, sheet_name='Classification_Char', index=False)
            
            # Sheet 3: All Classification Numeric Values
            if not classification_num_df.empty:
                classification_num_df.to_excel(writer, sheet_name='Classification_Num', index=False)
            else:
                empty_df = pd.DataFrame(columns=['Characteristic_Code', 'Description', 'Value', 'Unit'])
                empty_df.to_excel(writer, sheet_name='Classification_Num', index=False)
            
            # Sheet 4: Summary (recipe info)
            summary_data = {
                'Recipe_ID': [recipe_id],
                'File_Path': [recipe_json_path],
                'Has_Classification': ['Yes' if 'Classification' in data and data['Classification'] else 'No'],
                'Total_Specified_Fields': [len(SPECIFIED_FIELDS_ORDERED)],
                'Fields_Found': [fields_found],
                'Fields_Missing': [len(SPECIFIED_FIELDS_ORDERED) - fields_found],
                'Total_Char_Fields': [len(classification_char_df) if not classification_char_df.empty else 0],
                'Total_Num_Fields': [len(classification_num_df) if not classification_num_df.empty else 0]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 5: Consolidated view (60 fields in specified order)
            consolidated_df.to_excel(writer, sheet_name='Consolidated', index=False)
        
        print(f"✅ Successfully extracted data to: {output_excel_path}")
        print(f"   Recipe ID: {recipe_id}")
        print(f"   Specified fields (total): {len(SPECIFIED_FIELDS_ORDERED)}")
        print(f"   Fields found: {fields_found}")
        print(f"   Fields missing: {len(SPECIFIED_FIELDS_ORDERED) - fields_found}")
        print(f"   Classification char fields: {len(classification_char_df)}")
        print(f"   Classification num fields: {len(classification_num_df)}")
        print(f"   Consolidated sheet rows: {len(consolidated_df)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python extract_specified_recipe_fields_sorted.py <input_json_file> [output_excel_file]")
        sys.exit(1)
    
    input_json = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_excel = sys.argv[2]
    else:
        # Generate output filename from input
        input_path = Path(input_json)
        output_excel = input_path.parent / f"{input_path.stem}_specified_fields.xlsx"
    
    if not os.path.exists(input_json):
        print(f"❌ Error: Input file not found: {input_json}")
        sys.exit(1)
    
    success = extract_recipe_to_excel(input_json, str(output_excel))
    
    if success:
        print(f"\n✅ Extraction complete!")
        print(f"   Output file: {output_excel}")
        sys.exit(0)
    else:
        print(f"\n❌ Extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
