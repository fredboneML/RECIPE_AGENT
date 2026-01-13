#!/usr/bin/env python3
"""
Extract specific fields from recipe JSON and export to Excel.

Extracts:
- Z_FRUCHTG, Z_INH02, Z_INH19 (specific fields)
- Classification data (charactDescr and valueschar/valuesnum)

Outputs to a multi-sheet Excel file for easy filtering and navigation.

Usage:
    python extract_recipe_fields.py <input_json_file> [output_excel_file]
    
    If output_excel_file is not provided, it will be generated from the input filename.

Output Excel Structure:
    - Specific_Fields: Contains Z_FRUCHTG, Z_INH02, Z_INH19 with their values
    - Classification_Char: All character classification fields (charactDescr + valueschar)
    - Classification_Num: All numeric classification fields (charactDescr + valuesnum)
    - Summary: Metadata about the extraction (recipe ID, field counts, etc.)
    - Consolidated: Unified view with Field_Type, Characteristic_Code, Description, Value, Sort

Example:
    python extract_recipe_fields.py app/data/000000000000442939_AT10_01_L.json
"""
import json
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

def extract_specific_fields(data: Dict[str, Any], fields: List[str]) -> pd.DataFrame:
    """
    Extract specific fields (Z_FRUCHTG, Z_INH02, Z_INH19) from Classification.
    
    Args:
        data: The recipe JSON data
        fields: List of field names to extract
        
    Returns:
        DataFrame with the specific fields
    """
    results = []
    
    if 'Classification' not in data or not data['Classification']:
        return pd.DataFrame(columns=['Field', 'Description', 'Type', 'Value', 'Value_Neutral', 'Unit'])
    
    classification = data['Classification']
    
    # Check in valueschar
    if 'valueschar' in classification and classification['valueschar']:
        for item in classification['valueschar']:
            charact = item.get('charact', '')
            if charact in fields:
                results.append({
                    'Field': charact,
                    'Description': item.get('charactDescr', ''),
                    'Type': 'Character',
                    'Value': item.get('valueCharLong', item.get('valueChar', '')),
                    'Value_Neutral': item.get('valueNeutralLong', item.get('valueNeutral', '')),
                    'Unit': None
                })
    
    # Check in valuesnum
    if 'valuesnum' in classification and classification['valuesnum']:
        for item in classification['valuesnum']:
            charact = item.get('charact', '')
            if charact in fields:
                value_from = item.get('valueFrom', '')
                unit = item.get('unitFrom', '')
                
                # Format the value for display
                if value_from:
                    if unit:
                        value_str = f"{value_from} {unit}"
                    else:
                        value_str = str(value_from)
                else:
                    value_str = 'N/A'
                
                results.append({
                    'Field': charact,
                    'Description': item.get('charactDescr', ''),
                    'Type': 'Numeric',
                    'Value': value_str,
                    'Value_Neutral': value_from,  # Store numeric value
                    'Unit': unit if unit else None
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


def create_consolidated_sheet(
    specific_df: pd.DataFrame,
    classification_char_df: pd.DataFrame,
    classification_num_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a consolidated sheet with all fields in a unified format.
    
    Args:
        specific_df: DataFrame with Z_FRUCHTG, Z_INH02, Z_INH19
        classification_char_df: DataFrame with all character classification fields
        classification_num_df: DataFrame with all numeric classification fields
        
    Returns:
        Consolidated DataFrame with columns: Field_Type, Characteristic_Code, Description, Value, Sort
    """
    consolidated_rows = []
    
    # Add specific fields (Z_FRUCHTG, Z_INH02, Z_INH19)
    if not specific_df.empty:
        for _, row in specific_df.iterrows():
            field_name = row['Field']
            # Format value for display
            value = row['Value']
            if pd.notna(row.get('Unit')) and row.get('Unit'):
                # Already formatted in Value column
                pass
            
            consolidated_rows.append({
                'Field_Type': field_name,  # 'Z_FRUCHTG', 'Z_INH02', or 'Z_INH19'
                'Characteristic_Code': field_name,
                'Description': row['Description'],
                'Value': value,
                'Sort': ''  # Empty for user to fill
            })
    
    # Add all Classification character fields
    if not classification_char_df.empty:
        for _, row in classification_char_df.iterrows():
            char_code = row.get('Characteristic_Code', '')
            # Skip if it's one of the specific fields (already added above)
            if char_code not in ['Z_FRUCHTG', 'Z_INH02', 'Z_INH19']:
                consolidated_rows.append({
                    'Field_Type': 'Classification',
                    'Characteristic_Code': char_code,
                    'Description': row.get('Description', ''),
                    'Value': row.get('Value', ''),
                    'Sort': ''  # Empty for user to fill
                })
    
    # Add all Classification numeric fields
    if not classification_num_df.empty:
        for _, row in classification_num_df.iterrows():
            char_code = row.get('Characteristic_Code', '')
            # Skip if it's one of the specific fields (already added above)
            if char_code not in ['Z_FRUCHTG', 'Z_INH02', 'Z_INH19']:
                # Format numeric value
                value = row.get('Value', '')
                unit = row.get('Unit', '')
                if pd.notna(unit) and unit:
                    value_str = f"{value} {unit}" if pd.notna(value) else ''
                else:
                    value_str = str(value) if pd.notna(value) else ''
                
                consolidated_rows.append({
                    'Field_Type': 'Classification',
                    'Characteristic_Code': char_code,
                    'Description': row.get('Description', ''),
                    'Value': value_str,
                    'Sort': ''  # Empty for user to fill
                })
    
    df = pd.DataFrame(consolidated_rows)
    # Fill Sort column with empty strings instead of NaN
    if 'Sort' in df.columns:
        df['Sort'] = df['Sort'].fillna('')
    return df


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


def extract_recipe_to_excel(recipe_json_path: str, output_excel_path: str) -> bool:
    """
    Extract fields from recipe JSON and save to Excel.
    
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
        
        # Extract specific fields
        specific_fields = ['Z_FRUCHTG', 'Z_INH02', 'Z_INH19']
        specific_df = extract_specific_fields(data, specific_fields)
        
        # Extract Classification data
        classification_char_df = extract_classification_char(data)
        classification_num_df = extract_classification_num(data)
        
        # Create consolidated sheet
        consolidated_df = create_consolidated_sheet(
            specific_df, classification_char_df, classification_num_df
        )
        
        # Create Excel writer
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            # Sheet 1: Specific Fields
            if not specific_df.empty:
                specific_df.to_excel(writer, sheet_name='Specific_Fields', index=False)
            else:
                # Create empty sheet with headers
                empty_df = pd.DataFrame(columns=['Field', 'Description', 'Type', 'Value', 'Value_Neutral', 'Unit'])
                empty_df.to_excel(writer, sheet_name='Specific_Fields', index=False)
            
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
                'Z_FRUCHTG_Found': ['Yes' if not specific_df.empty and 'Z_FRUCHTG' in specific_df['Field'].values else 'No'],
                'Z_INH02_Found': ['Yes' if not specific_df.empty and 'Z_INH02' in specific_df['Field'].values else 'No'],
                'Z_INH19_Found': ['Yes' if not specific_df.empty and 'Z_INH19' in specific_df['Field'].values else 'No'],
                'Total_Char_Fields': [len(classification_char_df) if not classification_char_df.empty else 0],
                'Total_Num_Fields': [len(classification_num_df) if not classification_num_df.empty else 0]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 5: Consolidated view
            if not consolidated_df.empty:
                consolidated_df.to_excel(writer, sheet_name='Consolidated', index=False)
            else:
                empty_df = pd.DataFrame(columns=['Field_Type', 'Characteristic_Code', 'Description', 'Value', 'Sort'])
                empty_df.to_excel(writer, sheet_name='Consolidated', index=False)
        
        print(f"✅ Successfully extracted data to: {output_excel_path}")
        print(f"   Recipe ID: {recipe_id}")
        print(f"   Specific fields found: {len(specific_df)}")
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
        print("Usage: python extract_recipe_fields.py <input_json_file> [output_excel_file]")
        sys.exit(1)
    
    input_json = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_excel = sys.argv[2]
    else:
        # Generate output filename from input
        input_path = Path(input_json)
        output_excel = input_path.parent / f"{input_path.stem}_extracted.xlsx"
    
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
