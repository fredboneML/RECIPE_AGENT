# Excel Output - Sheet Descriptions

## Overview
This Excel file contains extracted data from recipe JSON files, organized into 5 sheets for easy analysis and filtering.

---

## Sheet 1: Specific_Fields
**Purpose:** Contains the three specific fields requested: Z_FRUCHTG, Z_INH02, and Z_INH19.

**Contents:**
- Field name (Z_FRUCHTG, Z_INH02, or Z_INH19)
- Description of each field
- Type (Character or Numeric)
- Value (formatted for display)
- Value_Neutral (for filtering/sorting)
- Unit (if applicable)

**Use Case:** Quick reference for the three key fields of interest.

---

## Sheet 2: Classification_Char
**Purpose:** Contains all character-based classification fields from the recipe.

**Contents:**
- Characteristic_Code (the Z_... field identifier)
- Description (human-readable name)
- Value (the actual character value)
- Value_Neutral (neutral/shortened value)
- Additional metadata columns

**Use Case:** View and filter all text-based classification attributes (e.g., "Süßstoff", "Aspartam", "Industrie").

---

## Sheet 3: Classification_Num
**Purpose:** Contains all numeric classification fields from the recipe.

**Contents:**
- Characteristic_Code (the Z_... field identifier)
- Description (human-readable name)
- Value (the numeric value)
- Unit (measurement unit if applicable)
- Additional metadata columns

**Use Case:** View and filter all numeric classification attributes (e.g., Brix, Fruchtgehalt, Abfülltemperatur).

---

## Sheet 4: Summary
**Purpose:** Provides metadata and overview of the extraction process.

**Contents:**
- Recipe ID
- File path
- Whether Classification data exists
- Which specific fields were found (Z_FRUCHTG, Z_INH02, Z_INH19)
- Total counts of character and numeric fields

**Use Case:** Quick overview to verify extraction completeness and data availability.

---

## Sheet 5: Consolidated
**Purpose:** Unified view combining all fields in a single table for comprehensive analysis and custom sorting.

**Contents:**
- **Field_Type:** Indicates whether the row is 'Z_FRUCHTG', 'Z_INH02', 'Z_INH19', or 'Classification'
- **Characteristic_Code:** The Z_... field identifier
- **Description:** Human-readable description of the field
- **Value:** The corresponding value (formatted appropriately)
- **Sort:** Empty column reserved for user-defined sorting values

**Use Case:** 
- Single view of all recipe classification data
- Easy filtering by Field_Type to separate specific fields from general classification
- Custom sorting using the Sort column

### About the Sort Column
The **Sort** column is intentionally left empty for you to define your own sorting logic. You can:
- Add numeric values (1, 2, 3...) to control the display order
- Add category labels to group related fields
- Use any sorting scheme that fits your analysis needs

This allows you to organize the consolidated data according to your business logic, priority, or analysis requirements without modifying the source data.

---

## Notes
- All sheets are filterable and sortable in Excel
- The Consolidated sheet provides the most comprehensive view for analysis
- Specific fields (Z_FRUCHTG, Z_INH02, Z_INH19) appear in both the Specific_Fields sheet and the Consolidated sheet for convenience
