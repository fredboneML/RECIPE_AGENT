# Recipe Reindexing Guide - Enhanced Feature Encoding

## 🎯 Problem Identified

The recipe search feature matching was not working optimally because:

1. **Indexing** used simple text embedding and basic feature storage
2. **Search** expected sophisticated feature encoding (binary detection, value normalization, etc.)
3. This **mismatch** caused poor feature matching (only 7.69% match rate)

## ✅ Solution Implemented

Updated the indexing script to use `EnhancedTwoStepRecipeManager` logic with **pre-analyzed feature types** from `charactDescr_valueCharLong_map.json`, which provides:

### **Enhanced Features:**
- **Pre-Analysis from Feature Map**: Analyzes all 349 unique features across 639,912 recipes BEFORE indexing
  - Automatically categorizes each feature as binary, numerical, range, or categorical
  - Uses actual value patterns from the entire database for accurate detection
  - Dramatically improves feature encoding accuracy
- **Binary Feature Detection**: Automatically detects Yes/No, Ja/Nein, allowed/not allowed patterns
- **Value Type Detection**: Identifies numerical (pH, Brix), percentage, range (3.0-4.5), and categorical features
- **Sophisticated Encoding**: Proper encoding for each feature type based on pre-analysis
- **Named Vectors in Qdrant**: 
  - `text` vector (384 dim) for Step 1: Text similarity search
  - `features` vector (484 dim) for Step 2: Feature-based refinement

## 📝 Files Modified/Created

### 1. `/app/backend/data/feature_analyzer.py` (NEW)
**Purpose:** Pre-analyzes all features from `charactDescr_valueCharLong_map.json` before indexing
**Features:**
- ✅ Loads comprehensive feature map (349 unique features across all recipes)
- ✅ Categorizes each feature as binary, numerical, range, or categorical
- ✅ Uses multilingual boolean detection (Yes/No, Ja/Nein, allowed/not allowed, etc.)
- ✅ Detects numerical subtypes (pH, Brix, percentage, temperature, etc.)
- ✅ Identifies range patterns (3.0-4.5, 25-35)
- ✅ Provides feature configuration for optimized indexing

### 2. `/app/backend/init_vector_index.py` & `/app/data/init_vector_index_qdrant.py`
**Changes:**
- ✅ Imports `EnhancedTwoStepRecipeManager` from `two_step_recipe_search.py`
- ✅ Imports `FeatureAnalyzer` for pre-analysis
- ✅ **PRE-ANALYZES** all features from `charactDescr_valueCharLong_map.json` before processing recipes
- ✅ Injects pre-analyzed feature types into `EnhancedTwoStepRecipeManager`
- ✅ Creates Qdrant collection with **named vectors** (text + features)
- ✅ Uses `EnhancedTwoStepRecipeManager.update_recipes()` to process all recipes
- ✅ Stores both text and feature vectors in Qdrant
- ✅ Logs comprehensive feature analysis statistics

### 3. `/app/backend/src/qdrant_recipe_manager.py`
**Changes:**
- ✅ Updated to search using named vector `"text"` (with fallback for backward compatibility)
- ✅ Enhanced `_match_feature_value()` method with multiple matching strategies
- ✅ Cleaned up unused imports and variables

### 4. `/app/docker-compose.yml`
**Changes:**
- ✅ Mounted `../Test_Input` directory to backend container
- ✅ Added `FEATURE_MAP_PATH` environment variable
- ✅ Ensures `charactDescr_valueCharLong_map.json` is accessible during indexing

## 🚀 How to Reindex Your Recipes

### **Step 1: Stop the Application**
```bash
cd /Volumes/ExternalDrive/Recipe_Agent/app
docker-compose down
```

### **Step 2: Delete Old Qdrant Collection**
The new collection structure is incompatible with the old one. You need to either:

**Option A: Delete Qdrant data volume**
```bash
docker volume rm app_qdrant-data
```

**Option B: Delete collection via Qdrant API** (if you want to keep other collections)
```bash
# Start only Qdrant
docker-compose up -d qdrant

# Delete the collection
curl -X DELETE "http://localhost:6333/collections/food_recipes_two_step"

# Stop Qdrant
docker-compose stop qdrant
```

### **Step 3: Reindex Recipes**
```bash
# Start all services
docker-compose up -d

# The init_vector_index service will automatically:
# 1. Pre-analyze features from charactDescr_valueCharLong_map.json
# 2. Load all 639,912 recipes
# 3. Process recipes with EnhancedTwoStepRecipeManager (with injected feature types)
# 4. Create text and feature vectors
# 5. Upload to Qdrant with named vectors

# Monitor the indexing progress
docker-compose logs -f backend_app
```

**Expected Output:**
```
INFO:vector_index_init:==============================================================
INFO:vector_index_init:PRE-ANALYZING FEATURES
INFO:vector_index_init:==============================================================
INFO:vector_index_init:Loaded feature map with 349 unique features
INFO:vector_index_init:Starting comprehensive feature analysis...
INFO:vector_index_init:✅ Binary features: 87
INFO:vector_index_init:✅ Numerical features: 34
INFO:vector_index_init:✅ Range features: 12
INFO:vector_index_init:✅ Categorical features: 216
INFO:vector_index_init:==============================================================
INFO:vector_index_init:✅ Pre-analyzed 349 features
INFO:vector_index_init:   Binary: 87
INFO:vector_index_init:   Numerical: 34
INFO:vector_index_init:   Range: 12
INFO:vector_index_init:   Categorical: 216
INFO:vector_index_init:==============================================================
INFO:vector_index_init:Step 1: Loading recipes from JSON files...
INFO:vector_index_init:Loaded 1000/639912 recipes
INFO:vector_index_init:...
INFO:vector_index_init:Loaded 639912 valid recipes
INFO:vector_index_init:Injecting pre-analyzed feature types into manager...
INFO:vector_index_init:✅ Injected 349 pre-analyzed feature types
INFO:vector_index_init:   Binary features: 87
INFO:vector_index_init:   Numerical features: 46
INFO:vector_index_init:   Categorical features: 216
INFO:vector_index_init:Step 2: Processing recipes with EnhancedTwoStepRecipeManager...
INFO:vector_index_init:Analyzing feature patterns...
INFO:vector_index_init:Processing 639912 recipes with descriptions...
INFO:vector_index_init:Step 3: Uploading recipes to Qdrant with named vectors...
INFO:vector_index_init:Uploaded batch: 1 to 10/639912
INFO:vector_index_init:...
INFO:vector_index_init:Successfully indexed 639912 recipes in Qdrant
INFO:vector_index_init:==============================================================
INFO:vector_index_init:FEATURE ANALYSIS:
INFO:vector_index_init:  Total unique features: 349
INFO:vector_index_init:  Binary features detected: XX
INFO:vector_index_init:  Numerical features: XX
INFO:vector_index_init:  Categorical features: XX
INFO:vector_index_init:==============================================================
```

### **Step 4: Test the Search**
Upload Brief 1 again and observe the improved feature matching:

**Expected Improvements:**
- ✅ Feature scores should be **> 0.15** (15%+) instead of 0.0769 (7.69%)
- ✅ Combined scores should be **> 0.35** instead of 0.26
- ✅ Better recipe ranking based on feature matching
- ✅ More relevant results overall

## 📊 What to Expect

### **Before (Old Indexing):**
```
Feature Score: 0.0769 (7.69% - 1 out of 13 features matched)
Combined Score: 0.2559
Ranking: Text similarity dominated
```

### **After (Enhanced Indexing):**
```
Feature Score: 0.15-0.40 (15-40% - 2-5 out of 13 features matched)
Combined Score: 0.35-0.50
Ranking: Balanced text + feature similarity
```

## 🔍 Technical Details

### **Feature Encoding Process:**

1. **Analysis Phase** (during indexing):
   - Scans all 639,912 recipes
   - Identifies unique feature names (349 total)
   - Detects value patterns:
     - Binary: Yes/No, allowed/not allowed
     - Numerical: pH, Brix, percentages
     - Categorical: Flavors, colors, etc.

2. **Encoding Phase**:
   - **Binary features**: Encoded as -1.0 (negative) or +1.0 (positive)
   - **Numerical features**: Normalized to [-1.0, 1.0] range
   - **Categorical features**: Label-encoded and scaled
   - **Text features**: SentenceTransformer embeddings (384 dim)

3. **Combined Vector**:
   - Text embedding: 384 dimensions
   - Categorical encoding: 100 dimensions
   - **Total feature vector**: 484 dimensions

### **Search Process:**

**Step 1: Text Search**
- Uses `text` named vector (384 dim)
- Finds top 50 candidates by description similarity
- Fast cosine similarity in vector space

**Step 2: Feature Refinement**
- Compares query features with candidate features
- Uses enhanced matching:
  - Exact match
  - Boolean normalization (Yes ↔ allowed)
  - Substring matching
  - Partial word matching
- Reranks using weighted score: **30% text + 70% features**

## ⏱️ Estimated Reindexing Time

- **Small dataset** (< 1,000 recipes): ~5-10 minutes
- **Medium dataset** (1,000-10,000 recipes): ~30-60 minutes  
- **Large dataset** (639,912 recipes): **~2-4 hours**

The indexing time depends on:
- CPU speed (feature analysis and encoding)
- Disk I/O (reading JSON files)
- Network (uploading to Qdrant)

## 🐛 Troubleshooting

### **Issue: "Collection already exists" error**
**Solution:** Delete the old collection first (see Step 2 above)

### **Issue: Indexing takes too long**
**Solution:** This is normal for 639K recipes. Monitor progress with `docker-compose logs -f backend_app`

### **Issue: Out of memory**
**Solution:** Increase Docker memory limit or reduce batch size in `init_vector_index_qdrant.py`

### **Issue: Feature scores still low after reindexing**
**Solution:** Check extracted features in logs - may need to adjust `DataExtractorRouterAgent` prompt

## 📈 Next Steps

After reindexing:
1. Test with Brief 1 (Peach Apricot)
2. Test with other briefs
3. Monitor feature match rates in logs
4. Fine-tune `DataExtractorRouterAgent` if needed

---

**Created:** November 22, 2025  
**Last Updated:** November 22, 2025

