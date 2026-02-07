# Recipe Search Agent

An intelligent recipe search system that uses AI agents to extract requirements from customer briefs and find matching recipes from a database of 600K+ recipes using semantic search and feature-based filtering.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Agents & End-to-End Flow](#core-agents--end-to-end-flow)
- [Prerequisites](#prerequisites)
- [Deployment on a New Server](#deployment-on-a-new-server)
  - [Step 1: Infrastructure Setup](#step-1-infrastructure-setup)
  - [Step 2: Database Initialization](#step-2-database-initialization)
  - [Step 3: Qdrant Vector Database Setup](#step-3-qdrant-vector-database-setup)
  - [Step 4: Recipe Indexing](#step-4-recipe-indexing)
  - [Step 5: Create Payload Indexes](#step-5-create-payload-indexes)
  - [Step 6: Application Deployment](#step-6-application-deployment)
  - [Step 7: Nginx Reverse Proxy (Production)](#step-7-nginx-reverse-proxy-production)
  - [Step 8: User Management](#step-8-user-management)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Recipe Search Agent is a production-ready system that:

- **Extracts structured information** from customer briefs (PDFs, images, text) using LLM-powered agents
- **Searches 600K+ recipes** using semantic similarity and feature-based matching
- **Filters by numerical constraints** (Brix >40, pH <4.1, fruit content 30-40%, etc.)
- **Handles multilingual queries** (English, German, French, Italian, Spanish, Portuguese, Dutch, Danish)
- **Provides comparison tables** showing 60 specified fields across matching recipes
- **Supports SSO authentication** via Azure AD (Microsoft Entra ID)

---

## Architecture

### Components

1. **Frontend** (React): User interface for uploading briefs and viewing results
2. **Backend** (FastAPI): REST API with AI agents for extraction and search
3. **PostgreSQL**: Stores user data, conversations, and translation caches
4. **Qdrant**: Vector database for semantic recipe search (600K+ recipes)
5. **Docker Compose**: Orchestrates all services

### Key Features

- **60 Specified Fields**: Structured extraction of key recipe attributes (Brix, pH, viscosity, allergens, etc.)
- **Binary Opposition Mapping**: Handles "no sugar" vs "sugar", "preservative-free", etc.
- **Multilingual Normalization**: Converts German/French/etc. feature names to English for consistent search
- **Numerical Range Filtering**: Efficient filtering on numerical constraints using Qdrant payload indexes
- **Two-Step Search**: Combines text similarity with feature matching for precise results

---

## Core Agents & End-to-End Flow

### Flow Diagram

```
User Uploads Brief
       ↓
┌──────────────────────────────────────────────────────────────┐
│  Data Extractor Router Agent (gpt-4o-mini)                   │
│  ├─ Extract text description                                 │
│  ├─ Extract features & values                                │
│  ├─ Parse numerical constraints → numerical_filters          │
│  └─ Parse categorical constraints → categorical_filters      │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│  Recipe Search Agent (Text-First Architecture)               │
│  ├─ Step 0: Recipe name matching (exact/partial/fuzzy)       │
│  │   └─ Hyphen/space normalization + fuzzy word-overlap      │
│  ├─ Step 1a: Text-based semantic search (100 candidates)     │
│  ├─ Step 1a-: Name-line semantic search (200 candidates)     │
│  ├─ Step 1a-f: Flavor + MST-format variant searches          │
│  │   └─ German translations, FP/FZ prefixes, hyphen variants │
│  ├─ Step 1a+: Original query text search (100 candidates)    │
│  ├─ Flavor Safeguard (English + German flavor search)        │
│  ├─ Step 1b: Apply filters IN-MEMORY on candidates           │
│  │   └─ Progressive relaxation (strict → 10% → 20% → skip)  │
│  └─ Step 2: Feature scoring + ranking → Top 3                │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│  Qdrant Vector Database                                      │
│  ├─ Named vectors: text (384d) + features (484d)             │
│  ├─ Payload indexes: numerical.Z_*, spec_fields.Z_*          │
│  └─ Filter: Range (numerical) + MatchValue (categorical)     │
└──────────────────────────────────────────────────────────────┘
       ↓
   Top 3 Results + Comparison Table (60 fields) → UI
```

### 1. **Data Extractor Router Agent** (`data_extractor_router.py`)

**Purpose**: Extracts structured information from customer briefs

**Input**: 
- Customer brief (PDF, image, or text)
- Multilingual content (EN, DE, FR, IT, ES, PT, NL, DA)

**Processing**:
1. Uses LLM (gpt-4o-mini) to extract:
   - **Text description**: Natural language summary of the brief
   - **Features**: Structured list of product attributes (flavors, colors, stabilizers, etc.)
   - **Feature values**: Corresponding values for each feature
   - **Numerical constraints**: Range queries (e.g., "Brix >40", "pH <4.1", "fruit content 30-40%")
   - **Categorical constraints**: Yes/No attributes for filtering (preservatives, artificial colors, allergen-free, etc.)

2. Normalizes multilingual features to English using `feature_normalizer.py`

3. Parses numerical constraints using `numerical_constraint_parser.py`:
   - `>30%` → `{"gt": 30.0}`
   - `<4.1` → `{"lt": 4.1}`
   - `30+/-5°` → `{"gte": 25.0, "lte": 35.0}`
   - `6-9 +/- 2` → `{"gte": 7.0, "lte": 11.0}`

4. Parses categorical constraints using `categorical_constraint_parser.py`:
   - Maps brief field names to Z_* codes (e.g., "Preserved" → "Z_INH04")
   - Normalizes multilingual values to "Yes"/"No":
     - German: "Ja"/"Nein", "ohne"/"mit", "frei von" → "Yes"/"No"
     - French: "Oui"/"Non", "sans"/"avec" → "Yes"/"No"
   - Handles special values: "Saccharose", "Stückig", "homogen", etc.

**Output**:
```python
{
    "search_type": "two_step",
    "text_description": "Peach and apricot fruit preparation...",
    "features": ["Flavour: Peach, Apricot", "Color: Vibrant deep orange", ...],
    "feature_values": ["Peach", "Apricot", "Vibrant deep orange", ...],
    "numerical_filters": {
        "Z_FRUCHTG": {"gt": 30.0},
        "Z_PH": {"lt": 4.1},
        "Z_BRIX": {"gte": 25.0, "lte": 35.0},
        "Z_VISK20S": {"gte": 7.0, "lte": 11.0}
    },
    "categorical_filters": {
        "Z_INH04": {"value": "No"},   # No preservatives (Preserved = No)
        "Z_INH05": {"value": "No"},   # No artificial colors
        "Z_INH09": {"value": "Yes"},  # Natural flavors
        "Z_INH12": {"value": "Yes"},  # Allergen-free
        "Z_INH01": {"value": "Saccharose"}  # Sugar type
    },
    "reasoning": "Extracted flavors, product segment..."
}
```

**Supported Categorical Fields** (18 fields):
| Field Code | Field Name | Possible Values |
|------------|------------|-----------------|
| Z_INH01 | Sugar type | Saccharose, Fructose, Glucose, No sugar, etc. |
| Z_INH04 | Preserved | Yes, No |
| Z_INH05 | Artificial colors | Yes, No |
| Z_INH06 | Artificial flavors | Yes, No |
| Z_INH09 | Natural flavor | Yes, No |
| Z_INH10 | GMO | positive, negative, neutral |
| Z_INH12 | Allergen-free | Yes (allergen-free), No (contains allergens) |
| Z_INH13 | Sweetener | Yes, No |
| Z_INH14 | Starch | Yes, No |
| Z_INH15 | Halal | Yes, No |
| Z_INH16 | Kosher | Yes, No |
| Z_KON | Consistency | Stückig (chunky), homogen (homogeneous) |
| Z_BIOPRO | Organic | Yes, No |
| Z_FAIR | Fair Trade | Yes, No |
| Z_VEGAN | Vegan | Yes, No |
| Z_VEGET | Vegetarian | Yes, No |
| Z_LMFRREI | Lactose-free | Yes, No |
| Z_GLUTFR | Gluten-free | Yes, No |

### 2. **Recipe Search Agent** (`recipe_search_agent.py`)

**Purpose**: Searches Qdrant for matching recipes and generates comparison table

**Input**: Output from Data Extractor Router Agent

**Processing**:

1. **Text-First Hybrid Search** (`qdrant_recipe_manager.py`):

   The search uses a **Text-First** architecture: semantic search finds a broad candidate pool first (no filters), then filters are applied in-memory with progressive relaxation.

   - **Step 0: Recipe Name Matching** (using original query)
     - Searches for exact and partial matches on MaterialMasterShorttext using Qdrant `MatchText`
     - **Hyphen/space normalization**: Treats hyphens and spaces equivalently so "ALOE VERA PASSIONFRUCHT" matches "FP ALOE VERA-PASSIONFRUCHT TJ"
     - Example: Query "FZ Orange Mango Grüner Tee" finds "FZ Orange Mango Grüner Tee Smoothie"
     - **Exact match**: Score 0.95 (query == recipe name)
     - **Partial match**: Score 0.92 (query contained in recipe name)
     - **Prefix match**: Score 0.90 (recipe name starts with query)
     - **Fuzzy fallback**: If exact/partial matching fails, falls back to word-overlap matching using individual word `MatchText` queries plus LLM-corrected flavor terms (handles typos like "passionfruch" → "Passionfruit")
     - Uses original query text (before LLM translation) for German/French product names

   - **Step 1a: Text-based semantic search** (100 candidates, NO filters)
     - Uses `paraphrase-multilingual-MiniLM-L12-v2` embeddings
     - Searches with LLM-generated description stripped of constraint language
     - Description format: `MaterialMasterShorttext: FP ALOE VERA-PASSIONFRUCHT, Flavour: Aloe Vera, Passionfruit, Produktsegment: Trinkjoghurt`
     - Only country/version filters applied; no numerical/categorical filters

   - **Step 1a-: Name-line semantic search** (200 candidates)
     - Searches with just the best product name line from the original query
     - Catches recipes missed by the full description search due to noise from constraint text

   - **Step 1a-f: Flavor + MST-format variant searches** (50 candidates each)
     - Uses LLM-extracted flavor terms (which correct user typos) to build multiple search variants:
       - Plain flavor terms (e.g., "Aloe Vera Passionfruit")
       - MST-format with product prefixes (e.g., "FP ALOE VERA PASSIONFRUIT", "FZ ALOE VERA-PASSIONFRUIT")
       - German translations (e.g., "Aloe Vera Passionfrucht") using a built-in English→German flavor dictionary
       - German MST-format variants (e.g., "FP ALOE VERA-PASSIONFRUCHT")
     - Bridges the language gap between English queries and German-indexed recipes

   - **Step 1a+: Original query text search** (100 candidates, if different from translated)
     - Searches with original user query (e.g., "FZ Orange Mango Grüner Tee")
     - Finds German/French recipe names that semantic search might miss
     - Merges additional candidates with deduplication

   - **Flavor Safeguard**:
     - Additional vector search for extracted flavor terms (100 candidates)
     - Includes German translations of flavor keywords (e.g., "Passionfruit" → "Passionfrucht")
     - Ensures flavor-specific recipes aren't missed by embedding similarity

   - **Step 1b: Apply filters IN-MEMORY** on the combined candidate pool
     - Progressive relaxation if strict filters return too few results:
       - Level 0: Strict filters (exact match)
       - Level 1: 10% tolerance on numerical filters
       - Level 2: 20% tolerance on numerical filters
       - Level 3: Skip numerical filters entirely (categorical only)
       - Level 4: No filters at all (returns unfiltered results with warning)

   - **Step 2: Feature scoring and ranking**
     - Calculates feature similarity scores for all candidates using feature encoder
     - Combines text similarity + feature scores
     - Returns top N results

2. **Filtering** (applied in-memory on text search candidates):
   - **Version filter**: P (Production) or L (Legacy) — applied during Qdrant search
   - **Country filter**: Single country or multi-country selection (uses `MatchAny`) — applied during Qdrant search
   - **Numerical range filters**: Applied in-memory on candidates (with progressive tolerance)
     - Uses `gt`, `gte`, `lt`, `lte` operators on fields like `Z_BRIX`, `Z_PH`, `Z_FRUCHTG`
   - **Categorical filters**: Applied in-memory on candidates (soft matching, missing = unknown)
     - Matches fields like `Z_INH04`, `Z_INH05`, `Z_INH12` against "Yes"/"No"/"Saccharose"

   **Filter Example** (Qdrant query):
   ```python
   # Brief: "Brix >40, no preservatives, allergen-free"
   filter_conditions = [
       FieldCondition(key="numerical.Z_BRIX", range=Range(gt=40.0)),
       FieldCondition(key="spec_fields.Z_INH04", match=MatchValue(value="No")),
       FieldCondition(key="spec_fields.Z_INH12", match=MatchValue(value="Yes")),
   ]
   ```

3. **Scoring**:
   - Text similarity score (from semantic search)
   - Feature search score (from feature encoder)
   - Flavor boost for flavor keyword matches
   - Name match boost (0.90–0.95 for exact/partial name matches)

4. **Translation**:
   - Detects query language (EN, DE, FR, etc.)
   - Translates field names using cached translations from database
   - Translates recipe values if needed

5. **Comparison Table Generation**:
   - Extracts all 60 specified fields for top 3 recipes
   - Orders fields according to `SPECIFIED_FIELDS_60` constant
   - Creates structured table with field codes, names, and values

**Output**:
```python
{
    "recipes": [
        {
            "recipe_name": "000000000000242036_PL10_06_P",
            "recipe_id": "...",
            "values": [None, "Yes", None, ...]  # 60 values in order
        },
        ...
    ],
    "field_definitions": [
        {"code": "Z_MAKTX", "en": "Material short text", "de": "Materialkurztext", "display": "Materialkurztext"},
        ...
    ],
    "has_data": True
}
```

### 3. **Response Generation**

The backend uses LLM to:
- Generate natural language explanation of results
- Create follow-up questions for refinement
- Format the response in the detected language

**Final Response**:
```json
{
    "response": "Ich habe 3 ähnliche Rezepte gefunden...",
    "followup_questions": ["1. Welche Geschmacksrichtungen...", ...],
    "comparison_table": {
        "field_definitions": [...],
        "recipes": [...],
        "has_data": true
    }
}
```

---

## Prerequisites

### Software Requirements

- **Docker** and **Docker Compose**
- **Python 3.9+** (for running scripts outside containers)
- **Git**
- **PostgreSQL 13+** (via Docker)
- **Qdrant** (via Docker)

### Data Requirements

- Recipe JSON files in `app/data/` directory
- Feature mapping file: `Test_Input/charactDescr_valueCharLong_map.json`
- Feature extraction mappings: `app/data/feature_extraction_mappings.json`

### API Keys

- **OpenAI API Key** (or Azure OpenAI) for LLM operations
- **Azure AD credentials** (if using SSO)

---

## Deployment on a New Server

### Step 1: Infrastructure Setup

1. **Clone the repository**:
```bash
git clone https://github.com/fredboneML/RECIPE_AGENT.git
cd RECIPE_AGENT/app
```

2. **Create environment file**:
```bash
cp ../env.example .env
# Edit .env with your configuration (see Environment Variables section)
```

3. **Create Docker network**:
```bash
docker network create app-network
```

4. **Configure Microsoft Defender for Endpoint (MDE)** (Azure servers only):

   **IMPORTANT**: These commands ensure the server is properly onboarded, protected, compliant, and audit-ready. This is the correct way to configure MDE for future servers.

   ```bash
   # Make script executable
   chmod +x ../enable-mde.sh
   
   # Run MDE configuration
   ../enable-mde.sh
   ```

   **What it does**:
   - Disables passive mode (enables active protection)
   - Enables real-time protection
   - Enables behavior monitoring
   - Restarts Microsoft Defender service
   - Verifies health status

   **Important notes** (read once):
   - ✅ **Safe to run multiple times** (idempotent)
   - ❌ **Do not run if another AV/EDR is intentionally installed**
   - ⏳ **Azure portals may take 30–120 min to reflect compliance**
   - 📌 **If passive mode reappears, it's being enforced by tenant policy**

5. **Deploy infrastructure** (PostgreSQL + Qdrant):
```bash
./deploy-infrastructure.sh
```

This will:
- Start PostgreSQL database
- Start Qdrant vector database
- Wait for both services to be ready

**Verify infrastructure**:
```bash
# Check PostgreSQL
docker-compose -f infrastructure-compose.yml exec database pg_isready -U ${POSTGRES_USER}

# Check Qdrant
curl http://localhost:6333/healthz
```

---

### Step 2: Database Initialization

The database is automatically initialized by the `db_init` container when infrastructure is deployed. However, you can also run it manually:

```bash
# Inside backend container or with Python environment
cd app/backend
python3 init_db.py
```

**What it does**:
- Creates all required tables:
  - `users`: User authentication
  - `user_memory`: Conversation history
  - `query_cache`: Query result caching
  - `recipe_translation_cache`: Cached recipe translations
  - `field_name_translation_cache`: Cached field name translations
  - `azure_ad_users`: Azure AD user tracking (if SSO enabled)
  - `azure_ad_group_mappings`: Group-to-role mappings (if SSO enabled)
- Creates indexes for performance
- Sets up triggers for automatic password hashing
- Creates initial admin and read-only users (from `.env`)

**Verify database**:
```bash
docker-compose -f infrastructure-compose.yml exec database psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "\dt"
```

---

### Step 3: Qdrant Vector Database Setup

Qdrant is started automatically with infrastructure. The collection is created during recipe indexing.

**Verify Qdrant**:
```bash
curl http://localhost:6333/collections
```

**Check collection status** (after indexing):
```bash
curl http://localhost:6333/collections/food_recipes_two_step | jq '{points: .result.points_count, status: .result.status}'
```

---

### Step 4: Recipe Indexing

**IMPORTANT**: Indexing 600K recipes can take several hours. Run this in a screen/tmux session.

#### Indexing Tools & Models

| Component | Tool/Model | Purpose |
|-----------|------------|---------|
| **Script** | `init_vector_index_qdrant.py` | Main indexing script |
| **Vector DB** | Qdrant | Stores vectors and payload for semantic search |
| **Embedding Model** | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim multilingual text embeddings |
| **Feature Normalizer** | `feature_normalizer.py` | Multilingual → English normalization |
| **Recipe Manager** | `EnhancedTwoStepRecipeManager` | Handles encoding and upserts |

#### Recipe Encoding Process

Before indexing, recipes undergo sophisticated encoding to ensure accurate search matching:

**1. Field Extraction (60 Specified Fields)**
- Extracts exactly 60 predefined fields (Z_MAKTX, Z_BRIX, Z_PH, Z_FRUCHTG, etc.)
- Handles missing fields gracefully (stores as `None` in payload)
- Maintains field order according to `SPECIFIED_FIELDS_ORDERED`

**2. Multilingual Feature Normalization**
- **Feature Name Normalization**: Converts German/French/etc. feature names to English
  - German "Stärke" → English "Starch"
  - German "Künstliche Farben" → English "Artificial colors"
  - Uses `FeatureNormalizer` with mappings from `feature_extraction_mappings.json`
- **Value Normalization**: Converts multilingual values to standardized English
  - German "Ja" / "Nein" → English "Yes" / "No"
  - German "Stärke enthalten" → English "Yes"
  - German "keine künstl. Farbe" → English "No"
  - French "Oui" / "Non" → English "Yes" / "No"
- **Purpose**: Enables English queries to match German/French-indexed recipes via vector similarity

**3. Binary Opposition Mapping**
- **Detects binary features**: Features with Yes/No, with/without, present/absent patterns
- **Maps oppositions**: 
  - "no sugar" → encoded as -1.0 (negative)
  - "sugar" → encoded as +1.0 (positive)
  - "preservative-free" → encoded as -1.0
  - "with preservatives" → encoded as +1.0
- **Multilingual support**: Handles German ("nein", "ohne"), French ("non", "sans"), etc.
- **Purpose**: Ensures "no preservatives" query matches "preservative-free" recipes

**4. Feature Type Detection**
- **Binary**: Yes/No, with/without patterns → encoded as -1.0, 0.0, or +1.0
- **Numerical**: Brix, pH, viscosity, percentages → normalized to [-1, 1] range
- **Categorical**: Flavors, colors, stabilizers → hash-based encoding (0-1 range)
- **Text**: Material descriptions → embedded using SentenceTransformer

**5. Vector Creation**

Each recipe is encoded into **two named vectors** in Qdrant:

**a) Text Vector (384 dimensions)**
- Created from searchable text: `"Material short text: FP Cherry Vanilla Drink, Standard product: Yes, Brix: 49.0, ..."`
- Uses `paraphrase-multilingual-MiniLM-L12-v2` embedding model
- Enables semantic similarity search for natural language queries

**b) Feature Vector (484 dimensions)**
- **First 384 dim**: Text embedding of normalized feature text
- **Last 100 dim**: Categorical encoding of features
  - Binary features: -1.0 (negative), 0.0 (missing), +1.0 (positive)
  - Numerical features: Normalized to [-1, 1] (e.g., Brix 50 → 0.5)
  - Categorical features: Hash-based encoding (0-1 range)
- Enables precise feature-based matching

**6. Payload Structure**

Each recipe point in Qdrant stores:
```json
{
  "recipe_name": "000000000000242036_PL10_06_P",
  "description": "Peach and apricot fruit preparation...",
  "country": "PL",
  "version": "P",
  "spec_fields": {
    "Z_MAKTX": "FP Cherry Vanilla Drink",
    "Z_INH01": "Yes",
    "Z_BRIX": null,  // Missing field
    ...
  },
  "numerical": {
    "Z_BRIX": 49.0,
    "Z_PH": 3.7,
    "Z_FRUCHTG": 61.6,
    ...
  },
  "feature_text": "Material short text: FP Cherry Vanilla Drink | Standard product: Yes | ...",
  "num_available": 30,
  "num_missing": 30
}
```

**7. Batch Processing**
- Processes recipes in configurable batches (default: 1000 files per batch)
- Uses `EnhancedTwoStepRecipeManager` for consistent encoding
- Analyzes feature patterns across batches to build binary opposition mappings
- Upserts to Qdrant in smaller batches (default: 100 points per upsert)

**Key Benefits**:
- ✅ **Multilingual search**: English queries match German/French recipes
- ✅ **Binary opposition**: "no sugar" matches "sugar-free" recipes
- ✅ **Consistent encoding**: Same logic used for indexing and search
- ✅ **Efficient storage**: Only stores available fields, marks missing as `None`

1. **Prepare recipe data**:
   - Place all recipe JSON files in `app/data/`
   - Ensure `Test_Input/charactDescr_valueCharLong_map.json` exists

2. **Run indexing script**:
```bash
cd app/backend

# Set environment variables
export QDRANT_HOST=localhost  # or 'qdrant' if in Docker network
export QDRANT_PORT=6333
export RECIPE_COLLECTION_NAME=food_recipes_two_step
export EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
export RECIPE_DATA_DIR=/path/to/app/data
export FEATURE_MAP_PATH=/path/to/Test_Input/charactDescr_valueCharLong_map.json
export FEATURE_MAPPINGS_PATH=/path/to/app/data/feature_extraction_mappings.json

# Run indexing
python3 init_vector_index_qdrant.py
```

**Or use Docker** (if running in container):
```bash
docker-compose -f infrastructure-compose.yml run --rm vector_index_init
```

**What it does**:
- Creates Qdrant collection `food_recipes_two_step` with named vectors:
  - `text`: 384-dimensional embeddings for semantic search
  - `features`: 484-dimensional feature vectors (384 text + 100 categorical)
- Extracts 60 specified fields from each recipe
- Normalizes multilingual features to English
- Applies binary opposition mapping
- Indexes recipes in batches (configurable batch size)
- Stores payload with:
  - `spec_fields`: All 60 specified fields (with `None` for missing)
  - `numerical`: Numerical values for range queries
  - `country`, `version`, `recipe_name`, `description`, etc.

**Monitor progress**:
```bash
# Check Qdrant collection size
curl -s http://localhost:6333/collections/food_recipes_two_step | jq '.result.points_count'

# Check logs
docker-compose -f infrastructure-compose.yml logs -f vector_index_init
```

---

### Step 5: Create Payload Indexes

**CRITICAL**: Without payload indexes, searches with numerical and categorical filters will timeout on large databases!

After indexing a significant portion of recipes (e.g., 200K+), create indexes:

```bash
cd app/backend
python3 create_payload_indexes.py localhost 6333
```

#### Indexes Created

The script creates **3 types of payload indexes** in Qdrant:

**1. FLOAT Indexes (for numerical range queries)**

| Index Path | Field | Purpose |
|------------|-------|---------|
| `numerical.Z_BRIX` | Brix | Sugar content filtering (e.g., >40, 25-35) |
| `numerical.Z_PH` | pH | Acidity filtering (e.g., <4.1) |
| `numerical.Z_FRUCHTG` | Fruit content % | Fruit percentage filtering |
| `numerical.Z_VISK20S` | Viscosity 20°C/s⁻¹ | Viscosity filtering |
| `numerical.Z_VISK4S` | Viscosity 4°C/s⁻¹ | Cold viscosity filtering |
| `numerical.Z_VISK60S` | Viscosity 60°C/s⁻¹ | Hot viscosity filtering |
| `numerical.Z_VISK70S` | Viscosity 70°C/s⁻¹ | Hot viscosity filtering |
| `numerical.Z_VERHAN` | Dilution | Dilution ratio filtering |
| `numerical.Z_TRMA` | Dry matter % | Dry matter filtering |
| `numerical.Z_PAST` | Pasteurization temp | Temperature filtering |
| `numerical.Z_HALTB` | Shelf life (months) | Shelf life filtering |

**2. KEYWORD Indexes (for categorical exact-match queries)**

| Index Path | Field | Typical Values |
|------------|-------|----------------|
| `spec_fields.Z_INH01` | Sugar type | Saccharose, Fructose, No sugar |
| `spec_fields.Z_INH04` | Preserved | Yes, No |
| `spec_fields.Z_INH05` | Artificial colors | Yes, No |
| `spec_fields.Z_INH06` | Artificial flavors | Yes, No |
| `spec_fields.Z_INH09` | Natural flavor | Yes, No |
| `spec_fields.Z_INH10` | GMO | positive, negative, neutral |
| `spec_fields.Z_INH12` | Allergen-free | Yes, No |
| `spec_fields.Z_INH13` | Sweetener | Yes, No |
| `spec_fields.Z_INH14` | Starch | Yes, No |
| `spec_fields.Z_INH15` | Halal | Yes, No |
| `spec_fields.Z_INH16` | Kosher | Yes, No |
| `spec_fields.Z_KON` | Consistency | Stückig, homogen |
| `spec_fields.Z_BIOPRO` | Organic | Yes, No |
| `spec_fields.Z_FAIR` | Fair Trade | Yes, No |
| `spec_fields.Z_VEGAN` | Vegan | Yes, No |
| `spec_fields.Z_VEGET` | Vegetarian | Yes, No |
| `spec_fields.Z_LMFRREI` | Lactose-free | Yes, No |
| `spec_fields.Z_GLUTFR` | Gluten-free | Yes, No |
| `version` | Recipe version | P (Production), L (Legacy) |
| `country` | Country code | DE, FR, PL, AT, etc. |
| `recipe_name` | Full recipe name | For exact name matching |

**3. TEXT Index (for full-text search)**

| Index Path | Purpose |
|------------|---------|
| `description` | Full-text search on recipe descriptions |

#### Verify Indexes

```bash
curl -s http://localhost:6333/collections/food_recipes_two_step | jq '.result.payload_schema'
```

You should see entries like:
```json
{
  "numerical.Z_BRIX": {"data_type": "Float", "params": {"type": "float", "is_tenant": false, ...}},
  "numerical.Z_PH": {"data_type": "Float", ...},
  "spec_fields.Z_INH04": {"data_type": "Keyword", ...},
  "spec_fields.Z_INH12": {"data_type": "Keyword", ...},
  "version": {"data_type": "Keyword", ...},
  "description": {"data_type": "Text", ...}
}
```

**Note**: Index creation can take time for large collections (5-15 minutes for 600K+ recipes). The script will wait for each index to be created.

---

### Step 6: Application Deployment

1. **Deploy application**:
```bash
cd app
./deploy-app.sh
```

This will:
- Build frontend and backend containers
- Start the application services
- Connect to the `app-network` Docker network

2. **Verify deployment**:
```bash
# Check backend health
curl http://localhost:8000/health

# Check frontend (if accessible)
curl http://localhost:3000
```

3. **View logs**:
```bash
docker-compose -f app-compose.yml logs -f
```

---

### Step 7: Nginx Reverse Proxy (Production)

**CRITICAL**: For production deployments with HTTPS and long-running AI queries, you need a reverse proxy with extended timeouts.

#### Complete Nginx Configuration

1. **Install nginx**:
```bash
sudo apt update
sudo apt install nginx
```

2. **Create nginx configuration**:
```bash
sudo nano /etc/nginx/sites-available/recipe-agent
```

3. **Paste the complete configuration** (replace `your-domain.com` with your actual domain):

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    
    # Allow Let's Encrypt renewals
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    # Redirect to HTTPS
    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL certificates (will be added by Certbot)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Frontend application (default location)
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
    
    # Backend API - routes /api/* to backend:8000/api/*
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # CRITICAL: Extended timeouts for AI/search queries (can take 60-120s)
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        
        # Large file uploads (briefs/documents up to 50MB)
        client_max_body_size 50M;
        
        # CORS headers (if backend doesn't handle them)
        add_header 'Access-Control-Allow-Origin' '$http_origin' always;
        add_header 'Access-Control-Allow-Credentials' 'true' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'Accept,Authorization,Cache-Control,Content-Type,DNT,If-Modified-Since,Keep-Alive,Origin,User-Agent,X-Requested-With' always;
        
        # Handle preflight requests
        if ($request_method = 'OPTIONS') {
            return 204;
        }
    }
    
    # WebSocket support - routes /ws to frontend:3000/ws
    location /ws {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

4. **Enable the site**:
```bash
sudo ln -sf /etc/nginx/sites-available/recipe-agent /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default  # Remove default site
```

5. **Obtain SSL certificates** (using Certbot):
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate (will auto-update nginx config)
sudo certbot --nginx -d your-domain.com
```

6. **Test and reload nginx**:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

#### Key Timeout Settings Explained

| Setting | Location | Value | Why |
|---------|----------|-------|-----|
| `proxy_read_timeout` | `/api` block | **300s** | AI/search queries can take 60-120s with 600K+ recipes |
| `proxy_connect_timeout` | `/api` block | **300s** | Backend may be slow to respond under load |
| `proxy_send_timeout` | `/api` block | **300s** | Large requests need time to upload |
| `client_max_body_size` | `/api` block | **50M** | Allow large document uploads (PDFs, DOCX) |
| `proxy_read_timeout` | `/ws` block | **86400s** | Keep WebSocket alive for 24 hours |

**⚠️ Important**: Without these timeout settings in the `/api` block, you will get **504 Gateway Timeout** errors when the AI search takes longer than 60 seconds (the default nginx timeout).

#### Option B: Using Docker nginx container

#### Option B: Using Docker nginx container

1. **Add nginx to docker-compose** (create `production-compose.yml`):
```yaml
services:
  nginx:
    build:
      context: .
      dockerfile: nginx.Dockerfile
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /etc/letsencrypt/live/your-domain:/etc/nginx/ssl:ro
    depends_on:
      - frontend_app
      - backend_app
    networks:
      - app-network
```

2. **Build and start**:
```bash
docker-compose -f production-compose.yml up -d nginx
```

#### Option C: Azure Application Gateway

If using Azure Application Gateway, configure the backend settings:

1. **Go to Azure Portal** → Application Gateway → Backend settings
2. **Set timeouts**:
   - Request timeout: **300 seconds** (5 minutes)
   - Connection draining timeout: 300 seconds

3. **Health probe settings**:
   - Path: `/health`
   - Interval: 30 seconds
   - Unhealthy threshold: 3

#### Key Timeout Settings

| Setting | Value | Why |
|---------|-------|-----|
| `proxy_read_timeout` | 300s | AI/search queries can take 60-120s |
| `proxy_connect_timeout` | 300s | Backend may be slow to respond |
| `client_max_body_size` | 50M | Large document uploads (PDFs, DOCX) |

---

### Step 8: User Management

### Option A: Add Users Without SSO (SQL)

1. **Connect to database**:
```bash
docker-compose -f infrastructure-compose.yml exec database psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}
```

2. **Insert user** (password will be auto-hashed by trigger):
```sql
-- Insert user (password will be auto-generated if not provided)
INSERT INTO users (username, password_hash, role) 
VALUES ('newuser', 'plaintext_password_here', 'read_only')
ON CONFLICT (username) DO NOTHING;

-- Or let trigger generate password
INSERT INTO users (username, role) 
VALUES ('newuser', 'read_only');

-- Get generated password
SELECT * FROM get_generated_password('newuser');
```

**Available roles**:
- `admin`: Full access
- `write`: Can create/edit recipes
- `read_only`: Read-only access

### Option B: Add Users With SSO (Azure AD)

1. **Follow Azure AD SSO Setup**:
   - See `app/docs/AZURE_AD_SSO_SETUP.md` for detailed instructions
   - Create Azure AD app registration
   - Configure security groups
   - Set up group-to-role mappings

2. **Configure environment variables**:
```bash
# In .env file
AZURE_AD_TENANT_ID=your-tenant-id
AZURE_AD_CLIENT_ID=your-client-id
SSO_ENABLED=true
LOCAL_AUTH_ENABLED=true  # Keep both enabled during migration
```

3. **Create group mappings in database**:
```sql
-- Insert group mappings (use Object IDs from Azure AD)
INSERT INTO azure_ad_group_mappings (group_id, group_name, app_role) VALUES
('group-object-id-1', 'Recipe-Agent-Admins', 'admin'),
('group-object-id-2', 'Recipe-Agent-Writers', 'write'),
('group-object-id-3', 'Recipe-Agent-Users', 'read_only');
```

4. **Users are automatically created** on first SSO login

**Verify SSO setup**:
```bash
# Check if tables exist
docker-compose -f infrastructure-compose.yml exec database psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "SELECT * FROM azure_ad_group_mappings;"
```

---

## Environment Variables

### Required Variables

```bash
# Database
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_DB=your_db_name
DB_HOST=database  # or 'localhost' if not in Docker
DB_PORT=5432

# OpenAI / Azure OpenAI
AI_ANALYZER_OPENAI_API_KEY=your_openai_api_key
# OR for Azure OpenAI:
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_MODEL_DEPLOYMENT=gpt-4o-mini

# JWT Authentication
JWT_SECRET_KEY=your_secret_key_here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440  # 24 hours

# Initial Users (for non-SSO)
ADMIN_USER=admin
ADMIN_PASSWORD=your_admin_password
READ_USER=readonly
READ_USER_PASSWORD=your_readonly_password

# Azure AD SSO (optional)
AZURE_AD_TENANT_ID=your-tenant-id
AZURE_AD_CLIENT_ID=your-client-id
SSO_ENABLED=true
LOCAL_AUTH_ENABLED=true

# Application
HOST=localhost  # or your domain
ALLOWED_ORIGINS=http://localhost:3000,https://your-domain.com
```

### Optional Variables

```bash
# Qdrant (defaults work for Docker)
QDRANT_HOST=qdrant
QDRANT_PORT=6333
RECIPE_COLLECTION_NAME=food_recipes_two_step

# Embedding Model
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Data Directories
RECIPE_DATA_DIR=/usr/src/app/data
FEATURE_MAP_PATH=/usr/src/app/Test_Input/charactDescr_valueCharLong_map.json
```

---

## Troubleshooting

### 504 Gateway Timeout / UI Cancels Request

**Problem**: The backend completes processing (visible in logs), but the UI shows "Error: Failed to submit query" or the request is cancelled. Browser console shows `504` status.

**Causes**:
1. **Reverse proxy timeout too short** (default nginx: 60s, default Azure Application Gateway: 30s)
2. **Complex search queries** can take 60-120 seconds with 600K+ recipes

**Solution**:

**Option 1: Nginx** - Increase timeout in nginx config:
```nginx
# In /etc/nginx/sites-available/recipe-agent or nginx.conf
proxy_read_timeout 300s;
proxy_send_timeout 300s;
proxy_connect_timeout 300s;
```
Then reload: `sudo nginx -s reload`

**Option 2: Azure Application Gateway**:
1. Go to Azure Portal → Application Gateway → Backend settings
2. Set **Request timeout: 300 seconds** (5 minutes)

**Verify backend is completing** (check logs show response):
```bash
docker-compose -f app-compose.yml logs -f backend_app | grep "Returning response"
```

If you see `Returning response for conversation_id:` but UI doesn't show results, it's definitely a proxy timeout issue.

### WebSocket Connection Failed

**Problem**: Browser console shows `WebSocket connection to 'wss://...:3000/ws' failed: net::ERR_CONNECTION_TIMED_OUT`

**Causes**:
1. Port 3000 not exposed in firewall
2. WebSocket not configured in reverse proxy

**Solution**: Add WebSocket support to nginx:
```nginx
location /ws {
    proxy_pass http://backend:8000/ws;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400s;
}
```

For Azure Application Gateway, ensure WebSocket is enabled in HTTP settings.

---

### Searches Timeout or Return Empty Results

**Problem**: Searches with numerical filters (Brix >40, pH <4.1) timeout after 60 seconds.

**Solution**: Create payload indexes (see [Step 5: Create Payload Indexes](#step-5-create-payload-indexes))

```bash
cd app/backend
python3 create_payload_indexes.py localhost 6333
```

### Qdrant Collection Not Found

**Problem**: `Collection 'food_recipes_two_step' doesn't exist`

**Solution**: Run recipe indexing (see [Step 4: Recipe Indexing](#step-4-recipe-indexing))

### Database Connection Errors

**Problem**: Backend cannot connect to PostgreSQL

**Solution**:
1. Check if database container is running: `docker ps | grep postgres`
2. Verify network: `docker network ls | grep app-network`
3. Check environment variables: `echo $POSTGRES_USER $POSTGRES_PASSWORD`
4. Test connection: `docker-compose -f infrastructure-compose.yml exec database pg_isready`

### Frontend Shows Old Comparison Table Format

**Problem**: UI still shows "Characteristic | Value" instead of "Code | Field Name | Recipe 1 | Recipe 2 | Recipe 3"

**Solution**:
1. Rebuild frontend: `docker-compose -f app-compose.yml build --no-cache frontend_app`
2. Restart: `docker-compose -f app-compose.yml restart frontend_app`
3. Hard refresh browser (Ctrl+Shift+R or Cmd+Shift+R)

### SSO Not Working

**Problem**: "Sign in with Microsoft" button doesn't appear or authentication fails

**Solution**:
1. Verify environment variables: `echo $AZURE_AD_TENANT_ID $AZURE_AD_CLIENT_ID`
2. Check SSO enabled: `echo $SSO_ENABLED` (should be `true`)
3. Verify Azure AD app registration redirect URI matches your domain
4. Check browser console for JavaScript errors
5. See `app/docs/AZURE_AD_SSO_SETUP.md` for detailed troubleshooting

### Indexing Fails or Is Slow

**Problem**: Recipe indexing fails or takes too long

**Solution**:
1. Check available disk space: `df -h`
2. Monitor Qdrant memory: `docker stats qdrant`
3. Reduce batch size in `init_vector_index_qdrant.py`:
   ```python
   FILE_BATCH_SIZE = 100  # Default: 1000
   QDRANT_UPSERT_BATCH_SIZE = 50  # Default: 100
   ```
4. Check logs: `docker-compose -f infrastructure-compose.yml logs vector_index_init`

### Translation Cache Issues

**Problem**: Field names or recipe values not translating

**Solution**:
1. Check database tables exist:
   ```sql
   SELECT * FROM field_name_translation_cache LIMIT 1;
   SELECT * FROM recipe_translation_cache LIMIT 1;
   ```
2. Clear cache if needed:
   ```sql
   DELETE FROM field_name_translation_cache;
   DELETE FROM recipe_translation_cache;
   ```
3. Translations will be regenerated on next query

### Product Name Search Not Finding Expected Recipe

**Problem**: Searching for a recipe name doesn't find the expected result (e.g., "ALOE VERA PASSIONFRUCHT" not finding "FP ALOE VERA-PASSIONFRUCHT TJ")

**Possible Causes**:
1. **Hyphen/space mismatch**: Indexed name uses hyphens (e.g., "VERA-PASSIONFRUCHT") but query uses spaces — now handled by normalization
2. **Language gap**: Query uses English ("Passionfruit") but index uses German ("Passionfrucht") — now handled by German translation variants in Step 1a-f
3. **Typos/truncations**: User typed "passionfruch" instead of "Passionfrucht" — now handled by fuzzy matching in Step 0 and LLM-corrected flavor terms in Step 1a-f
4. **Missing product prefix**: User typed "ALOE VERA" but index has "FP ALOE VERA" — now handled by MST-format variant searches with FP/FZ prefixes
5. Recipe might be in a different country/version than filtered
6. Recipe might not be indexed yet

**Solution**:
1. Verify recipe exists in Qdrant:
   ```bash
   curl -X POST "http://localhost:6333/collections/food_recipes_two_step/points/scroll" \
     -H "Content-Type: application/json" \
     -d '{"filter":{"must":[{"key":"description","match":{"text":"ALOE VERA"}}]},"limit":10,"with_payload":true}'
   ```

2. Check country filter — the recipe might be indexed under a different country

3. Check logs for the full search flow:
   ```
   Step 0 (name match): Exact/partial/fuzzy name matching
   Step 0 fuzzy: Word-overlap fallback with LLM-corrected flavors
   Step 1a: Text-based semantic search (100 candidates)
   Step 1a-: Name-line semantic search (200 candidates)
   Step 1a-f: Flavor + MST-format variant searches (German translations, FP/FZ prefixes)
   Step 1a+: Original query search (untranslated)
   Flavor safeguard: Additional flavor keyword search (English + German)
   Step 1b: Filter application with progressive relaxation
   ```

**Note**: The search architecture includes multiple redundant search paths to handle typos, language gaps, hyphen/space differences, and missing product prefixes. If a recipe still isn't found, check the backend logs for each step to see which variants were searched and what scores the target recipe received.

---

## Additional Resources

- **API Documentation**: `app/docs/api_documentation.md`
- **Backend Documentation**: `app/docs/backend_documentation.md`
- **Frontend Documentation**: `app/docs/frontend_documentation.md`
- **Azure AD SSO Setup**: `app/docs/AZURE_AD_SSO_SETUP.md`
- **Project Overview**: `app/docs/project_overview.md`

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the documentation in `app/docs/`
3. Check application logs: `docker-compose -f app-compose.yml logs -f`

---

## License

[Your License Here]
