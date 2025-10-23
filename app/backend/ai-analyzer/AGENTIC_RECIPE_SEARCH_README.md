# Agentic Recipe Search System

## Overview

This document describes the new **Agentic Recipe Search System** that replaces the non-agentic approach with an intelligent multi-agent system. The system uses four coordinated agents to provide intelligent recipe search with natural language explanations.

## Architecture

### Four Agent System

The agentic system consists of four specialized agents:

#### 1. **Data Extractor & Router Agent** (`data_extractor_router.py`)
- **Purpose**: Extracts meaningful information from supplier project briefs and determines the optimal search strategy
- **Responsibilities**:
  - Parse supplier briefs to extract key recipe characteristics (flavors, colors, ingredients, attributes)
  - Identify structured feature requirements vs. descriptive text
  - Route to either:
    - **Text-only search**: For descriptive, sensory-focused briefs
    - **Two-step search**: For briefs with specific feature requirements
  - Provide reasoning for routing decision

#### 2. **Search & Reranker Agent** (`search_reranker.py`)
- **Purpose**: Executes the appropriate search strategy and finds top K similar recipes
- **Responsibilities**:
  - Execute text-only search (using description similarity)
  - Execute two-step search (text similarity + feature refinement)
  - Apply custom reranking criteria if provided
  - Return ranked results with similarity scores

#### 3. **Similar Recipe Generator Agent** (`recipe_generator.py`)
- **Purpose**: Presents search results and explains similarities in natural language
- **Responsibilities**:
  - Generate structured results table
  - Create natural language explanations of why recipes match
  - Highlight key matching features and characteristics
  - Provide actionable insights for product development

#### 4. **Recipe Search Manager Agent** (`recipe_search_manager.py`)
- **Purpose**: Coordinates all three specialized agents and manages the workflow
- **Responsibilities**:
  - Initialize and manage all specialized agents
  - Orchestrate the three-stage workflow:
    - Stage 1: Data extraction and routing
    - Stage 2: Search and reranking
    - Stage 3: Result generation and explanation
  - Validate setup and handle errors
  - Provide comprehensive output with intermediate results

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Supplier Project Brief                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Data Extractor & Router Agent                         │
│  ─────────────────────────────────────────────────────────────  │
│  • Extracts key recipe information                               │
│  • Decides: Text-only OR Two-step search                        │
│  • Provides routing reasoning                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Search & Reranker Agent                               │
│  ─────────────────────────────────────────────────────────────  │
│  • Executes appropriate search strategy                          │
│  • Finds top K similar recipes                                   │
│  • Applies optional reranking                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Similar Recipe Generator Agent                        │
│  ─────────────────────────────────────────────────────────────  │
│  • Generates structured results table                            │
│  • Creates natural language explanation                          │
│  • Provides actionable insights                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Final Results with Explanation               │
│  • Results table with similarity scores                          │
│  • Natural language explanation                                  │
│  • Metadata and intermediate outputs                             │
└─────────────────────────────────────────────────────────────────┘
```

## Files Created

### Core Agent Files
1. **`ai_analyzer/agents/data_extractor_router.py`**
   - Data Extractor & Router Agent implementation
   
2. **`ai_analyzer/agents/search_reranker.py`**
   - Search & Reranker Agent implementation
   
3. **`ai_analyzer/agents/recipe_generator.py`**
   - Similar Recipe Generator Agent implementation
   
4. **`ai_analyzer/agents/recipe_search_manager.py`**
   - Manager Agent that coordinates all three agents
   
5. **`ai_analyzer/agents/recipe_search_workflow.py`**
   - High-level workflow for recipe search operations

### Infrastructure Files
6. **`backend/init_vector_index.py`**
   - Initializes Qdrant vector index and indexes recipes
   - Runs at startup in Docker (similar to `init_db.py`)
   
7. **`examples/agentic_recipe_search_demo.py`**
   - Demonstration script showing how to use the system
   - Includes two examples:
     - Example 1: Text-only search
     - Example 2: Two-step search with features

### Updated Files
8. **`ai_analyzer/agents/factory.py`**
   - Added methods to create recipe search agents:
     - `create_recipe_search_manager()`
     - `create_data_extractor_router_agent()`
     - `create_search_reranker_agent()`
     - `create_recipe_generator_agent()`

9. **`ai_analyzer/agents/__init__.py`**
   - Added exports for new recipe search agents

10. **`app/infrastructure-compose.yml`**
    - Added `vector_index_init` service
    - Initializes vector index before starting other services

## Usage

### Basic Usage

```python
from ai_analyzer.agents.recipe_search_workflow import RecipeSearchWorkflow

# Initialize workflow
workflow = RecipeSearchWorkflow(
    collection_name="food_recipes_two_step",
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    default_top_k=3
)

# Load recipes
workflow.load_recipes(
    features_list=features_list,
    values_list=values_list,
    descriptions_list=descriptions_list,
    metadata_list=metadata_list
)

# Execute search
results = workflow.search(
    supplier_brief=supplier_brief_text,
    top_k=3,
    save_results=True
)

# Access results
print(results['results_table'])  # DataFrame with results
print(results['explanation'])    # Natural language explanation
print(results['summary'])         # Summary statistics
```

### Advanced Usage with Custom Reranking

```python
# Custom reranking criteria
custom_reranking = {
    'text_score': 0.4,
    'feature_score': 0.6
}

results = workflow.search(
    supplier_brief=supplier_brief_text,
    top_k=5,
    custom_reranking=custom_reranking
)
```

### Direct Manager Usage

```python
from ai_analyzer.agents.recipe_search_manager import run_agentic_recipe_search
from src.two_step_recipe_search import EnhancedTwoStepRecipeManager

# Initialize recipe manager
recipe_manager = EnhancedTwoStepRecipeManager(
    collection_name="food_recipes_two_step",
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
)

# Load recipes...
recipe_manager.update_recipes(...)

# Run search
results = run_agentic_recipe_search(
    supplier_brief=supplier_brief_text,
    recipe_manager=recipe_manager,
    top_k=3
)
```

## Example Outputs

### Results Table
```
Rank  Recipe_Name                  Text_Similarity  Feature_Similarity  Combined_Score  Description
1     521082_FIT_BANANA           0.8523           0.7891              0.8207          Recipe Name: 521082 FIT BANANA...
2     521124_FRUIT_CHO_BANANA     0.8201           0.7456              0.7829          Recipe Name: FRUIT - CHO BANANA...
3     521173_FIT_BANANA_HERCULES  0.7989           0.7234              0.7612          Recipe Name: 521173 FRUIT...
```

### Natural Language Explanation
```markdown
## Recipe Search Results (Two Step)

Based on your supplier brief for a banana-flavored fruit preparation, I've identified the top 3 matching recipes:

### 1. 521082 FIT BANANA (Similarity: 0.8207)

This recipe is the closest match to your requirements. Key matching characteristics include:
- **Flavor**: Natural banana flavor without artificial additives
- **Stabilizers**: Uses Starch, LBG, and Pectin (matching your stabilization system)
- **Product Claims**: No preservatives, no artificial colors, natural flavors only
- **Certifications**: Halal and Kosher certified
- **Allergens**: Milk-containing products (as specified)

The high similarity score (0.8207) indicates excellent alignment with your brief...

[Continues with detailed analysis of other recipes]
```

## Vector Index Initialization

The system automatically initializes the vector index when Docker starts:

1. `vector_index_init` service runs `init_vector_index.py`
2. Creates Qdrant collection `food_recipes_two_step`
3. Loads recipes from `/app/backend/ai-analyzer/data/`
4. Indexes recipes with text embeddings and feature vectors
5. If data folder is empty, creates empty index for later use

## Key Features

### Intelligent Routing
- Automatically detects whether to use text-only or two-step search
- Provides reasoning for routing decisions
- Adapts to different types of supplier briefs

### Dual Search Strategy
- **Text-only**: Fast, semantic similarity search for descriptive briefs
- **Two-step**: Combines text similarity with feature matching for precise requirements

### Natural Language Explanations
- Clear explanations of why recipes match
- Highlights key matching features
- Provides actionable insights for product development

### Flexible Configuration
- Configurable top K results
- Custom reranking criteria
- Multiple model providers (OpenAI, Groq, etc.)
- Adjustable embedding models

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Qdrant Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Recipe Search Configuration
RECIPE_COLLECTION_NAME=food_recipes_two_step
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
RECIPE_DATA_DIR=/usr/src/app/ai-analyzer/data

# AI Model Configuration (existing)
AI_ANALYZER_OPENAI_API_KEY=your_api_key
```

## Running the Demo

```bash
# Inside the container
python /usr/src/app/ai-analyzer/examples/agentic_recipe_search_demo.py
```

This will run two examples:
1. Text-only search with a descriptive supplier brief
2. Two-step search with structured features

Results are saved to Excel files:
- `example_1_text_only_results.xlsx`
- `example_2_two_step_results.xlsx`

## Advantages Over Non-Agentic Approach

1. **Intelligence**: Agents automatically determine the best search strategy
2. **Explainability**: Natural language explanations of results
3. **Flexibility**: Easy to add new agents or modify behavior
4. **Maintainability**: Modular design with clear separation of concerns
5. **Extensibility**: Can add more agents (e.g., quality checker, compliance validator)

## Next Steps

### Potential Enhancements
1. Add a **Quality Validator Agent** to check recipe compliance
2. Add a **Feature Extraction Enhancer** for better feature detection
3. Implement **Multi-language Support** for international briefs
4. Add **Batch Processing** for multiple briefs at once
5. Create **Web API** for easy integration

### Cron Job Integration
- Create scheduled task to index new recipes automatically
- Update vector index when new recipes are added
- Monitor index health and performance

## Troubleshooting

### Common Issues

**Issue**: No recipes loaded
- **Solution**: Check that JSON files exist in `/app/backend/ai-analyzer/data/`

**Issue**: Vector index not found
- **Solution**: Ensure `vector_index_init` service completed successfully

**Issue**: Low similarity scores
- **Solution**: Verify recipe descriptions are comprehensive and match search query format

**Issue**: Agent initialization fails
- **Solution**: Check API keys and model availability

## Support

For questions or issues, check:
1. Agent logs: Look for detailed error messages
2. Intermediate outputs: Review agent decisions at each stage
3. Validation results: Use `manager.validate_setup()` to check configuration

## Summary

The Agentic Recipe Search System provides an intelligent, explainable, and flexible approach to finding similar recipes. By coordinating four specialized agents, it can understand complex supplier briefs, execute appropriate search strategies, and provide clear explanations of results—making it a powerful tool for product development teams.

