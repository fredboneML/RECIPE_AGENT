#!/usr/bin/env python3
"""
Similar Recipe Generator Agent

This agent is responsible for:
1. Displaying search results in a structured table format
2. Explaining in natural language which recipes are most similar
3. Providing reasoning for similarity rankings
"""
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from ai_analyzer.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecipeGeneratorAgent:
    """Agent for generating and explaining recipe search results"""

    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4", api_key: Optional[str] = None):
        """Initialize the Recipe Generator Agent"""
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key or config.get('AI_ANALYZER_OPENAI_API_KEY')

        # Create the agent
        self.agent = Agent(
            name="Similar Recipe Generator",
            role="Present recipe search results and explain similarities in natural language",
            model=OpenAIChat(
                api_key=self.api_key
            ),
            instructions=[
                "You are an expert recipe analyst who explains recipe similarities in clear, natural language.",
                "Your task is to:",
                "1. Review the search results and similarity scores",
                "2. Explain which recipes are most similar to the search query and why",
                "3. Highlight key matching features and characteristics",
                "4. Provide actionable insights about the recommended recipes",
                "",
                "EXPLANATION GUIDELINES:",
                "- Start with an overview of the search results",
                "- For each top recipe, explain:",
                "  * What makes it similar to the search query",
                "  * Key matching features (flavor, color, ingredients, attributes)",
                "  * Any notable differences or unique characteristics",
                "- Use clear, professional language suitable for product development teams",
                "- Focus on actionable insights that help decision-making",
                "- Be specific about matching criteria (e.g., 'Both contain natural peach flavor')",
                "",
                "OUTPUT STRUCTURE:",
                "1. Search Summary (brief overview of what was searched)",
                "2. Top Recommendations (ranked list with explanations)",
                "3. Key Insights (overall patterns and recommendations)",
                "",
                "Keep explanations concise but informative."
            ],
            markdown=True
        )

        logger.info(
            f"Initialized RecipeGeneratorAgent with model {self.model_name}")

    def generate_results_table(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate a structured table of search results

        Args:
            results: List of search results from SearchRerankerAgent

        Returns:
            DataFrame with formatted results
        """
        if not results:
            logger.warning("No results to display")
            return pd.DataFrame()

        try:
            # Extract key information for table
            table_data = []

            for i, result in enumerate(results, 1):
                row = {
                    'Rank': i,
                    'Recipe_Name': result.get('metadata', {}).get('recipe_name', f'Recipe_{i}'),
                    'Text_Similarity': round(result.get('text_score', 0), 4),
                    'Feature_Similarity': round(result.get('feature_score', 0), 4) if result.get('feature_score') else 'N/A',
                    'Combined_Score': round(result.get('combined_score', result.get('text_score', 0)), 4),
                    'Description': result.get('description', '')[:200] + '...' if len(result.get('description', '')) > 200 else result.get('description', ''),
                    'Num_Features': len(result.get('features', []))
                }
                table_data.append(row)

            df = pd.DataFrame(table_data)
            logger.info(f"Generated results table with {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error generating results table: {e}")
            return pd.DataFrame()

    def generate_explanation(self,
                             supplier_brief: str,
                             search_type: str,
                             results: List[Dict[str, Any]],
                             metadata: Dict[str, Any]) -> str:
        """
        Generate natural language explanation of search results

        Args:
            supplier_brief: Original supplier brief
            search_type: Type of search performed ('text_only' or 'two_step')
            results: List of search results
            metadata: Search metadata

        Returns:
            Natural language explanation
        """
        if not results:
            return "No similar recipes were found for the given search criteria."

        try:
            # Prepare context for the agent
            results_summary = self._prepare_results_summary(
                results, search_type)

            prompt = f"""
Analyze the following recipe search results and provide a clear explanation:

ORIGINAL SEARCH QUERY:
{supplier_brief[:500]}...

SEARCH METHOD: {search_type.replace('_', ' ').title()}

SEARCH RESULTS:
{results_summary}

SEARCH METADATA:
- Total results found: {len(results)}
- Search strategy: {metadata.get('search_strategy', search_type)}
- Text candidates evaluated: {metadata.get('text_results_found', 'N/A')}
- Feature refinement applied: {metadata.get('refinement_completed', False)}

Provide a comprehensive explanation of these results, including:
1. A brief overview of what was searched
2. Detailed explanation of each top recipe and why it matches
3. Key insights and recommendations

Use clear, professional language suitable for product development teams.
"""

            logger.info("Generating natural language explanation...")
            response = self.agent.run(prompt)

            explanation = str(response.content) if hasattr(
                response, 'content') else str(response)
            logger.info("Explanation generated successfully")

            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            # Fallback to basic summary
            return self._generate_fallback_explanation(results, search_type)

    def _prepare_results_summary(self, results: List[Dict[str, Any]], search_type: str) -> str:
        """Prepare a summary of results for the agent"""
        summary_lines = []

        for i, result in enumerate(results[:5], 1):  # Top 5 for context
            recipe_name = result.get('metadata', {}).get(
                'recipe_name', f'Recipe_{i}')
            text_score = result.get('text_score', 0)
            feature_score = result.get('feature_score', 0) if result.get(
                'feature_score') else None
            combined_score = result.get('combined_score', text_score)
            description = result.get('description', '')[:300]

            summary = f"\n{i}. {recipe_name}\n"
            summary += f"   - Combined Score: {combined_score:.4f}\n"
            summary += f"   - Text Similarity: {text_score:.4f}\n"
            if feature_score is not None:
                summary += f"   - Feature Similarity: {feature_score:.4f}\n"
            summary += f"   - Description: {description}\n"

            # Add key features
            features = result.get('features', [])[:10]  # Top 10 features
            values = result.get('values', [])[:10]
            if features:
                summary += f"   - Key Features:\n"
                for feat, val in zip(features[:5], values[:5]):  # Show top 5
                    summary += f"     * {feat}: {val}\n"

            summary_lines.append(summary)

        return "\n".join(summary_lines)

    def _generate_fallback_explanation(self, results: List[Dict[str, Any]], search_type: str) -> str:
        """Generate a basic fallback explanation if agent fails"""
        explanation = f"## Recipe Search Results ({search_type.replace('_', ' ').title()})\n\n"
        explanation += f"Found {len(results)} similar recipes.\n\n"

        for i, result in enumerate(results, 1):
            recipe_name = result.get('metadata', {}).get(
                'recipe_name', f'Recipe_{i}')
            score = result.get('combined_score', result.get('text_score', 0))

            explanation += f"### {i}. {recipe_name}\n"
            explanation += f"**Similarity Score:** {score:.4f}\n\n"

            description = result.get('description', '')
            if description:
                explanation += f"{description[:300]}...\n\n"

        return explanation

    def format_complete_output(self,
                               table_df: pd.DataFrame,
                               explanation: str,
                               metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format complete output with table and explanation

        Args:
            table_df: Results DataFrame
            explanation: Natural language explanation
            metadata: Search metadata

        Returns:
            Dictionary with formatted output
        """
        return {
            "results_table": table_df,
            "explanation": explanation,
            "metadata": metadata,
            "summary": {
                "total_results": len(table_df),
                "search_type": metadata.get('search_strategy', 'unknown'),
                "top_recipe": table_df.iloc[0]['Recipe_Name'] if len(table_df) > 0 else None,
                "top_score": float(table_df.iloc[0]['Combined_Score']) if len(table_df) > 0 else 0.0
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "agent_name": "RecipeGeneratorAgent",
            "model_provider": self.model_provider,
            "model_name": self.model_name
        }
