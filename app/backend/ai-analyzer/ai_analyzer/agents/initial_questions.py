# File: backend/ai-analyzer/ai_analyzer/agents/initial_questions.py

from typing import List, Dict, Any, Optional
import json
import logging
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from .base import BaseAgent, AgentResponse, DatabaseContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InitialQuestionGenerator(BaseAgent):
    """Generates initial questions and categories for users"""

    async def process(self, db_context: DatabaseContext) -> AgentResponse:
        """Generate relevant questions based on database structure"""
        try:

            # Create a chat prompt template with enhanced focus on issues and insights
            prompt = ChatPromptTemplate.from_template("""
                As a call center manager analyzing customer interactions, examine this database schema:
                {schema}
                
                Generate specific, actionable questions that will reveal key insights about customer issues
                and call center performance. Focus especially on:
                
                1. Key issues within popular topics
                2. Sentiment patterns and customer satisfaction metrics
                3. Specific trends and changes that matter to managers
                4. Opportunities for immediate operational improvements
                5. Quantifiable metrics that show performance
                
                Generate insightful questions for these categories (include percentages and specific metrics):
                
                - Topic Issues: Questions about specific problems within popular topics
                - Sentiment Insights: Questions about customer satisfaction with quantifiable metrics
                - Trend Analysis: Questions about measurable changes over specific time periods
                - Customer Experience: Questions about specific pain points in customer interactions
                - Performance Metrics: Questions about operational effectiveness with clear KPIs

                Return the response in this exact JSON format (keep the exact category names):
                {{
                    "Topic Issues": {{
                        "description": "Analyze specific problems within call topics",
                        "questions": [
                            "What are the top 5 specific issues mentioned in calls and what percentage of calls mention each?",
                            "Which issues within technical support have the highest negative sentiment percentage?",
                            "What specific customer complaints appear most frequently in billing-related calls?"
                        ]
                    }},
                    "Sentiment Insights": {{
                        "description": "Quantify customer satisfaction patterns",
                        "questions": [
                            "What percentage of calls have positive vs. negative sentiment in the last 30 days?",
                            "Which topics have shown the greatest improvement in sentiment scores over the past month?",
                            "What are the top 3 topics with the highest negative sentiment rate and their percentages?"
                        ]
                    }},
                    "Trend Analysis": {{
                        "description": "Measure important changes over time",
                        "questions": [
                            "How have call volumes for our top 5 topics changed compared to last month (show % change)?",
                            "Which topics have shown the most significant growth in the past two weeks?",
                            "What's the week-over-week trend in customer satisfaction for billing issues?"
                        ]
                    }},
                    "Customer Experience": {{
                        "description": "Identify key customer pain points",
                        "questions": [
                            "What specific issues lead to the longest call durations?",
                            "Which customer problems are most likely to require multiple callbacks?",
                            "What percentage of customer complaints relate to product functionality vs. service issues?"
                        ]
                    }},
                    "Performance Metrics": {{
                        "description": "Analyze operational effectiveness with KPIs",
                        "questions": [
                            "What's our average call resolution rate across different topics?",
                            "Which topics have the highest percentage of calls lasting over 10 minutes?",
                            "How does the distribution of call sentiment vary by time of day?"
                        ]
                    }}
                }}
                
                Only use tables: {allowed_tables}
                All questions should be specific, actionable, and focused on revealing insights that a call center manager would need.
                Questions should be phrased to elicit numerical results and percentages where possible.
                Return valid JSON only.
            """)

            # Create and invoke the chain
            chain = prompt | self.llm
            result = await chain.ainvoke({
                "schema": json.dumps(db_context.table_schemas, indent=2),
                "allowed_tables": ", ".join(db_context.allowed_tables)
            })

            # Extract content from the response
            content = result.content if hasattr(
                result, 'content') else str(result)

            # Parse the structured response
            categories = self._parse_categories(content)

            return AgentResponse(
                success=True,
                content=categories
            )
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return AgentResponse(
                success=False,
                content={},
                error_message=str(e)
            )

    def _parse_categories(self, result: str) -> Dict[str, Dict[str, Any]]:
        """Parse the JSON response into structured categories"""
        try:
            # Extract JSON from the response if needed
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
            else:
                json_str = result

            # Parse the JSON
            parsed = json.loads(json_str)

            # Validate the structure
            required_categories = [
                "Trending Topics", "Sentiment Analysis", "Time-based Patterns",
                "Customer Experience", "Performance Metrics"
            ]

            # Ensure all required categories exist
            for category in required_categories:
                if category not in parsed:
                    parsed[category] = {
                        "description": f"Analysis of {category.lower()}",
                        "questions": [
                            f"What are the key {category.lower()}?",
                            f"How have {category.lower()} changed recently?",
                            f"Which {category.lower()} need attention?"
                        ]
                    }

            return parsed
        except Exception as e:
            logger.error(f"Error parsing categories JSON: {e}")
            return {}
