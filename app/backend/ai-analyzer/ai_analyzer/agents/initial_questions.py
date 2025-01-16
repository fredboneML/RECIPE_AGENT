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
            # Create a chat prompt template with properly escaped JSON example
            prompt = ChatPromptTemplate.from_template("""
                As a call center data analyst, analyze this database schema:
                {schema}
                
                Generate insightful questions for these categories:
                - Trending Topics: Questions about popular discussion topics and emerging issues
                - Sentiment Analysis: Questions about customer satisfaction and feedback
                - Time-based Patterns: Questions about trends and changes over time
                - Customer Experience: Questions about specific customer interactions
                - Performance Metrics: Questions about operational effectiveness

                Return the response in this exact JSON format (keep the exact category names):
                {{
                    "Trending Topics": {{
                        "description": "Analyze popular discussion topics",
                        "questions": ["Question 1", "Question 2", "Question 3"]
                    }},
                    "Sentiment Analysis": {{
                        "description": "Understand sentiment patterns",
                        "questions": ["Question 1", "Question 2", "Question 3"]
                    }},
                    "Time-based Patterns": {{
                        "description": "Track changes over time",
                        "questions": ["Question 1", "Question 2", "Question 3"]
                    }},
                    "Customer Experience": {{
                        "description": "Analyze customer interactions",
                        "questions": ["Question 1", "Question 2", "Question 3"]
                    }},
                    "Performance Metrics": {{
                        "description": "Measure operational effectiveness",
                        "questions": ["Question 1", "Question 2", "Question 3"]
                    }}
                }}
                
                Only use tables: {allowed_tables}
                Make sure questions are specific and actionable.
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
