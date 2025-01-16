# backend/ai-analyzer/ai_analyzer/agents/workflow.py
import logging
import os
from ai_analyzer.agents.initial_questions import InitialQuestionGenerator
from ai_analyzer.agents.response_analyzer import ResponseAnalyzerAgent
from ai_analyzer.agents.question_generator import QuestionGeneratorAgent
from ai_analyzer.agents.sql_generator import SQLGeneratorAgent
from ai_analyzer.agents.database_inspector import DatabaseInspectorAgent
from typing import List, Dict, Optional, Any
from sqlalchemy import create_engine, text
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Union


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CallAnalysisWorkflow:
    """
    Orchestrates the workflow for analyzing call center data through various specialized agents.
    Handles question processing, SQL generation, response analysis, and caching.
    """

    def __init__(self,
                 db_url: str,
                 model_provider: str,
                 model_name: str,
                 api_key: Optional[str] = None,
                 restricted_tables: Optional[List[str]] = None,
                 base_context: Optional[str] = None,
                 cache_manager=None):
        """
        Initialize the workflow with necessary components and configurations.

        Args:
            db_url: Database connection URL
            model_provider: AI model provider (e.g., "openai")
            model_name: Name of the model to use
            api_key: API key for the model provider
            restricted_tables: List of tables to exclude from queries
            base_context: Additional context for query processing
            cache_manager: Optional cache manager instance
        """
        # Initialize database connection
        self.engine = create_engine(db_url)
        self.db_inspector = DatabaseInspectorAgent(db_url)
        self.cache_manager = cache_manager

        # Initialize specialized agents
        self.initial_question_generator = InitialQuestionGenerator(
            model_provider, model_name, api_key)
        self.sql_generator = SQLGeneratorAgent(
            model_provider, model_name, api_key, base_context)
        self.question_generator = QuestionGeneratorAgent(
            model_provider, model_name, api_key)
        self.response_analyzer = ResponseAnalyzerAgent(
            model_provider, model_name, api_key)

        # Get database context and store configuration
        self.db_context = self.db_inspector.inspect_database(restricted_tables)
        self.base_context = base_context

    async def get_initial_questions(self) -> Dict[str, Dict[str, Any]]:
        """Get categorized initial questions for users to start with."""
        try:
            response = await self.initial_question_generator.process(self.db_context)
            if not response.success:
                logger.warning(
                    "Failed to generate initial questions, using defaults")
                return self._get_default_questions()
            return response.content if response.success else {}
        except Exception as e:
            logger.error(f"Error getting initial questions: {e}")
            return self._get_default_questions()

    async def process_user_question(self, question: str, conversation_id: str = None, db_session=None) -> Dict[str, Any]:
        try:
            # Get conversation context
            text_context, context_obj = self.db_inspector.get_conversation_context(
                db_session, conversation_id)

            # Format the context into a structured summary
            context_summary = None
            if text_context:
                context_summary = {
                    'previous_question': text_context.split('Previous Question: ')[-1].split('\n')[0] if 'Previous Question: ' in text_context else None,
                    'previous_results': text_context.split('Previous Answer: ')[-1].split('\n')[0] if 'Previous Answer: ' in text_context else None,
                }
                logger.info(f"Previous context summary: {context_summary}")

            # Enhance question with context if available
            if context_obj:
                enhanced_question = self.db_inspector.enhance_question_with_context(
                    question, context_obj)
                logger.info(f"Enhanced question: {enhanced_question}")
            else:
                enhanced_question = question

            # Generate SQL from the question with structured context
            logger.info("Generating SQL query...")
            sql_response = await self.sql_generator.process(
                question=enhanced_question,
                db_context=self.db_context,
                conversation_context=context_summary,
                context_obj=context_obj
            )

            logger.info(f"Generated SQL: {
                        sql_response.content if sql_response.success else 'Failed'}")

            if not sql_response.success:
                logger.error(f"SQL generation failed: {
                             sql_response.error_message}")
                return {
                    'success': False,
                    'error': sql_response.error_message,
                    'reformulated_question': sql_response.reformulated_question,
                    'followup_questions': sql_response.suggested_followup
                }

            # Execute the SQL and get the result
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(sql_response.content))
                    rows = result.fetchall()
                    column_names = result.keys()

                    # Format the results into a readable response
                    formatted_response = self._format_results(
                        rows, column_names)
            except Exception as e:
                logger.error(f"Error executing SQL: {e}")
                return {
                    'success': False,
                    'error': f"Error executing query: {str(e)}",
                    'followup_questions': self._get_default_followup_questions()
                }

            # Analyze the response with context
            analysis_response = await self.response_analyzer.process(
                question=question,
                response=formatted_response,
                conversation_context=text_context
            )

            # Cache successful response if cache manager is available
            if self.cache_manager and formatted_response:
                self.cache_manager.cache_response(question, formatted_response)

            return {
                'success': True,
                'response': formatted_response,
                'followup_questions': analysis_response.suggested_followup if analysis_response.success else []
            }

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'success': False,
                'error': str(e),
                'followup_questions': self._get_default_followup_questions()
            }

    def _format_results(self, rows: List[Any], column_names: List[str]) -> str:
        """Format SQL results into a readable response."""
        if not rows:
            return "No results found for your query."

        # Convert rows to list of dicts for easier handling
        results = [dict(zip(column_names, row)) for row in rows]

        # Different formatting based on number of rows
        if len(results) == 1:
            return self._format_single_result(results[0])
        else:
            return self._format_multiple_results(results)

    def _format_single_result(self, result: Dict[str, Any]) -> str:
        """Format a single result row."""
        formatted_parts = []
        for key, value in result.items():
            if isinstance(value, (int, float)):
                formatted_parts.append(
                    f"{key.replace('_', ' ').title()}: {value:,}")
            elif isinstance(value, datetime):
                formatted_parts.append(f"{key.replace('_', ' ').title()}: {
                                       value.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                formatted_parts.append(
                    f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(formatted_parts)

    def _format_multiple_results(self, results: List[Dict[str, Any]]) -> str:
        """Format multiple result rows."""
        # Get total number of results
        total = len(results)

        # Format each row
        formatted_rows = []
        for i, row in enumerate(results, 1):
            row_parts = []
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    row_parts.append(
                        f"{key.replace('_', ' ').title()}: {value:,}")
                elif isinstance(value, datetime):
                    row_parts.append(f"{key.replace('_', ' ').title()}: {
                                     value.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    row_parts.append(
                        f"{key.replace('_', ' ').title()}: {value}")
            formatted_rows.append(f"Result {i}:\n" + "\n".join(row_parts))

        # Combine all parts
        return f"Found {total} results:\n\n" + "\n\n".join(formatted_rows)

    def _get_default_questions(self) -> Dict[str, Dict[str, Any]]:
        """Provide default initial questions if generation fails."""
        return {
            "Trending Topics": {
                "description": "Analyze popular discussion topics",
                "questions": [
                    "What are the most discussed topics this month?",
                    "Which topics show increasing trends?",
                    "What topics are commonly mentioned in positive calls?"
                ]
            },
            "Customer Sentiment": {
                "description": "Understand customer satisfaction trends",
                "questions": [
                    "How has overall sentiment changed over time?",
                    "What topics generate the most positive feedback?",
                    "Which issues need immediate attention based on sentiment?"
                ]
            },
            "Performance Analysis": {
                "description": "Analyze call center performance metrics",
                "questions": [
                    "What is the distribution of sentiments across all calls?",
                    "How do sentiment patterns vary by topic?",
                    "What are the most common topics in negative feedback?"
                ]
            }
        }

    def _get_default_followup_questions(self) -> List[str]:
        """Provide default follow-up questions if generation fails."""
        return [
            "What are the most common topics in our calls?",
            "How has customer sentiment changed over time?",
            "Can you show me the breakdown of call topics by sentiment?"
        ]
