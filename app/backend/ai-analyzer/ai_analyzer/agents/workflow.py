# backend/ai-analyzer/ai_analyzer/agents/workflow.py

from typing import List, Dict, Optional, Any
from database_inspector import DatabaseInspectorAgent
from sql_generator import SQLGeneratorAgent
from question_generator import QuestionGeneratorAgent
from response_analyzer import ResponseAnalyzerAgent
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class CallAnalysisWorkflow:
    def __init__(self,
                 db_url: str,
                 model_provider: str,
                 model_name: str,
                 api_key: Optional[str] = None,
                 restricted_tables: Optional[List[str]] = None,
                 base_context: Optional[str] = None,
                 cache_manager=None):

        self.db_inspector = DatabaseInspectorAgent(db_url)
        self.cache_manager = cache_manager

        # Initialize agents with the same model provider
        self.question_generator = QuestionGeneratorAgent(
            model_provider, model_name, api_key)
        self.sql_generator = SQLGeneratorAgent(
            model_provider, model_name, api_key)
        self.response_analyzer = ResponseAnalyzerAgent(
            model_provider, model_name, api_key)

        # Get database context
        self.db_context = self.db_inspector.inspect_database(restricted_tables)
        if base_context:
            self.db_context.base_context = base_context

    async def process_user_question(self, question: str) -> Dict[str, Any]:
        """Process a user's question with caching and performance tracking"""

        start_time = datetime.now()

        # Check cache first
        if self.cache_manager:
            is_cached, cached_response, has_new_records = (
                self.cache_manager.check_cache(question)
            )

            if is_cached and not has_new_records:
                # Get follow-up questions for cached response
                analysis = await self.response_analyzer.process(
                    question, cached_response, None
                )

                self.cache_manager.track_query_performance(
                    query=question,
                    was_answered=True,
                    response_time=(datetime.now() -
                                   start_time).total_seconds(),
                    topic_category="cached_response"
                )

                return {
                    'success': True,
                    'response': cached_response,
                    'followup_questions': analysis.suggested_followup
                }

        # Process new question
        result = await self._process_question(question)

        # Cache successful responses
        if result['success'] and self.cache_manager:
            self.cache_manager.cache_response(
                question, result['response']
            )

        # Track performance
        if self.cache_manager:
            self.cache_manager.track_query_performance(
                query=question,
                was_answered=result['success'],
                response_time=(datetime.now() - start_time).total_seconds(),
                error_message=result.get('error'),
                topic_category=result.get('topic_category', 'unknown')
            )

        return result

    async def _process_question(self, question: str) -> Dict[str, Any]:
        """Internal method to process a question through the workflow"""
        # Generate SQL
        sql_response = await self.sql_generator.process(
            question,
            self.db_context
        )

        if not sql_response.success:
            # Analyze error and suggest alternatives
            analysis = await self.response_analyzer.process(
                question,
                None,
                sql_response.error_message
            )

            return {
                'success': False,
                'error': sql_response.error_message,
                'reformulated_question': sql_response.reformulated_question,
                'followup_questions': analysis.suggested_followup
            }

        # Execute SQL and get results
        try:
            with self.db_inspector.engine.connect() as conn:
                result = conn.execute(sql_response.content)
                data = result.fetchall()
        except Exception as e:
            analysis = await self.response_analyzer.process(
                question,
                None,
                str(e)
            )

            return {
                'success': False,
                'error': str(e),
                'reformulated_question': analysis.reformulated_question,
                'followup_questions': analysis.suggested_followup
            }

        # Analyze successful response
        analysis = await self.response_analyzer.process(
            question,
            data,
            None
        )

        return {
            'success': True,
            'response': analysis.content,
            'data': data,
            'sql': sql_response.content,
            'followup_questions': analysis.suggested_followup
        }

    async def get_suggested_questions(self) -> List[str]:
        """Get suggested questions for analysis"""
        response = await self.question_generator.process(self.db_context)
        return response.content if response.success else []
