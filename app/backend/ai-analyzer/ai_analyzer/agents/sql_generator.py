# backend/ai-analyzer/ai_analyzer/agents/sql_generator.py
from .base import BaseAgent, AgentResponse, DatabaseContext
from typing import Dict
import logging


logger = logging.getLogger(__name__)


class SQLGeneratorAgent(BaseAgent):
    async def process(self,
                      question: str,
                      db_context: DatabaseContext) -> AgentResponse:
        try:
            prompt = self._construct_sql_prompt(question, db_context)
            sql_query = await self.model_provider.generate_response(prompt)

            if not self._validate_query(sql_query, db_context):
                return AgentResponse(
                    success=False,
                    content=None,
                    error_message="Invalid query structure",
                    reformulated_question=f"Could you rephrase your question to focus on {
                        ', '.join(db_context.allowed_tables)} tables?"
                )

            return AgentResponse(
                success=True,
                content=sql_query
            )
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return AgentResponse(
                success=False,
                content=None,
                error_message=str(e)
            )

    def _construct_sql_prompt(self, question: str, db_context: DatabaseContext) -> str:
        # Include your existing SQL generation prompt logic here
        pass

    def _validate_query(self, query: str, db_context: DatabaseContext) -> bool:
        # Include your existing query validation logic here
        pass
