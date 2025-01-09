# backend/ai-analyzer/ai_analyzer/agents/question_generator.py

from .base import BaseAgent, AgentResponse, DatabaseContext
import logging


logger = logging.getLogger(__name__)


class QuestionGeneratorAgent(BaseAgent):
    async def process(self, db_context: DatabaseContext) -> AgentResponse:
        try:
            prompt = self._construct_question_prompt(db_context)
            questions = await self.model_provider.generate_response(prompt)

            # Parse the response into a list of questions
            question_list = [q.strip()
                             for q in questions.split('\n') if q.strip()]

            return AgentResponse(
                success=True,
                content=question_list
            )
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return AgentResponse(
                success=False,
                content=[],
                error_message=str(e)
            )

    def _construct_question_prompt(self, db_context: DatabaseContext) -> str:
        tables_info = "\n".join([
            f"Table: {table}\nColumns: {', '.join(schema.keys())}"
            for table, schema in db_context.table_schemas.items()
        ])

        return f"""
        As a call center data analyst, generate relevant questions based on this database:
        {tables_info}

        Focus on:
        1. Call sentiment trends
        2. Common topics/issues
        3. Customer satisfaction patterns
        4. Agent performance metrics
        5. Time-based analysis

        Format: Return each question on a new line.
        """
