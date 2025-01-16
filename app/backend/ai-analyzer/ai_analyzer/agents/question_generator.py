# File: backend/ai-analyzer/ai_analyzer/agents/question_generator.py

from .base import BaseAgent, AgentResponse, DatabaseContext
from langchain.prompts import PromptTemplate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionGeneratorAgent(BaseAgent):
    async def process(self, db_context: DatabaseContext) -> AgentResponse:
        try:
            prompt = self._construct_question_prompt(db_context)
            chain = self._create_chain(prompt)

            # Use LangChain to generate questions
            result = await chain.arun(input=str(db_context.table_schemas))
            questions = [q.strip() for q in result.split('\n') if q.strip()]

            return AgentResponse(
                success=True,
                content=questions
            )
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return AgentResponse(
                success=False,
                content=[],
                error_message=str(e)
            )

    def _construct_question_prompt(self, db_context: DatabaseContext) -> str:
        return """
        You are a call center data analyst. Based on the following database structure:
        {input}
        
        Generate relevant questions that would help managers analyze call transcriptions.
        Focus on:
        1. Call sentiment trends
        2. Common topics/issues
        3. Customer satisfaction patterns
        4. Agent performance metrics
        5. Time-based analysis
        
        Format: Return each question on a new line.
        """
