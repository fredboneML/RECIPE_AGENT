# File: backend/ai-analyzer/ai_analyzer/agents/sql_generator.py

from typing import Dict, Any, Optional, Tuple, List
import logging
from langchain.prompts import ChatPromptTemplate
import json
from .base import BaseAgent, AgentResponse, DatabaseContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLGeneratorAgent(BaseAgent):
    """Enhanced SQL generator for database queries"""

    def __init__(self,
                 model_provider: str,
                 model_name: str,
                 api_key: Optional[str] = None,
                 base_context: Optional[str] = None,
                 **kwargs):
        # First call parent's __init__ with only the required arguments
        super().__init__(model_provider, model_name, api_key)
        # Then store base_context as instance variable
        self.base_context = base_context or ""

    async def process(self, question: str, db_context: DatabaseContext,
                      conversation_context: Optional[Dict] = None,
                      context_obj: Optional[Any] = None) -> AgentResponse:
        try:
            # Build context description
            context_description = ""
            if conversation_context:
                context_description = f"""
                Previous question: {conversation_context.get('previous_question')}
                Previous results: {conversation_context.get('previous_results')}
                """
                logger.info(f"Using conversation context: {
                            context_description}")

            # Create prompt with better context handling
            prompt = ChatPromptTemplate.from_template("""
                {context_info}

                Rules:
                {base_rules}
                
                Available tables: {table_names}
                Schema information: {schema_info}
                
                Current question: {question}
                
                Generate an SQL query that considers the previous results and answers the current question.
                The query should build upon the previous results if they are relevant.
                Return SQL query only.
            """)

            # Create the chain with structured context
            chain = prompt | self.llm

            # Access the dataclass attributes properly
            result = await chain.ainvoke({
                "context_info": context_description or "No previous context",
                "base_rules": self.base_context,
                # Access as attribute
                "table_names": ", ".join(db_context.allowed_tables),
                # Access as attribute
                "schema_info": json.dumps(db_context.table_schemas, indent=2),
                "question": question
            })

            logger.info(f"LLM Response: {result}")

            # Extract content from the response
            content = result.content if hasattr(
                result, 'content') else str(result)

            # Validate the generated SQL
            is_valid, error_message = self._validate_sql(content)
            if not is_valid:
                return AgentResponse(
                    success=False,
                    content=None,
                    error_message=error_message,
                    reformulated_question=self._suggest_reformulation(
                        question),
                    suggested_followup=self._generate_followup_suggestions(
                        question)
                )

            return AgentResponse(
                success=True,
                content=content
            )

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return AgentResponse(
                success=False,
                content=None,
                error_message=str(e),
                reformulated_question=self._suggest_reformulation(question),
                suggested_followup=self._generate_followup_suggestions(
                    question)
            )

    def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate the generated SQL query"""
        sql = sql.strip().upper()

        # Basic validation checks
        if not sql.startswith('WITH BASE_DATA AS'):
            return False, "Query must start with base_data CTE"

        if 'DELETE' in sql or 'UPDATE' in sql or 'INSERT' in sql:
            return False, "Only SELECT statements are allowed"

        # Check for column name references after aliasing
        sql_parts = sql.split('FROM')
        # Skip the first part (base CTE)
        for i, part in enumerate(sql_parts[1:], 1):
            # Check if we're using the correct column names after aliasing
            if 'WHERE CLEAN_TOPIC' in part and 'AS TOPIC' in sql_parts[i-1]:
                return False, "Use 'topic' instead of 'clean_topic' after aliasing"

        return True, None

    def _suggest_reformulation(self, question: str) -> str:
        """Suggest a reformulation of the question"""
        # Add specific terms to help guide the query generation
        if 'trend' in question.lower():
            return f"How has {question} changed over the last 30 days?"
        elif 'compare' in question.lower():
            return f"What is the difference in {question} between this week and last week?"
        else:
            return f"Can you show me the statistics for {question} from the last 30 days?"

    def _generate_followup_suggestions(self, question: str) -> List[str]:
        """Generate follow-up question suggestions"""
        return [
            "What are the top topics in the last 30 days?",
            "How has sentiment changed week over week?",
            "Which topics have the highest positive sentiment?"
        ]
