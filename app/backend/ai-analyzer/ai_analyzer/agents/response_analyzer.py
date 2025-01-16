# File: backend/ai-analyzer/ai_analyzer/agents/response_analyzer.py

from .base import BaseAgent, AgentResponse
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisOutput(BaseModel):
    summary: str = Field(description="Main analysis or error explanation")
    followup_questions: List[str] = Field(
        description="Suggested follow-up questions")
    reformulation: Optional[str] = Field(
        description="Reformulated question if there was an error")


class ResponseAnalyzerAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_parser = PydanticOutputParser(
            pydantic_object=AnalysisOutput)

    async def process(self, question: str, response: Any,
                      conversation_context: Optional[str] = None) -> AgentResponse:
        try:
            context_summary = ""
            if conversation_context:
                latest_interaction = conversation_context.split('\n\n')[-2:]
                context_summary = '\n'.join(latest_interaction)

            # Enhanced prompt to generate better follow-up questions
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a call center data analyst. Analyze the query results and return the response in exactly this JSON format:
                {{
                    "summary": "A clear analysis of the results, including key findings and trends",
                    "followup_questions": [
                        "A question about deeper analysis of the current results",
                        "A question about related trends or patterns",
                        "A question that explores a different aspect of the data"
                    ],
                    "reformulation": null
                }}
                
                Guidelines for follow-up questions:
                - Make them specific and related to the current results
                - Include questions about trends over time
                - Ask about correlations with other metrics
                - Explore different aspects of the same topic
                - Make them actionable for business insights
                """),
                ("human",
                 """Previous context: {context}
                 Original question: {question}
                 Results to analyze: {response}
                 
                 Generate a complete analysis with 3-4 relevant follow-up questions.""")
            ])

            # Create the chain and invoke
            chain = prompt | self.llm
            result = await chain.ainvoke({
                "context": context_summary,
                "question": question,
                "response": response
            })

            # Parse and validate output
            content = result.content if hasattr(
                result, 'content') else str(result)
            parsed_output = self.output_parser.parse(content)

            # Ensure we have follow-up questions
            if not parsed_output.followup_questions or len(parsed_output.followup_questions) == 0:
                parsed_output.followup_questions = self._generate_default_followups(
                    question, response)

            return AgentResponse(
                success=True,
                content=parsed_output.summary,
                suggested_followup=parsed_output.followup_questions,
                reformulated_question=parsed_output.reformulation
            )

        except Exception as e:
            logger.error(f"Error analyzing response: {e}")
            return AgentResponse(
                success=False,
                content="Unable to analyze response",
                error_message=str(e),
                suggested_followup=self._generate_default_followups(
                    question, response)
            )

    def _generate_default_followups(self, question: str, response: str) -> List[str]:
        """Generate context-aware default follow-up questions"""
        if "sentiment" in question.lower():
            return [
                "How has this sentiment distribution changed over time?",
                "Which topics are most associated with positive sentiment?",
                "What are the common characteristics of negative feedback?"
            ]
        elif "topic" in question.lower():
            return [
                "How do these topics compare to last month's data?",
                "What is the sentiment distribution for these top topics?",
                "Are there any emerging trends in these topics?"
            ]
        else:
            return [
                "Would you like to see a trend analysis of these results?",
                "Should we analyze any specific aspect in more detail?",
                "Would you like to compare these results with another time period?"
            ]
