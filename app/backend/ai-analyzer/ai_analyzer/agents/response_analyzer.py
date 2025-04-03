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

    def _detect_language(self, text: str) -> str:
        """Detect if the text is in Dutch or English"""
        # Get total word count and cleaned words
        words = text.lower().split()
        total_words = len(words)
        logger.info(f"Total words in text: {total_words}")

        # Count English words
        english_words = ['what', 'which', 'how', 'where', 'when', 'who', 'why', 'did', 'does', 'has', 'had', 'it', 'there', 'they', 'show',
                         'is', 'are', 'was', 'were', 'the', 'this', 'that', 'these', 'those', 'our', 'your', 'an', 'a', 'top', 'give', 'do']
        nb_en = sum(1 for word in words if word in english_words)
        logger.info(
            f"English words found: {[word for word in words if word in english_words]}")

        # Count Dutch words
        dutch_words = ['wat', 'hoe', 'waarom', 'welke', 'kunnen', 'waar', 'wie', 'wanneer', 'onderwerp',
                       'kun', 'kunt', 'je', 'jij', 'u', 'bent', 'zijn', 'waar', 'wat', 'wie', 'hoe',
                       'waarom', 'wanneer', 'welk', 'welke', 'het', 'de', 'een', 'het', 'deze', 'dit',
                       'die', 'dat', 'mijn', 'uw', 'jullie', 'ons', 'onze', 'geen', 'niet', 'met',
                       'over', 'door', 'om', 'op', 'voor', 'na', 'bij', 'aan', 'in', 'uit', 'te',
                       'bedrijf', 'waarom', 'tevreden', 'graag', 'gaan', 'wordt', 'komen', 'zal']
        nb_dutch = sum(1 for word in words if word in dutch_words)
        logger.info(
            f"Dutch words found: {[word for word in words if word in dutch_words]}")

        logger.info(
            f"number of Dutch words: {nb_dutch}, number of English words: {nb_en}")

        # Set language based on majority
        if nb_dutch > nb_en:
            return "Dutch"
        elif nb_en > nb_dutch:
            return "English"
        else:
            # Default to English if no clear majority
            return "English"

    async def process(self, question: str, response: Any,
                      conversation_context: Optional[str] = None) -> AgentResponse:
        try:
            context_summary = ""
            if conversation_context:
                latest_interaction = conversation_context.split('\n\n')[-2:]
                context_summary = '\n'.join(latest_interaction)

            # Detect language of the question
            detected_language = self._detect_language(question)
            logger.info(f"Detected language: {detected_language}")

            # Enhanced prompt to generate better insights and follow-up questions
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a professional call center data analyst specializing in customer interaction analysis. 
                Your task is to analyze query results, highlight important patterns, and generate insightful follow-up questions.
                
                Return the response in exactly this JSON format:
                {{
                    "summary": "A clear, concise analysis of the results that highlights key metrics, trends, and actionable insights",
                    "followup_questions": [
                        "A specific question about issues within the topics mentioned",
                        "A question that explores metrics and aggregated values",
                        "A trend-related question that provides context from previous periods"
                    ],
                    "reformulation": null
                }}
                
                Guidelines for your analysis:
                - Highlight specific issues within topics mentioned
                - Emphasize numerical metrics and percentages
                - Identify trends, patterns, and anomalies
                - Make clear connections between customer issues and metrics
                - Format numerical values consistently (e.g., "85.2%" not "85.2 percent")
                
                Guidelines for follow-up questions:
                - Focus on specific issues within topics
                - Include questions about aggregate metrics (percentages, counts, averages)
                - Ask about comparisons with previous time periods
                - Make questions highly specific and actionable
                - Phrase questions as a call center analyst would
                - Include references to specific values from the results
                
                IMPORTANT: When analyzing SQL query results, base your analysis ONLY on the SQL results provided, not on any call transcriptions that may be included in the prompt.
                The call transcriptions are meant to help the SQL agent generate the query, not for you to analyze directly.
                Focus your analysis entirely on the structured data from the SQL results, including metrics, counts, percentages, and trends.
                Make your response user-friendly by avoiding SQL terminology - present the data in plain language without mentioning SQL, queries, or database terms.
                
                IMPORTANT: Respond in {detected_language} language to match the user's question language.
                """),
                ("human",
                 """Previous context: {context}
                 Original question: {question}
                 Results to analyze: {response}
                 
                 Generate a comprehensive analysis that highlights key issues and metrics, followed by 3-4 relevant follow-up questions.""")
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
