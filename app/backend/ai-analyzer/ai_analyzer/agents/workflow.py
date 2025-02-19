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
from sqlalchemy.orm import Session
import re


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# In workflow.py

# In workflow.py

class CallAnalysisWorkflow:
    def __init__(self, db_url: str, model_provider: str, model_name: str, api_key: Optional[str] = None,
                 restricted_tables: Optional[List[str]] = None, base_context: Optional[str] = None,
                 cache_manager=None):
        self.db_url = db_url
        self.db_inspector = DatabaseInspectorAgent(db_url)
        self.cache_manager = cache_manager
        # Initialize conversation history with tenant isolation
        # {tenant_code: {conversation_id: [{"question": str, "sql": str, "result": str}]}}
        self.conversation_history: Dict[str,
                                        Dict[str, List[Dict[str, str]]]] = {}
        self.base_context = base_context

        # Create enhanced context
        enhanced_context = self._create_enhanced_context(base_context)

        # Initialize SQL generator with all required parameters
        self.sql_generator = SQLGeneratorAgent(
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key,
            base_context=enhanced_context
        )

        # Add initial questions generator
        self.initial_questions_generator = InitialQuestionGenerator(
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key
        )

        # Add example queries to conversation history template
        self.example_queries = {
            "question": "Initial example queries",
            "sql": self._get_example_queries(),
            "result": self._get_example_results()
        }

    def _create_enhanced_context(self, base_context: Optional[str]) -> str:
        """Create enhanced context with mandatory tenant filtering"""
        context = base_context or ""
        return f"""
        {context}
        
        CRITICAL TENANT ISOLATION RULES:
        1. EVERY query MUST use this exact base CTE structure:
        WITH base_data AS (
            SELECT 
                t.id,
                t.transcription_id,
                t.transcription,
                t.topic,
                LOWER(TRIM(t.topic)) as clean_topic,
                t.summary,
                t.processing_date,
                t.sentiment,
                CASE
                    WHEN LOWER(TRIM(t.sentiment)) IN ('neutral', 'neutraal') THEN 'neutral'
                    ELSE LOWER(TRIM(t.sentiment))
                END AS clean_sentiment,
                t.call_duration_secs,
                t.telephone_number,
                t.call_direction
            FROM transcription t
            WHERE t.processing_date >= CURRENT_DATE - INTERVAL '300 days'
            AND t.tenant_code = :tenant_code
        ),
        topic_trends AS (
            SELECT 
                clean_topic as topic,
                COUNT(*) as mention_count,
                COUNT(*) FILTER (WHERE processing_date >= CURRENT_DATE - INTERVAL '7 days') as recent_mentions,
                COUNT(*) FILTER (WHERE clean_sentiment = 'positief') as positive_mentions,
                COUNT(*) FILTER (WHERE clean_sentiment = 'negatief') as negative_mentions,
                ROUND(AVG(CASE 
                    WHEN clean_sentiment = 'positief' THEN 1
                    WHEN clean_sentiment = 'negatief' THEN -1
                    ELSE 0 
                END)::numeric, 2) as sentiment_score
            FROM base_data
            WHERE clean_topic IS NOT NULL
            GROUP BY clean_topic
            HAVING COUNT(*) > 5
            ORDER BY recent_mentions DESC, mention_count DESC
        )
        """

    def _get_example_queries(self) -> str:
        """Get example queries for initial context"""
        return """
        Example 1 - Topics in positive calls:
        WITH base_data AS (...)
        SELECT clean_topic, COUNT(*) as mentions, 
        COUNT(*) FILTER (WHERE clean_sentiment = 'positief') as positive_count
        FROM base_data GROUP BY clean_topic;

        Example 2 - Sentiment trends:
        WITH base_data AS (...)
        SELECT clean_topic, clean_sentiment, COUNT(*) 
        FROM base_data GROUP BY clean_topic, clean_sentiment;
        """

    def _get_example_results(self) -> str:
        """Get example results for initial context"""
        return """
        Common patterns in our analysis:
        - Topics are grouped by frequency and sentiment
        - Sentiment is analyzed across time periods
        - Call durations are tracked per topic
        - Customer satisfaction is measured by positive sentiment rate
        """

    def _get_conversation_context(self, conversation_id: str, tenant_code: str) -> str:
        """Create context string from conversation history with tenant isolation"""
        if tenant_code not in self.conversation_history or conversation_id not in self.conversation_history[tenant_code]:
            # For new conversations, start with example context
            return f"""
            Initial Analysis Context:
            {self.example_queries['result']}
            
            Example Queries:
            {self.example_queries['sql']}
            """

        context = "Previous questions and findings:\n"
        # Include example context for reference
        context += f"\nInitial Context:\n{self.example_queries['result']}\n"

        # Last 3 interactions
        for interaction in self.conversation_history[tenant_code][conversation_id][-3:]:
            context += f"\nQuestion: {interaction['question']}\n"
            context += f"Findings: {interaction['result']}\n"
        return context

    async def process_user_question(self, question: str, conversation_id: str, db_session: Session, tenant_code: str):
        """Process user question with conversation history"""
        try:
            if not tenant_code:
                raise ValueError("Tenant code is required")

            # Initialize tenant conversation history if needed
            if tenant_code not in self.conversation_history:
                self.conversation_history[tenant_code] = {}

            # Initialize conversation history if needed
            if conversation_id not in self.conversation_history[tenant_code]:
                self.conversation_history[tenant_code][conversation_id] = []
                # Add example queries as first entry for new conversations
                self.conversation_history[tenant_code][conversation_id].append(
                    self.example_queries)

            # Get previous context for this conversation
            previous_context = self._get_conversation_context(
                conversation_id, tenant_code)

            try:
                # Generate SQL with conversation context
                sql = self.sql_generator.generate_sql_query(
                    question=question,
                    tenant_code=tenant_code,
                    conversation_context=previous_context
                )
            except Exception as e:
                error_str = str(e).lower()
                if 'context length' in error_str and 'exceed' in error_str:
                    return {
                        'success': False,
                        'error': 'Conversation context limit reached. Please start a new conversation to continue analysis.',
                        'context_exceeded': True,  # Flag to indicate context length issue
                        'followup_questions': [
                            "Start a new conversation to analyze recent trends",
                            "Begin fresh analysis of key metrics",
                            "Initiate new topic analysis"
                        ]
                    }
                raise  # Re-raise other exceptions

            # Execute query and get results
            try:
                result = await self.execute_query(sql, {"tenant_code": tenant_code}, db_session)
            except Exception as e:
                logger.error(f"Query execution error: {str(e)}")
                db_session.rollback()  # Rollback failed transaction
                raise

            # Store this interaction in conversation history
            try:
                self.conversation_history[tenant_code][conversation_id].append({
                    "question": question,
                    "sql": sql,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error storing conversation history: {str(e)}")
                # Continue even if history storage fails

            # Generate followup questions with context
            followup_questions = self._generate_followup_questions(
                question,
                result,
                conversation_id,
                tenant_code  # Add tenant_code parameter
            )

            return {
                'success': True,
                'response': result,
                'followup_questions': followup_questions
            }

        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}")
            error_str = str(e).lower()
            if 'context length' in error_str and 'exceed' in error_str:
                return {
                    'success': False,
                    'error': 'Conversation context limit reached. Please start a new conversation to continue analysis.',
                    'context_exceeded': True,
                    'followup_questions': [
                        "Start a new conversation to analyze recent trends",
                        "Begin fresh analysis of key metrics",
                        "Initiate new topic analysis"
                    ]
                }
            return {
                'success': False,
                'error': str(e),
                'followup_questions': self._get_default_followup_questions()
            }

    def _get_default_followup_questions(self) -> List[str]:
        return [
            "What are the most common topics in our calls from the last month?",
            "How has customer sentiment changed over time?",
            "Can you show me the breakdown of call topics by sentiment?"
        ]

    async def get_initial_questions(self) -> Dict[str, Any]:
        """Get categorized initial questions"""
        try:
            return {
                "Trending Topics": {
                    "description": "Analyze popular discussion topics",
                    "questions": [
                        "What are the most discussed topics this month?",
                        "Which topics show increasing trends?",
                        "What topics are commonly mentioned in positive calls?",
                        "How have topic patterns changed over time?",
                        "What are the emerging topics from recent calls?"
                    ]
                },
                "Customer Sentiment": {
                    "description": "Understand customer satisfaction trends",
                    "questions": [
                        "How has overall sentiment changed over time?",
                        "What topics generate the most positive feedback?",
                        "Which issues need immediate attention based on sentiment?",
                        "Show me the distribution of sentiments across topics",
                        "What topics have improving sentiment trends?"
                    ]
                },
                "Call Analysis": {
                    "description": "Analyze call patterns and duration",
                    "questions": [
                        "What is the average call duration by topic?",
                        "Which topics tend to have longer calls?",
                        "Show me the call volume trends by time of day",
                        "What's the distribution of call directions by topic?",
                        "Which days have the highest call volumes?"
                    ]
                },
                "Topic Correlations": {
                    "description": "Discover relationships between topics",
                    "questions": [
                        "Which topics often appear together?",
                        "What topics are related to technical issues?",
                        "Show me topics that commonly lead to follow-up calls",
                        "What topics frequently occur with complaints?",
                        "Which topics have similar sentiment patterns?"
                    ]
                },
                "Performance Metrics": {
                    "description": "Analyze key performance indicators",
                    "questions": [
                        "What's our overall customer satisfaction rate?",
                        "Show me topics with the highest resolution rates",
                        "Which topics need more attention based on metrics?",
                        "What are our best performing areas?",
                        "Show me trends in call handling efficiency"
                    ]
                },
                "Time-based Analysis": {
                    "description": "Understand temporal patterns",
                    "questions": [
                        "What are the busiest times for calls?",
                        "How do topics vary by time of day?",
                        "Show me weekly trends in call volumes",
                        "What patterns emerge during peak hours?",
                        "Which days show the best sentiment scores?"
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error getting initial questions: {e}")
            return {}

    def _generate_followup_questions(self, question: str, response: str, conversation_id: str, tenant_code: str) -> List[str]:
        """Generate dynamic followup questions based on conversation context and call analysis patterns"""
        try:
            # Get last 3 interactions for context
            recent_interactions = self.conversation_history.get(
                tenant_code, {}).get(conversation_id, [])[-3:]

            # Analyze current response content
            response_lower = response.lower()
            current_q = question.lower()

            # Extract current topics from response
            topics_mentioned = []
            for line in response.split('\n'):
                if 'topic:' in line.lower():
                    topic = line.split('topic:')[1].split(',')[0].strip()
                    topics_mentioned.append(topic)

            # Track key metrics and topics from response
            metrics = {
                'sentiment_mentioned': any(x in response_lower for x in ['sentiment', 'positief', 'negatief', 'satisfaction']),
                'topics_mentioned': any(x in response_lower for x in ['topic', 'onderwerp']),
                'time_mentioned': any(x in response_lower for x in ['time', 'period', 'week', 'month', 'recent']),
                'volume_mentioned': any(x in response_lower for x in ['count', 'mentions', 'calls']),
                'trends_mentioned': 'trend' in response_lower,
                'technical_support': any(x in response_lower for x in ['technische ondersteuning', 'technical support']),
                'customer_service': any(x in response_lower for x in ['klantenservice', 'customer service']),
                'positive_rate': any(x in response_lower for x in ['positive_rate', 'satisfaction_rate'])
            }

            # If we have specific topics in the response, prioritize topic-based questions
            if topics_mentioned:
                main_topic = topics_mentioned[0]
                if not any("sentiment" in interaction.get('question', '').lower() for interaction in recent_interactions):
                    return [
                        f"How has the sentiment changed for '{main_topic}'?",
                        f"What are the common patterns in calls about '{main_topic}'?",
                        "Which other topics are frequently mentioned together with these?"
                    ]
                return [
                    f"What time patterns do we see for '{main_topic}'?",
                    f"How does '{main_topic}' relate to other topics?",
                    "What are the emerging subtopics in these conversations?"
                ]

            # If response shows concerning metrics
            if 'negative' in response_lower or 'negatief' in response_lower:
                return [
                    "What specific issues are causing negative sentiment?",
                    "When do these negative interactions typically occur?",
                    "Which agents handle these challenging calls?"
                ]

            # If response shows positive trends
            if 'positive_rate: 100.00' in response_lower or 'positief' in response_lower:
                return [
                    "What best practices can we identify from these successful interactions?",
                    "Which team members are handling these positive calls?",
                    "How can we apply these successful approaches to other areas?"
                ]

            # If looking at technical support
            if metrics['technical_support']:
                return [
                    "What are the most common technical issues being resolved?",
                    "How do resolution times vary by issue type?",
                    "Which technical issues lead to follow-up calls?"
                ]

            # If analyzing customer service
            if metrics['customer_service']:
                return [
                    "What patterns emerge in customer escalations?",
                    "How do different handling approaches affect outcomes?",
                    "Which customer service scenarios need additional training?"
                ]

            # If analyzing trends
            if metrics['trends_mentioned']:
                return [
                    "What factors are driving these trend changes?",
                    "How do these trends compare to our targets?",
                    "Which areas show the most significant shifts?"
                ]

            # If looking at specific topics
            if metrics['topics_mentioned']:
                return [
                    "How do handling times vary across these topics?",
                    "Which topics most often lead to escalations?",
                    "What training needs do these topics suggest?"
                ]

            # Default questions based on call analysis patterns
            return [
                "How do these patterns affect our service quality?",
                "What operational changes could improve these metrics?",
                "Which areas need additional agent training?"
            ]

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return [
                "What other call patterns should we investigate?",
                "How can we improve our handling of these calls?",
                "What additional training might be helpful?"
            ]

    async def execute_query(self, sql: str, params: dict, db_session: Session) -> str:
        """Execute SQL query and format results"""
        try:
            # Log query execution
            logger.info(f"Executing SQL with params {params}:\n{sql}")

            # Execute query
            result = db_session.execute(text(sql), params)
            rows = result.fetchall()
            column_names = result.keys()

            # Format results into readable text
            if not rows:
                logger.info("Query returned no results")
                return "No results found for your query."

            # Convert rows to list of dicts for easier handling
            results = [dict(zip(column_names, row)) for row in rows]

            # Log result count
            logger.info(f"Query returned {len(results)} rows")

            # Format output based on number of rows
            if len(results) == 1:
                # Single row result
                return "\n".join(f"{k}: {v}" for k, v in results[0].items())
            else:
                # Multiple row result
                output = []
                for row in results:
                    row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
                    output.append(f"- {row_str}")
                return "\n".join(output)

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise ValueError(f"Error executing query: {str(e)}")
