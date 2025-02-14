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
        self.engine = create_engine(db_url)
        self.db_inspector = DatabaseInspectorAgent(db_url)
        self.cache_manager = cache_manager
        self.conversation_history = {}  # Track questions per conversation

        # Create enhanced context
        enhanced_context = self._create_enhanced_context(base_context)

        # Initialize specialized agents with correct parameters
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
            WHERE t.processing_date >= CURRENT_DATE - INTERVAL '60 days'
            AND t.tenant_code = :tenant_code
        )

        2. For trending topics analysis, use this pattern:
        WITH base_data AS (
            -- Base CTE with tenant filtering as shown above
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

    async def process_user_question(self, question: str, conversation_id: str, db_session: Session, tenant_code: str):
        """Process user question with mandatory tenant isolation"""
        try:
            # Initialize or update conversation history
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []
            self.conversation_history[conversation_id].append(question)

            # Generate SQL with tenant context
            sql = self.sql_generator.generate_sql_query(question, tenant_code)

            # Double-check tenant filtering is present
            if "WITH base_data AS" not in sql or "t.tenant_code = :tenant_code" not in sql:
                return {
                    'success': False,
                    'error': 'Generated SQL missing required tenant filtering',
                    'reformulated_question': f"Can you show me {question.lower().rstrip('?')}? from the last 60 days, ordered by recent activity?",
                    # Use default questions here
                    'followup_questions': self._get_default_followup_questions()
                }

            # Execute query with tenant parameter
            params = {'tenant_code': tenant_code}
            result = await self.execute_query(sql, params, db_session)

            return {
                'success': True,
                'response': result,
                'followup_questions': self._generate_followup_questions(
                    question,
                    result,
                    conversation_id
                )
            }

        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'reformulated_question': f"Can you show me {question.lower().rstrip('?')}? from the last 60 days, ordered by recent activity?",
                # Use default questions here
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

    def _generate_followup_questions(self, question: str, response: str, conversation_id: str) -> List[str]:
        """Generate context-aware followup questions based on query results and conversation history"""
        question_lower = question.lower()
        response_lower = response.lower()

        # Get conversation history
        previous_questions = self.conversation_history.get(conversation_id, [])
        asked_topics = set()
        for prev_q in previous_questions:
            # Extract topics from previous questions
            if "topic" in prev_q.lower():
                topic_match = re.search(r"'([^']*)'", prev_q)
                if topic_match:
                    asked_topics.add(topic_match.group(1))

        # Extract current topics from response
        topics_mentioned = []
        for line in response.split('\n'):
            if 'topic:' in line.lower():
                topic = line.split('topic:')[1].split(',')[0].strip()
                topics_mentioned.append(topic)

        # Filter out previously discussed topics
        new_topics = [t for t in topics_mentioned if t not in asked_topics]

        # Duration related questions
        if "duration" in question_lower or "call_duration" in response_lower:
            if not any("sentiment" in q.lower() for q in previous_questions):
                return [
                    "Is there a correlation between call duration and sentiment?",
                    "What time of day do we see longer calls?",
                    "Which topics consistently have longer durations?"
                ]
            return [
                "How do call durations vary by time of day?",
                "What patterns do we see in call duration trends?",
                "Which topics need attention based on call duration?"
            ]

        # Topic related questions with new topics
        if new_topics:
            main_topic = new_topics[0]
            if not any("sentiment" in q.lower() for q in previous_questions):
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

        # Sentiment questions considering history
        if "sentiment" in question_lower:
            if not any("duration" in q.lower() for q in previous_questions):
                return [
                    "Is there a correlation between sentiment and call duration?",
                    "What times of day show the best sentiment?",
                    "Which topics need attention based on sentiment trends?"
                ]
            return [
                "What patterns emerge in positive interactions?",
                "How can we improve sentiment scores?",
                "Which areas show improving sentiment trends?"
            ]

        # Trend questions with history context
        if any(word in question_lower for word in ["trend", "change", "pattern", "emerging"]):
            if not any("sentiment" in q.lower() for q in previous_questions):
                return [
                    "How does sentiment relate to these trends?",
                    "Which trending topics need attention?",
                    "What new patterns are emerging in customer interactions?"
                ]
            return [
                "What factors drive these trends?",
                "How can we capitalize on positive trends?",
                "Which areas need intervention based on trends?"
            ]

        # Default questions considering history
        asked_categories = set()
        for q in previous_questions:
            if "sentiment" in q.lower():
                asked_categories.add("sentiment")
            if "duration" in q.lower():
                asked_categories.add("duration")
            if "trend" in q.lower():
                asked_categories.add("trend")

        if "sentiment" not in asked_categories:
            return [
                "How has sentiment changed across topics?",
                "Which topics show the best satisfaction rates?",
                "What patterns emerge in customer sentiment?"
            ]
        if "duration" not in asked_categories:
            return [
                "What patterns do we see in call durations?",
                "Which topics have concerning duration trends?",
                "How can we optimize call handling time?"
            ]
        return [
            "What new patterns are emerging?",
            "Which areas need immediate attention?",
            "How can we improve our performance metrics?"
        ]

    async def execute_query(self, sql: str, params: dict, db_session: Session) -> str:
        """Execute SQL query and format results"""
        try:
            # Execute query
            result = db_session.execute(text(sql), params)
            rows = result.fetchall()
            column_names = result.keys()

            # Format results into readable text
            if not rows:
                return "No results found for your query."

            # Convert rows to list of dicts for easier handling
            results = [dict(zip(column_names, row)) for row in rows]

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
