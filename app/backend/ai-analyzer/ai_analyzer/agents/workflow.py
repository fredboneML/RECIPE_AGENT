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
# Import model logging utilities
from ai_analyzer.utils.model_logger import ModelLogger, get_model_config_from_env
import time


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# In workflow.py

# In workflow.py

class CallAnalysisWorkflow:

    def __init__(self, tenant_code: str):
        self.tenant_code = tenant_code

        # Get model configuration from environment
        model_config = get_model_config_from_env()
        self.model_provider = model_config["provider"]

        # Log initial model configuration
        logger.info(
            f"Initializing CallAnalysisWorkflow for tenant {tenant_code}")
        logger.info(f"Model provider: {self.model_provider}")

        # Handle special case for Groq
        if self.model_provider == "groq":
            self.model_name = model_config["groq_model_name"]
            self.use_openai_compatibility = model_config["groq_use_openai_compatibility"]

            # Log Groq-specific configuration
            logger.info(f"Using Groq model: {self.model_name}")
            logger.info(
                f"Groq OpenAI compatibility mode: {self.use_openai_compatibility}")

            # When using OpenAI compatibility mode, set provider to OpenAI with Groq base URL
            if self.use_openai_compatibility:
                self.model_provider = "openai"
                self.base_url = "https://api.groq.com/openai/v1"
                self.api_key = model_config.get("groq_api_key", "")
                logger.info("Configured Groq with OpenAI compatibility mode")
            else:
                self.api_key = model_config.get("groq_api_key", "")
                logger.info("Configured Groq in native mode")
        else:
            self.model_name = model_config.get("model_name", "gpt-3.5-turbo")
            self.api_key = model_config.get("api_key", "")
            logger.info(
                f"Using {self.model_provider} model: {self.model_name}")

        # Log initialization
        ModelLogger.log_model_usage(
            agent_name="CallAnalysisWorkflow",
            model_provider=self.model_provider,
            model_name=self.model_name,
            params={"tenant_code": tenant_code}
        )

        # Initialize workflow components
        self.db_inspector = DatabaseInspectorAgent(db_url)
        self.conversation_history = {}

        # Create enhanced context
        enhanced_context = self._create_enhanced_context(
            self.base_context or create_base_context(tenant_code))

        # Initialize SQL generator with model settings
        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self.sql_generator = SQLGeneratorAgent(
            model_provider=self.model_provider,
            model_name=self.model_name,
            api_key=self.api_key,
            base_context=enhanced_context,
            **kwargs
        )

        # Add placeholder for follow-up questions generator
        self.followup_generator = None

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

        # Last 10 interactions instead of 3
        for interaction in self.conversation_history[tenant_code][conversation_id][-10:]:
            context += f"\nQuestion: {interaction['question']}\n"
            context += f"Findings: {interaction['result']}\n"
        return context

    async def process_user_question(self, question: str, conversation_id: str, db_session: Session, tenant_code: str):
        """Process user question with conversation history"""
        start_time = datetime.utcnow()
        was_answered = False
        error_message = None
        tokens_used = 0
        topic_category = None

        try:
            if not tenant_code:
                raise ValueError("Tenant code is required")

            # Detect language (Dutch vs English) using word counting
            words = question.lower().split()
            total_words = len(words)

            # Count English words
            english_words = ['what', 'which', 'how', 'where', 'when', 'who', 'why', 'did', 'does', 'has', 'had', 'it', 'there', 'they', 'show',
                             'is', 'are', 'was', 'were', 'the', 'this', 'that', 'these', 'those', 'our', 'your', 'an', 'a', 'top', 'give', 'do']
            nb_en = sum(1 for word in words if word in english_words)

            # Count Dutch words
            dutch_words = ['wat', 'hoe', 'waarom', 'welke', 'kunnen', 'waar', 'wie', 'wanneer', 'onderwerp',
                           'kun', 'kunt', 'je', 'jij', 'u', 'bent', 'zijn', 'waar', 'wat', 'wie', 'hoe',
                           'waarom', 'wanneer', 'welk', 'welke', 'het', 'de', 'een', 'het', 'deze', 'dit',
                           'die', 'dat', 'mijn', 'uw', 'jullie', 'ons', 'onze', 'geen', 'niet', 'met',
                           'over', 'door', 'om', 'op', 'voor', 'na', 'bij', 'aan', 'in', 'uit', 'te',
                           'bedrijf', 'waarom', 'tevreden', 'graag', 'gaan', 'wordt', 'komen', 'zal']
            nb_dutch = sum(1 for word in words if word in dutch_words)

            # Set language based on majority
            is_dutch = nb_dutch > nb_en

            # Initialize tenant conversation history if needed
            if tenant_code not in self.conversation_history:
                self.conversation_history[tenant_code] = {}

            # Initialize conversation history if needed
            if conversation_id not in self.conversation_history[tenant_code]:
                self.conversation_history[tenant_code][conversation_id] = []
                # Add example queries as first entry for new conversations
                self.conversation_history[tenant_code][conversation_id].append(
                    self.example_queries)

            # Check cache for this exact question in this tenant
            if self.cache_manager:
                try:
                    # Pass conversation_id to get_query_result
                    cached_result = self.cache_manager.get_query_result(
                        question, tenant_code, conversation_id)
                    if cached_result:
                        logger.info(f"Cache hit for question: {question}")

                        # Store this interaction in conversation history
                        try:
                            self.conversation_history[tenant_code][conversation_id].append({
                                "question": question,
                                "sql": cached_result.get("sql", ""),
                                "result": cached_result.get("result", "")
                            })
                        except Exception as e:
                            logger.error(
                                f"Error storing cached result in conversation history: {str(e)}")

                        # Generate followup questions with context
                        followup_questions = self._generate_followup_questions(
                            question,
                            cached_result.get("result", ""),
                            conversation_id,
                            tenant_code
                        )

                        # Record performance data for cached query
                        was_answered = True
                        end_time = datetime.utcnow()
                        response_time = int(
                            (end_time - start_time).total_seconds() * 1000)

                        # Try to extract topic category from the question
                        topic_category = self._extract_topic_category(question)

                        self._record_query_performance(
                            db_session,
                            question,
                            was_answered,
                            None,
                            response_time,
                            topic_category,
                            tokens_used,
                            True  # is_cached
                        )

                        return {
                            'success': True,
                            'response': cached_result.get("result", ""),
                            'followup_questions': followup_questions,
                            'cached': True
                        }
                except Exception as e:
                    logger.warning(f"Cache lookup failed: {str(e)}")
                    # Continue with normal processing if cache lookup fails

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
                # Estimate tokens used for the SQL generation
                tokens_used += len(question) // 4 + len(previous_context) // 4
            except Exception as e:
                error_str = str(e).lower()
                error_message = error_str

                end_time = datetime.utcnow()
                response_time = int(
                    (end_time - start_time).total_seconds() * 1000)
                topic_category = self._extract_topic_category(question)

                self._record_query_performance(
                    db_session,
                    question,
                    False,
                    error_message,
                    response_time,
                    topic_category,
                    tokens_used
                )

                if 'context length' in error_str and 'exceed' in error_str:
                    error_msg = "Conversation context limit reached. Please start a new conversation to continue analysis." if not is_dutch else "Limiet van gesprekscontext bereikt. Start een nieuw gesprek om de analyse voort te zetten."
                    return {
                        'success': False,
                        'error': error_msg,
                        'context_exceeded': True,  # Flag to indicate context length issue
                        'followup_questions': [
                            "Start een nieuw gesprek om recente trends te analyseren" if is_dutch else "Start a new conversation to analyze recent trends",
                            "Begin een nieuwe analyse van belangrijke metrieken" if is_dutch else "Begin fresh analysis of key metrics",
                            "Start een nieuwe topic-analyse" if is_dutch else "Initiate new topic analysis"
                        ]
                    }
                raise  # Re-raise other exceptions

            # Execute query and get results
            try:
                result = await self.execute_query(sql, {"tenant_code": tenant_code}, db_session)
                was_answered = True

                # Try to extract topic category from the result
                if not topic_category:
                    topic_category = self._extract_topic_category(result)

                # Cache the successful result
                if self.cache_manager:
                    try:
                        # Use the existing store_query_result method instead of cache_query
                        self.cache_manager.store_query_result(
                            question=question,
                            sql=sql,
                            result=result,
                            tenant_code=tenant_code
                        )
                        logger.info(f"Cached result for question: {question}")
                    except Exception as e:
                        logger.warning(f"Failed to cache result: {str(e)}")

            except Exception as e:
                logger.error(f"Query execution error: {str(e)}")
                db_session.rollback()  # Rollback failed transaction

                error_message = str(e)
                end_time = datetime.utcnow()
                response_time = int(
                    (end_time - start_time).total_seconds() * 1000)

                self._record_query_performance(
                    db_session,
                    question,
                    False,
                    error_message,
                    response_time,
                    topic_category,
                    tokens_used
                )

                # Provide error message in appropriate language
                if is_dutch:
                    # Translate common error messages to Dutch
                    if "missing FROM-clause" in error_message:
                        error_message = "SQL-fout: ontbrekende FROM-clausule in de query"
                    elif "syntax error" in error_message:
                        error_message = "SQL-syntaxfout in de query"
                    elif "column" in error_message and "does not exist" in error_message:
                        error_message = "SQL-fout: een kolom in de query bestaat niet"
                    else:
                        error_message = f"Fout bij het uitvoeren van de query: {error_message}"

                return {
                    'success': False,
                    'error': error_message,
                    'followup_questions': [
                        "Wat zijn de meest voorkomende onderwerpen in onze gesprekken van de afgelopen maand?" if is_dutch else "What are the most common topics in our calls from the last month?",
                        "Hoe is het klantsentiment in de loop van de tijd veranderd?" if is_dutch else "How has customer sentiment changed over time?",
                        "Kunt u mij de verdeling van gespreksonderwerpen per sentiment tonen?" if is_dutch else "Can you show me the breakdown of call topics by sentiment?"
                    ]
                }

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
                tenant_code
            )

            # Record performance data for successful query
            end_time = datetime.utcnow()
            response_time = int(
                (end_time - start_time).total_seconds() * 1000)

            self._record_query_performance(
                db_session,
                question,
                True,
                None,
                response_time,
                topic_category,
                tokens_used
            )

            return {
                'success': True,
                'response': result,
                'followup_questions': followup_questions
            }

        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}")

            # Record performance data for failed query
            end_time = datetime.utcnow()
            response_time = int(
                (end_time - start_time).total_seconds() * 1000)
            error_message = str(e)

            try:
                self._record_query_performance(
                    db_session,
                    question,
                    False,
                    error_message,
                    response_time,
                    topic_category,
                    tokens_used
                )
            except Exception as perf_error:
                logger.error(
                    f"Failed to record performance data: {perf_error}")

            # Provide error message in appropriate language
            if is_dutch:
                error_message = f"Fout bij het verwerken van uw vraag: {error_message}"

            return {
                'success': False,
                'error': error_message,
                'followup_questions': [
                    "Wat zijn de meest voorkomende onderwerpen in onze gesprekken?" if is_dutch else "What are the most common topics in our calls?",
                    "Hoe is het klantsentiment in de loop van de tijd veranderd?" if is_dutch else "How has customer sentiment changed over time?",
                    "Kunt u mij de verdeling van gespreksonderwerpen per sentiment tonen?" if is_dutch else "Can you show me the breakdown of call topics by sentiment?"
                ]
            }

    async def execute_query(self, sql: str, params: dict, db_session: Session) -> str:
        """Execute SQL query and format results"""
        try:
            # Execute query
            logger.info(f"Executing SQL with params {params}:\n{sql}")
            result = db_session.execute(text(sql), params)
            rows = result.fetchall()
            column_names = result.keys()

            # Log result count
            row_count = len(rows)
            logger.info(f"Query returned {row_count} rows")

            # Format results into readable text
            if not rows:
                return "No results found for your query."

            # Format output based on number of rows
            if len(rows) == 1:
                # Single row result
                return "\n".join(f"{k}: {v}" for k, v in zip(column_names, rows[0]))
            else:
                # Multiple row result with row numbers
                output = []
                # Show first 10 rows with numbers
                for i, row in enumerate(rows[:10], 1):
                    row_str = ", ".join(
                        f"{k}: {v}" for k, v in zip(column_names, row))
                    output.append(f"{i}. {row_str}")

                # Add note if there are more rows
                if len(rows) > 10:
                    extra_count = len(rows) - 10
                    output.append(
                        f"\n(Showing top 10 results of {len(rows)} total)")

                return "\n".join(output)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise ValueError(f"Error executing query: {str(e)}")

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
            # Get last 10 interactions for enhanced context instead of 3
            recent_interactions = self.conversation_history.get(
                tenant_code, {}).get(conversation_id, [])[-10:]

            # Build conversation history for context awareness
            conversation_context = ""
            for interaction in recent_interactions:
                if 'question' in interaction and 'result' in interaction:
                    conversation_context += f"Q: {interaction['question']}\nA: {interaction['result']}\n\n"

            # Analyze current response content
            response_lower = response.lower()
            current_q = question.lower()

            # Detect language (Dutch vs English)
            is_dutch = any(dutch_word in current_q for dutch_word in
                           ['wat', 'hoe', 'waarom', 'welke', 'kunnen', 'waar', 'wie', 'wanneer', 'onderwerp'])

            # Extract numerical values for aggregation insights
            numeric_values = {}
            for line in response.split('\n'):
                line_lower = line.lower()
                # Look for percentages, counts, and other metrics
                if '%' in line or any(metric in line_lower for metric in ['count', 'rate', 'average', 'total', 'mentions']):
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        numeric_values[key] = value

            # Extract current topics from response with more detailed parsing
            topics_mentioned = []
            issues_mentioned = []
            for line in response.split('\n'):
                line_lower = line.lower()
                # Extract topics
                if 'topic:' in line_lower or 'onderwerp:' in line_lower:
                    try:
                        parts = line.split(':')
                        if len(parts) > 1:
                            topic_part = parts[1].strip()
                            # Clean up topic by removing trailing commas and metrics
                            if ',' in topic_part:
                                topic = topic_part.split(',')[0].strip()
                            else:
                                topic = topic_part
                            topics_mentioned.append(topic)
                    except Exception:
                        continue

            # Extract potential issues
            if any(issue_word in line_lower for issue_word in ['issue', 'problem', 'complaint', 'negative', 'negatief', 'klacht', 'probleem']):
                issues_mentioned.append(line)

            # Track key metrics and analysis dimensions from response
            metrics = {
                'sentiment_mentioned': any(x in response_lower for x in ['sentiment', 'positief', 'negatief', 'satisfaction']),
                'topics_mentioned': any(x in response_lower for x in ['topic', 'onderwerp']),
                'time_mentioned': any(x in response_lower for x in ['time', 'period', 'week', 'month', 'recent']),
                'volume_mentioned': any(x in response_lower for x in ['count', 'mentions', 'calls']),
                'trends_mentioned': 'trend' in response_lower,
                'technical_support': any(x in response_lower for x in ['technische ondersteuning', 'technical support']),
                'customer_service': any(x in response_lower for x in ['klantenservice', 'customer service']),
                'positive_rate': any(x in response_lower for x in ['positive_rate', 'satisfaction_rate']),
                'has_numbers': bool(numeric_values),
                'has_issues': bool(issues_mentioned)
            }

            # If we have specific topics in the response with issues, prioritize issue-focused questions
            if topics_mentioned and metrics['has_issues']:
                main_topic = topics_mentioned[0]
                if is_dutch:
                    return [
                        f"Wat zijn de top 3 specifieke problemen binnen het onderwerp '{main_topic}'?",
                        f"Hoeveel klanten hebben deze problemen gemeld en wat is het percentage ten opzichte van het totaal?",
                        f"Wat is de trend van deze problemen in vergelijking met vorige maand?"
                    ]
                else:
                    return [
                        f"What are the top 3 specific issues within the '{main_topic}' topic?",
                        f"How many customers reported these issues and what percentage of total calls do they represent?",
                        f"What is the trend of these issues compared to last month?"
                    ]

            # If we have specific topics in the response, prioritize topic insights with aggregates
            if topics_mentioned:
                main_topic = topics_mentioned[0]
                if is_dutch:
                    return [
                        f"Wat zijn de belangrijkste klantproblemen binnen '{main_topic}' en hoe vaak komen deze voor?",
                        f"Hoe is de verdeling van positieve vs. negatieve gesprekken over '{main_topic}'?",
                        f"Welke subtopics komen het meest voor samen met '{main_topic}' in recente gesprekken?"
                    ]
                else:
                    return [
                        f"What are the key customer issues within '{main_topic}' and how frequently do they occur?",
                        f"What's the distribution of positive vs. negative calls about '{main_topic}'?",
                        f"Which subtopics most frequently appear with '{main_topic}' in recent conversations?"
                    ]

            # If response shows concerning metrics or negative sentiment
            if 'negative' in response_lower or 'negatief' in response_lower:
                if is_dutch:
                    return [
                        "Wat zijn de top 5 onderwerpen die negatief sentiment veroorzaken?",
                        "Welke specifieke problemen leiden tot de meeste negatieve reacties?",
                        "Is er een patroon in tijdstip of datum waarop deze negatieve gesprekken plaatsvinden?"
                    ]
                else:
                    return [
                        "What are the top 5 topics causing negative sentiment?",
                        "Which specific issues lead to the most negative reactions?",
                        "Is there a pattern in time or date when these negative conversations occur?"
                    ]

            # If response shows positive trends with numerical insights
            if (metrics['positive_rate'] or 'positief' in response_lower) and metrics['has_numbers']:
                if is_dutch:
                    return [
                        f"Welke onderwerpen hebben de hoogste klanttevredenheid en wat zijn de percentages?",
                        f"Wat is de trend van positieve gesprekken in de afgelopen 30 dagen?",
                        f"Welke best practices kunnen we identificeren uit deze positieve interacties?"
                    ]
                else:
                    return [
                        f"Which topics have the highest customer satisfaction and what are the percentages?",
                        f"What is the trend of positive conversations over the past 30 days?",
                        f"What best practices can we identify from these positive interactions?"
                    ]

            # If analyzing trends with time context
            if metrics['trends_mentioned'] and metrics['time_mentioned']:
                if is_dutch:
                    return [
                        f"Wat zijn de belangrijkste veranderingen in onderwerpen ten opzichte van vorige maand?",
                        f"Welke onderwerpen laten een stijgende trend zien en met hoeveel procent?",
                        f"Zijn er opkomende onderwerpen die extra aandacht nodig hebben?"
                    ]
                else:
                    return [
                        f"What are the key topic changes compared to last month?",
                        f"Which topics show an increasing trend and by what percentage?",
                        f"Are there emerging topics that need additional attention?"
                    ]

            # If looking at technical support with aggregates
            if metrics['technical_support'] and metrics['has_numbers']:
                if is_dutch:
                    return [
                        f"Wat zijn de top 3 technische problemen en hun frequentie?",
                        f"Hoe verschilt de gemiddelde gespreksduur tussen verschillende technische problemen?",
                        f"Welke technische problemen worden het meest herhaald in vervolgoproepen?"
                    ]
                else:
                    return [
                        f"What are the top 3 technical issues and their frequency?",
                        f"How does average call duration differ between technical issues?",
                        f"Which technical issues are most repeated in follow-up calls?"
                    ]

            # If analyzing customer service with volume metrics
            if metrics['customer_service'] and metrics['volume_mentioned']:
                if is_dutch:
                    return [
                        "Wat zijn de meest voorkomende klantenservice-onderwerpen en hun volumeverdeling?",
                        "Welke klantenservice-problemen leiden tot de langste gesprekken?",
                        "Hoe verandert het gespreksvolume van klantenservice gedurende de dag?"
                    ]
                else:
                    return [
                        "What are the most common customer service topics and their volume distribution?",
                        "Which customer service issues lead to the longest conversations?",
                        "How does customer service call volume change throughout the day?"
                    ]

            # Default questions based on aggregated insights for call analysis
            if is_dutch:
                return [
                    "Wat zijn de top 5 meest besproken onderwerpen en hun relatieve percentages?",
                    "Welke gesprekstrends hebben we gezien in de afgelopen twee weken?",
                    "Hoe is de verdeling van positieve, neutrale en negatieve gesprekken per onderwerp?"
                ]
            else:
                return [
                    "What are the top 5 most discussed topics and their relative percentages?",
                    "What call trends have we seen over the past two weeks?",
                    "How is the distribution of positive, neutral, and negative calls per topic?"
                ]

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            # Default fallback questions
            return [
                "What are the most common topics in our calls from the last month?",
                "Which specific issues appear most frequently in customer calls?",
                "What percentage of calls show positive vs. negative sentiment?"
            ]

    def _record_query_performance(self, db_session, query_text, was_answered, error_message, response_time, topic_category, tokens_used, is_cached=False):
        """Record query performance metrics"""
        try:
            # Add cache indicator to query text if cached
            query_prefix = "[CACHED] " if is_cached else ""

            stmt = text("""
                INSERT INTO query_performance 
                (query_text, was_answered, error_message, response_time, topic_category, tokens_used)
                VALUES (:query_text, :was_answered, :error_message, :response_time, :topic_category, :tokens_used)
            """)

            db_session.execute(stmt, {
                "query_text": f"{query_prefix}{query_text}",
                "was_answered": was_answered,
                "error_message": error_message,
                "response_time": response_time,
                "topic_category": topic_category,
                "tokens_used": tokens_used
            })

            db_session.commit()
            logger.debug(
                f"Recorded query performance: {response_time}ms, answered: {was_answered}")
        except Exception as e:
            logger.error(f"Failed to record query performance: {e}")
            # Don't raise the exception to avoid disrupting the main flow

    def _extract_topic_category(self, text):
        """Extract topic category from text"""
        try:
            text_lower = text.lower()

            # Define common categories and their keywords
            categories = {
                "technical_support": ["technical", "support", "technische", "ondersteuning", "error", "fout", "bug", "issue"],
                "customer_service": ["customer", "service", "klantenservice", "klant", "support"],
                "billing": ["billing", "invoice", "facturering", "factuur", "betaling", "payment"],
                "product": ["product", "feature", "functie", "kenmerk"],
                "sentiment": ["sentiment", "positive", "negative", "positief", "negatief", "neutral", "neutraal"],
                "trend": ["trend", "change", "verandering", "pattern", "patroon"]
            }

            # Check for category keywords in the text
            for category, keywords in categories.items():
                if any(keyword in text_lower for keyword in keywords):
                    return category

            return None
        except Exception:
            return None
