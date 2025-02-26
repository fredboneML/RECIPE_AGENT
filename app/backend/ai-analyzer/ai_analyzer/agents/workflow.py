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

    async def execute_query(self, sql: str, params: dict, db_session: Session) -> str:
        """Execute SQL query and format results with enhanced readability and insight-focused presentation"""
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

            # Detect potential value columns for better formatting
            numeric_columns = []
            date_columns = []
            topic_columns = []
            sentiment_columns = []
            percentage_columns = []

            # Identify column types for better formatting
            for col in column_names:
                col_lower = col.lower()
                # Check first row value type when available
                if results and col in results[0]:
                    value = results[0][col]

                    # Identify topic columns
                    if 'topic' in col_lower:
                        topic_columns.append(col)

                    # Identify sentiment columns
                    elif 'sentiment' in col_lower:
                        sentiment_columns.append(col)

                    # Identify date columns
                    elif isinstance(value, datetime):
                        date_columns.append(col)

                    # Identify percentage columns
                    elif 'percent' in col_lower or 'rate' in col_lower or '%' in col_lower:
                        percentage_columns.append(col)

                    # Identify numeric columns
                    elif isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                        numeric_columns.append(col)

            # Format output based on number of rows and column types
            if len(results) == 1:
                # Single row result - format for readability
                formatted_items = []
                for k, v in results[0].items():
                    formatted_value = v
                    # Format percentages
                    if k in percentage_columns and isinstance(v, (int, float)):
                        formatted_value = f"{v:.2f}%"
                    # Format dates
                    elif k in date_columns and v:
                        if isinstance(v, datetime):
                            formatted_value = v.strftime("%Y-%m-%d")
                    # Format numbers
                    elif k in numeric_columns and v:
                        if isinstance(v, (int, float)) and v > 1000:
                            formatted_value = f"{v:,}"

                    formatted_items.append(f"{k}: {formatted_value}")
                return "\n".join(formatted_items)

            else:
                # Multiple row result - prioritize important information
                # Check if this is a topic-based result set
                has_topic = any(col in topic_columns for col in column_names)
                has_sentiment = any(
                    col in sentiment_columns for col in column_names)

                # Sort results if meaningful
                if has_topic and any(col in numeric_columns for col in column_names):
                    # Find the most relevant numeric column to sort by
                    sort_col = next((col for col in column_names if 'count' in col.lower() or 'mentions' in col.lower()),
                                    next((col for col in numeric_columns), None))
                    if sort_col:
                        results = sorted(results, key=lambda x: x.get(sort_col, 0) if x.get(
                            sort_col) is not None else 0, reverse=True)

                # Limit to top 10 if there are too many results
                if len(results) > 10:
                    extra_count = len(results) - 10
                    results = results[:10]

                # Format each row
                output = []
                for i, row in enumerate(results):
                    # Enhanced row formatting
                    row_parts = []

                    # Always include topic first if available
                    for topic_col in topic_columns:
                        if topic_col in row:
                            row_parts.append(f"{topic_col}: {row[topic_col]}")

                    # Then include sentiment if available
                    for sentiment_col in sentiment_columns:
                        if sentiment_col in row:
                            row_parts.append(
                                f"{sentiment_col}: {row[sentiment_col]}")

                    # Format percentages
                    for k in percentage_columns:
                        if k in row and row[k] is not None:
                            if isinstance(row[k], (int, float)):
                                row_parts.append(f"{k}: {row[k]:.2f}%")
                            else:
                                row_parts.append(f"{k}: {row[k]}")

                    # Format other numeric values
                    for k in numeric_columns:
                        if k in row and k not in percentage_columns and row[k] is not None:
                            if isinstance(row[k], (int, float)) and row[k] > 1000:
                                row_parts.append(f"{k}: {row[k]:,}")
                            else:
                                row_parts.append(f"{k}: {row[k]}")

                    # Format dates
                    for k in date_columns:
                        if k in row and row[k] is not None:
                            if isinstance(row[k], datetime):
                                row_parts.append(
                                    f"{k}: {row[k].strftime('%Y-%m-%d')}")
                            else:
                                row_parts.append(f"{k}: {row[k]}")

                    # Include any remaining columns
                    for k, v in row.items():
                        if k not in topic_columns and k not in sentiment_columns and \
                                k not in percentage_columns and k not in numeric_columns and k not in date_columns:
                            row_parts.append(f"{k}: {v}")

                    # Combine parts into a single row
                    output.append(f"{i+1}. {', '.join(row_parts)}")

                # Add note about additional results
                if len(results) == 10 and extra_count > 0:
                    output.append(
                        f"\n(Showing top 10 results of {10 + extra_count} total)")

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
            # Get last 3 interactions for enhanced context
            recent_interactions = self.conversation_history.get(
                tenant_code, {}).get(conversation_id, [])[-3:]

            # Build conversation history for context awareness
            conversation_context = ""
            for interaction in recent_interactions:
                if 'question' in interaction and 'result' in interaction:
                    conversation_context += f"Q: {interaction['question']}\nA: {interaction['result']}\n\n"

            # Analyze current response content
            response_lower = response.lower()
            current_q = question.lower()

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

            # Detect language (Dutch vs English)
            is_dutch = any(dutch_word in current_q.lower() for dutch_word in
                           ['wat', 'hoe', 'waarom', 'welke', 'kunnen', 'waar', 'wie', 'wanneer', 'onderwerp'])

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
                        "Welke onderwerpen hebben de hoogste klanttevredenheid en wat zijn de percentages?",
                        "Wat is de trend van positieve gesprekken in de afgelopen 30 dagen?",
                        "Welke best practices kunnen we identificeren uit deze positieve interacties?"
                    ]
                else:
                    return [
                        "Which topics have the highest customer satisfaction and what are the percentages?",
                        "What is the trend of positive conversations over the past 30 days?",
                        "What best practices can we identify from these positive interactions?"
                    ]

            # If analyzing trends with time context
            if metrics['trends_mentioned'] and metrics['time_mentioned']:
                if is_dutch:
                    return [
                        "Wat zijn de belangrijkste veranderingen in onderwerpen ten opzichte van vorige maand?",
                        "Welke onderwerpen laten een stijgende trend zien en met hoeveel procent?",
                        "Zijn er opkomende onderwerpen die extra aandacht nodig hebben?"
                    ]
                else:
                    return [
                        "What are the key topic changes compared to last month?",
                        "Which topics show an increasing trend and by what percentage?",
                        "Are there emerging topics that need additional attention?"
                    ]

            # If looking at technical support with aggregates
            if metrics['technical_support'] and metrics['has_numbers']:
                if is_dutch:
                    return [
                        "Wat zijn de top 3 technische problemen en hun frequentie?",
                        "Hoe verschilt de gemiddelde gespreksduur tussen verschillende technische problemen?",
                        "Welke technische problemen worden het meest herhaald in vervolgoproepen?"
                    ]
                else:
                    return [
                        "What are the top 3 technical issues and their frequency?",
                        "How does average call duration differ between technical issues?",
                        "Which technical issues are most repeated in follow-up calls?"
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
