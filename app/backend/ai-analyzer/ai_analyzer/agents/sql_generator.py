# File: backend/ai-analyzer/ai_analyzer/agents/sql_generator.py


from typing import Dict, Any, Optional, Tuple, List
import logging
from langchain.prompts import ChatPromptTemplate
import json
import re
from .base import BaseAgent, AgentResponse, DatabaseContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLGeneratorAgent(BaseAgent):
    """Enhanced SQL generator for database queries"""

    def __init__(self, model_provider: str, model_name: str, api_key: Optional[str] = None,
                 base_context: Optional[str] = None):
        # Call parent's __init__ with all required arguments
        super().__init__(model_provider, model_name, api_key)
        # Store base_context as instance variable
        self.base_context = base_context or ""

        # Initialize OpenAI client
        if model_provider.lower() == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

        self.model_name = model_name

    async def process(self, question: str, db_context: DatabaseContext,
                      conversation_context: Optional[Any] = None,
                      tenant_code: str = None) -> AgentResponse:
        """Process a question and generate SQL with tenant filtering"""
        try:
            if not tenant_code:
                return AgentResponse(
                    success=False,
                    error_message="Tenant code is required",
                    content=None
                )

            # Create the prompt with explicit tenant context
            prompt = ChatPromptTemplate.from_template("""
                You are a database query generator for a multi-tenant system.
                Current tenant: {tenant_code}

                {base_context}

                CRITICAL REQUIREMENTS:
                1. EVERY query must start with this exact base CTE:
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

                2. For trending topics, always use this pattern:
                WITH base_data AS (
                    -- Base CTE as shown above
                ),
                topic_trends AS (
                    SELECT 
                        clean_topic as topic,
                        COUNT(*) as mention_count,
                        COUNT(*) FILTER (WHERE processing_date >= CURRENT_DATE - INTERVAL '7 days') as recent_mentions,
                        COUNT(*) FILTER (WHERE clean_sentiment = 'positief') as positive_mentions,
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

                Available tables: {table_info}
                Question: {question}

                Generate an SQL query that:
                1. MUST start with the exact base_data CTE shown above
                2. MUST keep the tenant_code filter exactly as shown
                3. MUST use clean_topic and clean_sentiment for consistency
                4. Should limit results appropriately (e.g., TOP N, LIMIT)

                Return only the SQL query, nothing else.
            """)

            # Create the chain without passing base_context
            chain = prompt | self.llm

            # Generate SQL with tenant context
            result = await chain.ainvoke({
                "tenant_code": tenant_code,
                "base_context": self.base_context,
                "table_info": json.dumps(db_context.table_schemas, indent=2),
                "question": question
            })

            # Extract and validate SQL
            sql = result.content if hasattr(result, 'content') else str(result)
            if not self._validate_tenant_filtering(sql, tenant_code):
                return AgentResponse(
                    success=False,
                    content=None,
                    error_message="Generated SQL missing required tenant filtering",
                    reformulated_question=self._suggest_reformulation(question)
                )

            return AgentResponse(
                success=True,
                content=sql
            )

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return AgentResponse(
                success=False,
                content=None,
                error_message=str(e)
            )

    def _validate_tenant_filtering(self, sql: str, tenant_code: str) -> bool:
        """Strictly validate tenant filtering in SQL"""
        required_patterns = [
            r"WITH\s+base_data\s+AS\s*\(",
            r"WHERE\s+.*t\.tenant_code\s*=\s*:tenant_code",
            r"FROM\s+transcription\s+t"
        ]

        sql_upper = sql.upper()
        return all(bool(re.search(pattern, sql_upper)) for pattern in required_patterns)

    def _suggest_reformulation(self, question: str) -> str:
        """Suggest a reformulation of the question"""
        if "top" in question.lower() or "trend" in question.lower():
            return f"Can you show me {question.lower()} from the last 60 days, ordered by recent activity?"
        return f"Can you analyze {question.lower()} from the last 60 days?"

    def generate_sql_query(self, question: str, tenant_code: str) -> str:
        """Generate SQL query with mandatory tenant filtering"""

        # Add example queries to help guide the model
        example_queries = """
        Example 1 - Topics in positive calls:
        WITH base_data AS (
            -- Base CTE as required above
        ),
        topic_analysis AS (
            SELECT 
                clean_topic as topic,
                COUNT(*) as mention_count,
                COUNT(*) FILTER (WHERE clean_sentiment = 'positief') as positive_count,
                ROUND(CAST(COUNT(*) FILTER (WHERE clean_sentiment = 'positief') * 100.0 / 
                    NULLIF(COUNT(*), 0) AS NUMERIC), 2) as positive_rate
            FROM base_data
            GROUP BY clean_topic
            HAVING COUNT(*) > 0
            ORDER BY mention_count DESC
        )
        SELECT * FROM topic_analysis LIMIT 10;

        Example 2 - Emerging topics:
        WITH base_data AS (
            -- Base CTE as required above
        ),
        recent_topics AS (
            SELECT 
                clean_topic as topic,
                COUNT(*) as total_mentions,
                COUNT(*) FILTER (WHERE processing_date >= CURRENT_DATE - INTERVAL '7 days') as recent_mentions,
                COUNT(*) FILTER (WHERE clean_sentiment = 'positief') as positive_mentions,
                COUNT(*) FILTER (WHERE clean_sentiment = 'negatief') as negative_mentions
            FROM base_data
            GROUP BY clean_topic
            HAVING COUNT(*) > 2
        )
        SELECT 
            topic,
            total_mentions,
            recent_mentions,
            ROUND(CAST(recent_mentions * 100.0 / NULLIF(total_mentions, 0) AS NUMERIC), 2) as recent_percentage,
            ROUND(CAST(positive_mentions * 100.0 / NULLIF(total_mentions, 0) AS NUMERIC), 2) as satisfaction_rate
        FROM recent_topics
        WHERE recent_mentions > 0
        ORDER BY recent_mentions DESC, total_mentions DESC
        LIMIT 10;

        Example 3 - Sentiment analysis:
        WITH base_data AS (
            -- Base CTE as required above
        ),
        sentiment_stats AS (
            SELECT 
                clean_topic as topic,
                clean_sentiment as sentiment,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY clean_topic), 2) as percentage
            FROM base_data
            GROUP BY clean_topic, clean_sentiment
            HAVING COUNT(*) > 2
        )
        SELECT * FROM sentiment_stats
        ORDER BY count DESC, percentage DESC
        LIMIT 15;

        Example 4 - Topic correlations:
        WITH base_data AS (
            -- Base CTE as required above
        ),
        topic_pairs AS (
            SELECT 
                a.clean_topic as topic1,
                b.clean_topic as topic2,
                COUNT(*) as co_occurrence_count,
                ROUND(
                    COUNT(*) * 100.0 / (
                        SELECT COUNT(*) FROM base_data 
                        WHERE clean_topic = a.clean_topic
                    ),
                    2
                ) as correlation_percentage
            FROM base_data a
            JOIN base_data b ON 
                a.transcription_id = b.transcription_id AND 
                a.clean_topic < b.clean_topic
            GROUP BY a.clean_topic, b.clean_topic
            HAVING COUNT(*) > 1
        )
        SELECT * FROM topic_pairs
        ORDER BY co_occurrence_count DESC, correlation_percentage DESC
        LIMIT 15;
        """

        messages = [
            {"role": "system", "content": f"""You are a SQL expert that generates PostgreSQL queries. 
                CRITICAL: Every query MUST start with this exact base_data CTE:
                
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
                        t.clid,
                        t.telephone_number,
                        t.call_direction
                    FROM transcription t
                    WHERE t.processing_date >= CURRENT_DATE - INTERVAL '60 days'
                    AND t.tenant_code = :tenant_code  -- THIS LINE IS MANDATORY
                )
                
                IMPORTANT NOTES:
                - Positive sentiment is stored as 'positief'
                - Negative sentiment is stored as 'negatief'
                - Neutral sentiment is stored as 'neutral'
                - Always use clean_sentiment for comparisons
                - When using window functions, use the aliased column names
                - For trending analysis, use EXTRACT(YEAR/MONTH) from processing_date
                - For topic correlations, use self-join on transcription_id
                - Always include proper ORDER BY and LIMIT clauses
                - Each CTE must be properly chained with commas
                - Always end with a SELECT statement
                
                {example_queries}
                
                Never modify or remove the tenant_code filter.
                Always use this exact CTE structure as the start of every query.
                """},
            {"role": "user", "content": question}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # Lower temperature for more consistent output
                max_tokens=500
            )

            generated_sql = response.choices[0].message.content.strip()

            # Remove markdown code block if present
            if generated_sql.startswith('```'):
                # Remove opening ```sql or ``` and closing ```
                generated_sql = generated_sql.split(
                    '\n', 1)[1]  # Remove first line
                generated_sql = generated_sql.rsplit(
                    '\n', 1)[0]  # Remove last line
                generated_sql = generated_sql.strip()

            # Validate the generated SQL has required tenant filtering
            if "WITH base_data AS" not in generated_sql or "t.tenant_code = :tenant_code" not in generated_sql:
                raise ValueError(
                    "Generated SQL missing required tenant filtering")

            return generated_sql

        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            raise
