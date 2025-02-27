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

# Add at the top of the file, after imports
DEFAULT_DAYS_LOOKBACK = 300


class SQLGeneratorAgent(BaseAgent):
    """Enhanced SQL generator for database queries"""

    def __init__(self, model_provider: str, model_name: str, api_key: Optional[str] = None,
                 base_context: Optional[str] = None):
        # Call parent's __init__ with all required arguments
        super().__init__(model_provider, model_name, api_key)
        # Store base_context as instance variable
        self.base_context = base_context or ""
        self.days_lookback = DEFAULT_DAYS_LOOKBACK

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

            # Create the prompt with explicit tenant context and enhanced aggregation instructions
            prompt = ChatPromptTemplate.from_template("""
                You are a database query generator for a multi-tenant system specialized in call center analysis.
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
                    WHERE t.processing_date >= CURRENT_DATE - INTERVAL '300 days'
                    AND t.tenant_code = :tenant_code
                )

                2. For trending topics with issue analysis, always use this pattern:
                WITH base_data AS (
                    -- Base CTE as shown above
                ),
                topic_trends AS (
                    SELECT 
                        clean_topic as topic,
                        COUNT(*) as mention_count,
                        COUNT(*) FILTER (WHERE processing_date >= CURRENT_DATE - INTERVAL '7 days') as recent_mentions,
                        COUNT(*) FILTER (WHERE clean_sentiment = 'positief') as positive_mentions,
                        COUNT(*) FILTER (WHERE clean_sentiment = 'negatief') as negative_mentions,
                        COUNT(*) FILTER (WHERE clean_sentiment = 'neutral') as neutral_mentions,
                        ROUND(COUNT(*) FILTER (WHERE clean_sentiment = 'positief') * 100.0 / NULLIF(COUNT(*), 0), 2) as positive_percentage,
                        ROUND(COUNT(*) FILTER (WHERE clean_sentiment = 'negatief') * 100.0 / NULLIF(COUNT(*), 0), 2) as negative_percentage,
                        ROUND(AVG(CASE 
                            WHEN clean_sentiment = 'positief' THEN 1
                            WHEN clean_sentiment = 'negatief' THEN -1
                            ELSE 0 
                        END)::numeric, 2) as sentiment_score,
                        AVG(call_duration_secs) as avg_call_duration,
                        MIN(processing_date) as first_occurrence,
                        MAX(processing_date) as last_occurrence
                    FROM base_data
                    WHERE clean_topic IS NOT NULL
                    GROUP BY clean_topic
                    HAVING COUNT(*) > 5
                    ORDER BY recent_mentions DESC, mention_count DESC
                )

                3. For time-based analysis, include percentage changes:
                WITH base_data AS (
                    -- Base CTE as shown above
                ),
                time_periods AS (
                    SELECT
                        clean_topic,
                        clean_sentiment,
                        CASE
                            WHEN processing_date >= CURRENT_DATE - INTERVAL '7 days' THEN 'current_week'
                            WHEN processing_date >= CURRENT_DATE - INTERVAL '14 days' THEN 'previous_week'
                            WHEN processing_date >= CURRENT_DATE - INTERVAL '30 days' THEN 'current_month'
                            WHEN processing_date >= CURRENT_DATE - INTERVAL '60 days' THEN 'previous_month'
                            ELSE 'older'
                        END as time_period,
                        COUNT(*) as period_count
                    FROM base_data
                    GROUP BY clean_topic, clean_sentiment, time_period
                ),
                period_comparisons AS (
                    SELECT
                        clean_topic,
                        clean_sentiment,
                        SUM(CASE WHEN time_period = 'current_week' THEN period_count ELSE 0 END) as current_week_count,
                        SUM(CASE WHEN time_period = 'previous_week' THEN period_count ELSE 0 END) as previous_week_count,
                        CASE 
                            WHEN SUM(CASE WHEN time_period = 'previous_week' THEN period_count ELSE 0 END) > 0 
                            THEN ROUND((SUM(CASE WHEN time_period = 'current_week' THEN period_count ELSE 0 END) - 
                                      SUM(CASE WHEN time_period = 'previous_week' THEN period_count ELSE 0 END)) * 100.0 / 
                                      NULLIF(SUM(CASE WHEN time_period = 'previous_week' THEN period_count ELSE 0 END), 0), 2)
                            ELSE NULL
                        END as weekly_percentage_change
                    FROM time_periods
                    GROUP BY clean_topic, clean_sentiment
                )

                Available tables: {table_info}
                Question: {question}

                Generate an SQL query that:
                1. MUST start with the exact base_data CTE shown above
                2. MUST keep the tenant_code filter exactly as shown
                3. MUST use clean_topic and clean_sentiment for consistency
                4. INCLUDE percentages, aggregated metrics, and counts whenever possible
                5. Include insights about specific issues within topics when relevant
                6. ALWAYS include trend comparisons or time-based patterns when possible
                7. Limit results appropriately (e.g., TOP N, LIMIT)

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
            if not self.validate_sql(sql, tenant_code):
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

    def validate_sql(self, sql, tenant_code):
        """Validate SQL for security and correctness"""
        try:
            # Check for basic SQL injection patterns
            if any(pattern in sql.lower() for pattern in [
                "drop table", "drop database", "truncate table",
                "delete from", "update ", "insert into", "alter table"
            ]):
                return False, "SQL contains disallowed modification statements"

            # Check for tenant isolation
            tenant_filter_pattern = r"T\.TENANT_CODE\s*=\s*:TENANT_CODE"
            has_tenant_filter = bool(
                re.search(tenant_filter_pattern, sql, re.IGNORECASE))

            # Check for base CTE pattern
            has_base_cte = "WITH base_data AS (" in sql

            # Check for correct table name
            table_name = f"transcription_{tenant_code}"
            has_correct_table = table_name in sql

            # Check for column errors and fix them
            column_errors = []

            # Fix t.clean_topic reference - this is a derived column that needs to be created
            if "t.clean_topic" in sql and "LOWER(TRIM(t.topic))" not in sql:
                # Replace t.clean_topic with LOWER(TRIM(t.topic))
                sql = re.sub(
                    r't\.clean_topic\s*=\s*\'([^\']+)\'',
                    r"LOWER(TRIM(t.topic)) = '\1'",
                    sql
                )
                column_errors.append(
                    "Fixed t.clean_topic reference to use LOWER(TRIM(t.topic))")

            # Make sure clid is included in the base_data CTE if it's used later
            if "clid" in sql and "clid" not in sql.split("base_data AS (")[1].split(")")[0]:
                # Add clid to the base_data CTE
                base_cte_pattern = r"(WITH\s+base_data\s+AS\s*\(\s*SELECT\s+[^)]+)"
                base_cte_match = re.search(
                    base_cte_pattern, sql, re.IGNORECASE | re.DOTALL)

                if base_cte_match:
                    # Check if we need to add t.clid to the SELECT list
                    select_part = base_cte_match.group(1)
                    if "t.clid" not in select_part:
                        # Find the last column in the SELECT list
                        if "," in select_part:
                            # Add t.clid after the last column
                            modified_select = select_part.rstrip() + ",\n        t.clid"
                            sql = sql.replace(select_part, modified_select)
                            column_errors.append(
                                "Added missing column 't.clid' to base_data CTE")

            # Enhance phone number queries to check both telephone_number and clid
            if "telephone_number" in sql:
                # Find phone number equality or LIKE check with more flexible pattern matching
                phone_patterns = [
                    r'(telephone_number\s*=\s*[\'"]([^\'"]*)[\'"]\s*(?:OR|AND|;|$|\)))',
                    r'(telephone_number\s*LIKE\s*[\'"]([^\'"]*)[\'"]\s*(?:OR|AND|;|$|\)))'
                ]

                for pattern in phone_patterns:
                    phone_match = re.search(pattern, sql)
                    if phone_match:
                        phone_condition = phone_match.group(1)
                        phone_number = phone_match.group(2)

                        # Clean the phone number by removing special characters
                        clean_phone = re.sub(r'[/\s-]', '', phone_number)

                        # Replace with OR condition using LIKE with both original and cleaned number
                        new_condition = f"(telephone_number LIKE '%{phone_number}%' OR clid LIKE '%{phone_number}%' OR telephone_number LIKE '%{clean_phone}%' OR clid LIKE '%{clean_phone}%')"

                        # Make sure we're replacing the exact match
                        sql = sql.replace(phone_condition, new_condition)
                        column_errors.append(
                            f"Enhanced phone search to use LIKE with both telephone_number and clid for '{phone_number}' and cleaned version '{clean_phone}'")

                        # Add call_direction to the output if not already included
                        if "call_direction" not in sql.split("SELECT")[-1]:
                            # Find the SELECT statement
                            select_pattern = r'(SELECT\s+(?:[^;]+))\s+FROM'
                            select_match = re.search(
                                select_pattern, sql, re.IGNORECASE)

                            if select_match:
                                select_clause = select_match.group(1)
                                if "COUNT(*)" in select_clause:
                                    # For count queries, add call_direction as a group by
                                    new_select = select_clause.replace(
                                        "COUNT(*)", "call_direction, COUNT(*)")
                                    sql = sql.replace(
                                        select_clause, new_select)

                                    # Add GROUP BY if not present
                                    if "GROUP BY" not in sql:
                                        sql = sql.replace(
                                            ";", " GROUP BY call_direction;")
                                    column_errors.append(
                                        "Added call_direction to phone number query results")

                        # We found and replaced a pattern, so break the loop
                        break

            # Replace transcription with summary in final output
            if "transcription" in sql and "summary" in sql:
                # Replace transcription with summary in SELECT clauses outside of base_data CTE
                base_cte_end = sql.find(
                    "base_data AS (") + sql[sql.find("base_data AS ("):].find(")") + 1
                rest_of_sql = sql[base_cte_end:]

                # Replace transcription with summary in the rest of the SQL
                if "transcription" in rest_of_sql:
                    rest_of_sql = re.sub(
                        r'\btranscription\b', 'summary', rest_of_sql)
                    sql = sql[:base_cte_end] + rest_of_sql
                    column_errors.append(
                        "Replaced 'transcription' with 'summary' in output")

            # Rename clean_topic to topic in the final output for better readability
            final_select = sql.split("SELECT")[-1]
            if "clean_topic" in final_select and "as topic" not in final_select.lower():
                # Find if clean_topic is already being aliased
                if not re.search(r'clean_topic\s+as\s+\w+', final_select, re.IGNORECASE):
                    # Replace clean_topic with clean_topic as topic in the final SELECT
                    sql = re.sub(
                        r'(SELECT\s+(?:.*?,\s*)?)clean_topic(\s*,|\s*FROM|\s*$)',
                        r'\1clean_topic as topic\2',
                        sql,
                        flags=re.IGNORECASE
                    )
                    column_errors.append(
                        "Renamed 'clean_topic' to 'topic' in output for better readability")

            # Remove transcription_id from final output
            final_select = sql.split("SELECT")[-1]
            if "transcription_id" in final_select:
                # Find the final SELECT statement
                select_pattern = r'SELECT\s+(.*?)\s+FROM'
                select_match = re.search(
                    select_pattern, final_select, re.IGNORECASE | re.DOTALL)

                if select_match:
                    columns = select_match.group(1)
                    # Remove transcription_id from the column list
                    if "transcription_id" in columns:
                        new_columns = re.sub(
                            r'\btranscription_id\s*,\s*', '', columns)
                        new_columns = re.sub(
                            r',\s*\btranscription_id\b', '', new_columns)
                        new_columns = re.sub(
                            r'\btranscription_id\b', '', new_columns)

                        # Replace the columns in the final SELECT
                        sql = sql.replace(columns, new_columns)
                        column_errors.append(
                            "Removed 'transcription_id' from output")

            # Log validation details
            logger.info(f"SQL Validation Details:\n\
                    Base CTE present: {has_base_cte}\n\
                    Tenant filter present: {has_tenant_filter} (pattern: {tenant_filter_pattern})\n\
                    Correct table name: {has_correct_table}\n\
                    SQL Being Validated:\n\
                    {sql}\n\
                    {column_errors if column_errors else ''}")

            if not has_base_cte:
                return False, "SQL must use a base_data CTE for consistent tenant isolation"

            if not has_tenant_filter:
                return False, "SQL must include tenant isolation with t.tenant_code = :tenant_code"

            if not has_correct_table:
                return False, f"SQL must use the correct table name: {table_name}"

            return True, sql

        except Exception as e:
            logger.error(f"Error validating SQL: {e}")
            return False, f"SQL validation error: {str(e)}"

    def _suggest_reformulation(self, question: str) -> str:
        """Suggest a reformulation of the question"""
        if "top" in question.lower() or "trend" in question.lower():
            return f"Can you show me {question.lower()} from the last 300 days, ordered by recent activity?"
        return f"Can you analyze {question.lower()} from the last 300 days?"

    def generate_sql_query(self, question: str, tenant_code: str, conversation_context: str = "") -> str:
        """Generate SQL query with conversation context"""
        try:
            messages = [
                {"role": "system", "content": f"""You are a SQL expert that generates PostgreSQL queries.
                    {self.base_context}
                    
                    Current conversation context:
                    {conversation_context}
                    
                    CRITICAL REQUIREMENTS:
                    1. Return ONLY the SQL query, no explanations or markdown
                    2. The query MUST start with: WITH base_data AS (SELECT
                    3. The table name MUST be exactly: transcription_{tenant_code} t
                    4. The WHERE clause MUST include: t.tenant_code = :tenant_code
                    5. When referencing columns from base_data in subsequent CTEs, DO NOT use the 't' alias
                    6. In subsequent CTEs, use column names directly from base_data (e.g., 'summary' not 't.summary')
                    7. Do not modify these requirements under any circumstances
                    
                    Example of correct format:
                    WITH base_data AS (
                        SELECT t.* 
                        FROM transcription_{tenant_code} t
                        WHERE t.tenant_code = :tenant_code
                    ),
                    analysis AS (
                        SELECT
                            clean_topic,  -- Correct: no 't.' prefix in second CTE
                            COUNT(*) as count
                        FROM base_data
                        GROUP BY clean_topic
                    )
                    SELECT * FROM analysis;
                    """},
                {"role": "user", "content": question}
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1
            )

            generated_sql = response.choices[0].message.content.strip()

            # Clean up any markdown or explanations
            if '```' in generated_sql:
                sql_parts = generated_sql.split('```')
                for part in sql_parts:
                    if 'SELECT' in part.upper() or 'WITH' in part.upper():
                        generated_sql = part.strip()
                        break

            # Force correct table name
            generated_sql = re.sub(
                r'FROM\s+transcription(?:_\w+)?\s+t',
                f'FROM transcription_{tenant_code} t',
                generated_sql,
                flags=re.IGNORECASE
            )

            # Fix common error: using t.column in subsequent CTEs
            # Find all CTEs after the first one
            cte_pattern = r'WITH\s+base_data\s+AS\s*\([^)]+\)(,\s*[a-zA-Z0-9_]+\s+AS\s*\([^)]+\))*'
            cte_match = re.search(
                cte_pattern, generated_sql, re.DOTALL | re.IGNORECASE)

            if cte_match:
                cte_text = cte_match.group(0)
                # Replace t.column with just column in all CTEs after the first one
                base_cte_end = cte_text.find('),')
                if base_cte_end > 0:
                    subsequent_ctes = cte_text[base_cte_end:]
                    fixed_ctes = re.sub(
                        r't\.([a-zA-Z0-9_]+)', r'\1', subsequent_ctes, flags=re.IGNORECASE)
                    generated_sql = generated_sql.replace(
                        subsequent_ctes, fixed_ctes)

            # Validate the SQL
            if not self.validate_sql(generated_sql, tenant_code):
                logger.error(
                    f"Invalid SQL generated for tenant {tenant_code}: {generated_sql}")
                raise ValueError(
                    f"Generated SQL missing required tenant filtering for {tenant_code}")

            return generated_sql

        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            raise
