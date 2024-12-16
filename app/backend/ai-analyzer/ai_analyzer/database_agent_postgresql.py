# ai_analyzer/database_agent_postgresql.py
from typing import List, Optional, Dict
from enum import Enum
from pydantic import BaseModel
import re
import logging
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseSequentialChain
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from ai_analyzer.config import DATABASE_URL
from ai_analyzer.cache_manager import DatabaseCacheManager
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLOperation(str, Enum):
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    ALTER = "ALTER"
    DROP = "DROP"
    TRUNCATE = "TRUNCATE"


class SQLGuardrails(BaseModel):
    allowed_operations: List[SQLOperation]
    allowed_tables: List[str]
    max_rows_per_query: int
    sensitive_tables: List[str]
    sensitive_columns: Dict[str, List[str]]
    blacklisted_keywords: List[str]


class DatabaseAgentValidator:
    def __init__(self):
        # Initializing guardrails our specific configurations
        self.guardrails = SQLGuardrails(
            allowed_operations=[SQLOperation.SELECT],
            allowed_tables=['company', 'transcription'],
            max_rows_per_query=1000,
            sensitive_tables=['user', 'users', 'user_memory'],
            sensitive_columns={
                'company': [],
                'transcription': []
            },
            blacklisted_keywords=[
                'DELETE', 'DROP', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE',
                'EXECUTE', 'SHELL', 'SYSTEM', 'GRANT', 'REVOKE'
            ]
        )

        self.dangerous_patterns = [
            # r";\s*(\w+)",  # Multiple statements
            r"--",         # SQL comments
            r"/\*.*?\*/",  # Multi-line comments
            r"xp_\w+",     # Extended stored procedures
            r"sp_\w+",     # System stored procedures
            r"WAITFOR",    # Time delay attacks
            r"OPENROWSET",  # Remote access
            r"BULK INSERT"  # Bulk operations
        ]

    def validate_query(self, query: str) -> tuple[bool, Optional[str]]:
        """
        Comprehensive query validation against security rules.
        Returns (is_valid, error_message).
        """
        if not query:
            return False, "Empty query is not allowed"

        query = query.strip().upper()

        # 1. Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Query contains dangerous pattern: {pattern}"

        # 2. Validate operation type
        first_word = query.split()[0]
        if first_word not in [op.value for op in self.guardrails.allowed_operations]:
            return False, f"Operation {first_word} is not allowed"

        # 3. Check for blacklisted keywords
        for keyword in self.guardrails.blacklisted_keywords:
            if keyword.upper() in query:
                return False, f"Query contains blacklisted keyword: {keyword}"

        # 4. Extract and validate tables
        tables = self._extract_tables(query)

        # Check for sensitive tables
        for table in tables:
            if table.lower() in self.guardrails.sensitive_tables:
                return False, f"Access to sensitive table {table} is not allowed"

            if table.lower() not in self.guardrails.allowed_tables:
                return False, f"Table {table} is not in the allowed tables list"

        # 5. Validate sensitive columns
        for table in tables:
            if table in self.guardrails.sensitive_columns:
                sensitive_cols = self.guardrails.sensitive_columns[table]
                for col in sensitive_cols:
                    if col.upper() in query:
                        return False, f"Access to sensitive column {col} is not allowed"

        # 6. Validate row limit for SELECT queries
        # if not self._validate_row_limit(query):
        #    return False, f"Query must include LIMIT clause not exceeding {self.guardrails.max_rows_per_query} rows"

        return True, None

    def _extract_tables(self, query: str) -> List[str]:
        """
        Extract table names from the query, properly handling aliases.

        Parameters:
        query (str): SQL query string

        Returns:
        List[str]: List of base table names without aliases
        """
        tables = []

        # Enhanced pattern to match tables with optional aliases
        table_pattern = r'\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*(?:\s+(?:AS\s+)?[A-Za-z0-9_]+)?)'

        # Find all table references
        matches = re.finditer(table_pattern, query, re.IGNORECASE)

        for match in matches:
            # Extract the full table reference (including potential alias)
            table_full = match.group(1)
            # Split on whitespace and take first part (base table name)
            base_table = table_full.split()[0].lower()
            tables.append(base_table)

        # Return unique table names
        return list(set(tables))

    def _validate_row_limit(self, query: str) -> bool:
        """Validate that SELECT queries have appropriate LIMIT clause."""
        if not query.startswith('SELECT'):
            return True

        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        if not limit_match:
            return False

        limit_value = int(limit_match.group(1))
        return limit_value <= self.guardrails.max_rows_per_query

    def suggest_query_fix(self, query: str) -> Optional[str]:
        """Suggest fixes for invalid queries."""
        fixed_query = query.strip()

        # Add LIMIT clause if missing
        if fixed_query.upper().startswith('SELECT') and 'LIMIT' not in fixed_query.upper():
            fixed_query = f"{fixed_query} LIMIT {
                self.guardrails.max_rows_per_query}"

        # Remove multiple statements
        if ';' in fixed_query:
            fixed_query = fixed_query.split(';')[0]

        # Remove comments
        fixed_query = re.sub(r'--.*$', '', fixed_query, flags=re.MULTILINE)
        fixed_query = re.sub(r'/\*.*?\*/', '', fixed_query, flags=re.DOTALL)

        return fixed_query if fixed_query != query else None


def get_db_connection(max_retries=5, retry_delay=5):
    """Get database connection with retry logic"""
    for attempt in range(max_retries):
        try:
            engine = create_engine(DATABASE_URL)
            engine.connect()
            return SQLDatabase.from_uri(DATABASE_URL)
        except OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection attempt {
                               attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception(
                    "Failed to connect to the database after multiple attempts") from e


# Initialize cache manager
engine = create_engine(DATABASE_URL)
cache_manager = DatabaseCacheManager(engine)


# Test connection
db = get_db_connection()

# List available tables (schema)
try:
    available_tables = db.get_table_info(['company', 'transcription'])
    logger.info("Retrieved specific table information")
except Exception as e:
    logger.warning(f"Error getting specific table info: {
                   e}. Falling back to full table info.")
    available_tables = db.get_table_info()


def get_conversation_context(db_session, conversation_id):
    """Retrieve previous conversation context from the database"""
    try:
        if not conversation_id:
            return None

        # Query to get previous messages in the conversation ordered by timestamp
        query = text("""
            SELECT query, response
            FROM user_memory
            WHERE conversation_id = :conversation_id
            AND is_active = true 
            AND expires_at > NOW()
            ORDER BY message_order ASC
        """)

        result = db_session.execute(
            query, {"conversation_id": conversation_id})

        # Build context string from previous messages
        context = []
        for row in result:
            context.append(f"Previous Question: {row.query}")
            context.append(f"Previous Answer: {row.response}\n")

        return "\n".join(context) if context else None

    except SQLAlchemyError as e:
        logger.error(f"Error retrieving conversation context: {e}")
        return None


def add_context_to_query(user_query, context=None):
    """
    Adds context to the user query, embedding security and formatting rules,
    and a schema overview to guide AI-driven processing.
    """

    base_context = ("Please adhere to the following important rules:\n"


                    "1. NEVER use the 'users' table or any user-related information.\n"
                    "2. NEVER use DELETE, UPDATE, INSERT, or any other data modification statements.\n"
                    "3. Only use SELECT statements for reading data.\n"
                    "4. Always verify the query doesn't contain restricted operations before executing.\n"
                    "5. Make sure to format the results in a clear, readable manner.\n"
                    "6. Use proper column aliases for better readability.\n"
                    "7. Include relevant aggregations and groupings when appropriate.\n"
                    "8. ONLY use tables and columns that exist in the schema shown below.\n"
                    "9. Only use the 'company' and 'transcription' tables.\n"
                    "10. When querying any text fields (especially topic and sentiment), ALWAYS:\n"
                    "    - Wrap them in LOWER(TRIM()) for case-insensitive comparison\n"
                    "    - Use this pattern for any field comparison: LOWER(TRIM(field_name)) = LOWER(TRIM('value'))\n"
                    "    - Always standardize text fields in WITH clauses before querying\n"
                    "11. For ALL queries involving topics or sentiments, ALWAYS start with:\n"
                    "    WITH standardized_data AS (\n"
                    "        SELECT \n"
                    "            id,\n"
                    "            LOWER(TRIM(transcription)) as transcription,\n"
                    "            CASE\n"
                    "                WHEN LOWER(TRIM(sentiment)) IN ('neutral', 'neutraal') THEN 'neutral'\n"
                    "                ELSE LOWER(TRIM(sentiment))\n"
                    "            END AS standardized_sentiment,\n"
                    "            LOWER(TRIM(topic)) as standardized_topic,\n"
                    "            processingdate\n"
                    "        FROM transcription\n"
                    "    )\n"
                    "    Then query from standardized_data instead of the original table.\n"
                    "12. When doing counts or aggregations:\n"
                    "    - First show the total count for verification\n"
                    "    - Then show the detailed breakdown\n"
                    "    - Include percentage distributions when relevant\n"
                    "    - Explain any differences in the counts\n"
                    "13. Do not include any Markdown formatting like triple backticks or code blocks in any part of your response.\n"
                    "14. Do not prefix your SQL queries with 'SQL' or any other language identifiers.\n"
                    "15. Always start with the standardized_data CTE before any topic or sentiment analysis.\n\n"
                    f"Schema of available tables:\n{available_tables}\n\n"

                    )

    # Add previous conversation context if available
    if context:
        base_context += "Previous Conversation Context:\n"
        base_context += f"{context}\n\n"
        base_context += "Please consider the above conversation context when answering the following question:\n"

    return f"{base_context} Please only use these tables in your SQL query and do not make up non-existing ones to answer the following question: {user_query}"


def generate_db_response(question: str, context: Optional[str] = None) -> str:
    """Generate new response using OpenAI with context length handling"""
    validator = DatabaseAgentValidator()
    db = get_db_connection()
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv('AI_ANALYZER_OPENAI_API_KEY'),
        model_name='gpt-3.5-turbo'
    )
    db_chain = SQLDatabaseSequentialChain.from_llm(
        llm, db, verbose=True, use_query_checker=True)

    processed_query = add_context_to_query(question, context)
    sql_match = re.search(r'SELECT.*?(?=\n|$)',
                          processed_query, re.IGNORECASE | re.DOTALL)

    if sql_match:
        sql_query = sql_match.group(0)
        is_valid, error_message = validator.validate_query(sql_query)

        if not is_valid:
            suggested_fix = validator.suggest_query_fix(sql_query)
            error_response = f"Query validation failed: {error_message}"
            if suggested_fix:
                error_response += f"\nSuggested fix: {suggested_fix}"
            return error_response

        try:
            start_time = time.time()
            result = db_chain.run(processed_query)
            response_time = int((time.time() - start_time)
                                * 1000)  # Convert to milliseconds

            # Track successful query
            cache_manager.track_query_performance(
                query=question,
                was_answered=True,
                response_time=response_time,
                topic_category=detect_query_topic(question),
                tokens_used=estimate_tokens_used(result)
            )

            return result

        except Exception as e:
            error_msg = str(e).lower()
            response_time = int((time.time() - start_time) * 1000)

            # Check for context length related errors
            if any(phrase in error_msg for phrase in [
                "context length", "maximum context length", "too many tokens",
                "context window", "token limit"
            ]):
                # Track failed query
                cache_manager.track_query_performance(
                    query=question,
                    was_answered=False,
                    response_time=response_time,
                    error_message="Context length exceeded",
                    topic_category=detect_query_topic(question)
                )

                return (
                    "The requested time range contains too much data to process. Please try:\n"
                    "1. Reducing the time range (e.g., last week instead of last month)\n"
                    "2. Narrowing down your search criteria\n"
                    "3. Breaking your question into smaller parts\n"
                    "\nFor example, if you asked about a year of data, try asking about a month or a quarter instead."
                )

            # Track other types of failures
            cache_manager.track_query_performance(
                query=question,
                was_answered=False,
                response_time=response_time,
                error_message=str(e),
                topic_category=detect_query_topic(question)
            )

            return f"An error occurred: {str(e)}"
    else:
        cache_manager.track_query_performance(
            query=question,
            was_answered=False,
            error_message="No valid SQL query found",
            topic_category="invalid_query"
        )
        return "No valid SQL query found in the response"


def detect_query_topic(question: str) -> str:
    """Detect the topic category of a question"""
    question_lower = question.lower()

    categories = {
        'sentiment_analysis': ['sentiment', 'positive', 'negative', 'neutral'],
        'topic_analysis': ['topic', 'subjects', 'themes'],
        'time_analysis': ['trend', 'over time', 'period', 'duration'],
        'company_specific': ['company', 'client', 'customer'],
        'performance_metrics': ['average', 'count', 'total', 'number of'],
        'comparison': ['compare', 'difference', 'versus', 'vs']
    }

    for category, keywords in categories.items():
        if any(keyword in question_lower for keyword in keywords):
            return category

    return 'general'


def estimate_tokens_used(response: str) -> int:
    """Estimate the number of tokens used in a response"""
    # Rough estimation: 1 token ≈ 4 characters for English text
    return len(response) // 4


def answer_question(question: str, conversation_id=None, db_session=None):
    """Enhanced question answering with performance tracking and error handling"""
    try:
        logger.info(f"Received question: {question}")
        start_time = time.time()

        # Check cache first
        is_cached, cached_response, has_new_records = cache_manager.check_cache(
            question)

        if is_cached and not has_new_records:
            response_time = int((time.time() - start_time) * 1000)
            logger.info("Using cached response")

            # Track cached response
            cache_manager.track_query_performance(
                query=question,
                was_answered=True,
                response_time=response_time,
                topic_category=detect_query_topic(question),
                tokens_used=estimate_tokens_used(cached_response)
            )

            return cached_response

        # Get conversation context if available
        context = None
        if conversation_id and db_session:
            context = get_conversation_context(db_session, conversation_id)
            logger.info(f"Retrieved conversation context for ID {
                        conversation_id}")

        # Generate new response
        response = generate_db_response(question, context)

        # Cache the new response if successful
        if response and "error" not in response.lower():
            cache_manager.cache_response(question, response)
            logger.info("Cached new response")

        return response

    except Exception as e:
        # Track unexpected errors
        cache_manager.track_query_performance(
            query=question,
            was_answered=False,
            error_message=f"Unexpected error: {str(e)}",
            topic_category=detect_query_topic(question)
        )

        logger.error(f"Error processing question: {e}")
        return f"An error occurred while processing your question: {e}"


# Example usage
if __name__ == "__main__":
    # Sample questions to test the agent
    sample_questions = [
        "What are the top 10 topics discussed in all transcriptions?",
        "What is the overall sentiment trend of our calls with Company X over the past month?",
        "Which companies have the highest average positive sentiment in their calls?",
        "How does the sentiment of calls last week compare to the previous week?",
        "What are the most frequently discussed topics in calls with Company Y?",
        "What are the top concerns expressed by customers in the last quarter?",
        "Which topics are associated with negative sentiments?",
        "What is the average call duration with Company Z?",
        "How many calls were made to each company last month?",
        "What is the correlation between call frequency and customer satisfaction?",
        "Are there any emerging topics in customer calls that we should be aware of?",
        "How has the sentiment changed over time for Company A?",
        "Which companies show an increasing trend in negative call sentiments?",
        "Compare the average sentiment scores between Company X and Company Y.",
        "Which topics lead to higher customer satisfaction?",
        "What differences are there in call topics between domestic and international clients?"
    ]

    # Example usage with a single question
    # question = "What are the top 10 topics discussed in all transcriptions?"
    # answer_question(question)

    # Alternatively, iterate through all sample questions
    # for q in sample_questions:
    #     answer_question(q)
    #     print("\n" + "-"*50 + "\n")
