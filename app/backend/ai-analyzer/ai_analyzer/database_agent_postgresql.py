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
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseAgentValidator:
    def validate_query(self, query: str, tenant_code: str) -> tuple[bool, Optional[str]]:
        """Enhanced query validation with tenant isolation checks"""
        if not query or not tenant_code:
            return False, "Empty query or missing tenant code"

        query = query.strip().upper()

        # Check for tenant isolation
        if 'TENANT_CODE = :TENANT_CODE' not in query:
            return False, "Query must include tenant isolation"

        # Check for date filtering
        if 'CURRENT_DATE - INTERVAL' not in query:
            return False, "Query must include date filtering"

        # Rest of the validation logic...
        return True, None


class ConversationContext:
    def __init__(self):
        self.topics = {}  # Store mentioned topics and their occurrences
        self.current_focus = None  # Track current topic of focus
        self.last_query_result = None  # Store last query result

    def update_from_result(self, result_text: str):
        """Update context based on query results"""
        try:
            # Extract topics and occurrences from result text
            lines = result_text.split('\n')
            for line in lines:
                if '-' in line and 'occurrences' in line.lower():
                    parts = line.split('-')
                    if len(parts) == 2:
                        topic = parts[0].strip()
                        count = int(
                            re.search(r'(\d+)\s+occurrences', parts[1]).group(1))
                        self.topics[topic] = count

            # Set current focus to the first topic if not set
            if not self.current_focus and self.topics:
                self.current_focus = list(self.topics.keys())[0]

        except Exception as e:
            logger.error(f"Error updating context: {e}")

    def get_current_topic(self) -> Optional[str]:
        """Get the currently focused topic"""
        return self.current_focus

    def update_focus(self, query: str):
        """Update focus based on new query"""
        # Check if query mentions any known topics
        for topic in self.topics.keys():
            if topic.lower() in query.lower():
                self.current_focus = topic
                break

        # Handle "first", "second", etc. references
        ordinal_matches = re.findall(
            r'(first|second|third|fourth|fifth)', query.lower())
        if ordinal_matches:
            ordinal_map = {
                'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4
            }
            idx = ordinal_map.get(ordinal_matches[0])
            if idx is not None and idx < len(self.topics):
                self.current_focus = list(self.topics.keys())[idx]


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
    available_tables = db.get_table_info(['transcription'])
    logger.info("Retrieved specific table information")
except Exception as e:
    logger.warning(f"Error getting specific table info: {
                   e}. Falling back to full table info.")
    available_tables = db.get_table_info()


def get_conversation_context(db_session, conversation_id):
    """Enhanced conversation context retrieval"""
    try:
        if not conversation_id:
            return None, None

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

        # Build context string and ConversationContext object
        context_text = []
        context_obj = ConversationContext()

        for row in result:
            context_text.append(f"Previous Question: {row.query}")
            context_text.append(f"Previous Answer: {row.response}\n")
            context_obj.update_from_result(row.response)

        return "\n".join(context_text) if context_text else None, context_obj

    except SQLAlchemyError as e:
        logger.error(f"Error retrieving conversation context: {e}")
        return None, None


def add_context_to_query(user_query: str, text_context: Optional[str] = None, context_obj: Optional[ConversationContext] = None):
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
                    "10. Every query MUST start with defining the base_data CTE exactly like this:\n"
                    "    WITH base_data AS (\n"
                    "        SELECT \n"
                    "            t.id,\n"
                    "            t.transcription_id,\n"
                    "            t.transcription,\n"
                    "            t.topic,\n"
                    "            LOWER(TRIM(t.topic)) as clean_topic,\n"
                    "            t.summary,\n"
                    "            t.processingdate,\n"
                    "            t.sentiment,\n"
                    "            CASE\n"
                    "                WHEN LOWER(TRIM(t.sentiment)) IN ('neutral', 'neutraal') THEN 'neutral'\n"
                    "                ELSE LOWER(TRIM(t.sentiment))\n"
                    "            END AS clean_sentiment,\n"
                    "            c.clid,\n"
                    "            LOWER(TRIM(c.clid)) as clean_clid\n"
                    "        FROM transcription t\n"
                    "        LEFT JOIN company c ON t.transcription_id = c.transcription_id\n"
                    "    )\n"
                    "11. For topic analysis, your complete query should look like this:\n"
                    "    WITH base_data AS (\n"
                    "        -- Base data CTE definition as shown above\n"
                    "    ),\n"
                    "    topic_analysis AS (\n"
                    "        SELECT\n"
                    "            clean_topic as topic,\n"
                    "            COUNT(*) as total_count,\n"
                    "            COUNT(*) FILTER (WHERE clean_sentiment = 'positief') as positive_count,\n"
                    "            COUNT(*) FILTER (WHERE clean_sentiment = 'negatief') as negative_count,\n"
                    "            COUNT(*) FILTER (WHERE clean_sentiment = 'neutral') as neutral_count,\n"
                    "            ROUND(CAST(COUNT(*) FILTER (WHERE clean_sentiment = 'positief') * 100.0 / \n"
                    "                NULLIF(COUNT(*), 0) AS NUMERIC), 2) as satisfaction_rate\n"
                    "        FROM base_data\n"
                    "        GROUP BY clean_topic\n"
                    "        HAVING COUNT(*) > 0\n"
                    "    )\n"
                    "    SELECT * FROM topic_analysis ...\n"
                    "12. For time-based analysis, your complete query should look like this:\n"
                    "    WITH base_data AS (\n"
                    "        -- Base data CTE definition as shown above\n"
                    "    ),\n"
                    "    time_based_data AS (\n"
                    "        SELECT \n"
                    "            id,\n"
                    "            transcription_id,\n"
                    "            clean_topic,\n"
                    "            clean_sentiment,\n"
                    "            clean_clid,\n"
                    "            processingdate,\n"
                    "            CASE\n"
                    "                WHEN processingdate >= CURRENT_DATE - INTERVAL '7 days' THEN 'Current Week'\n"
                    "                WHEN processingdate >= CURRENT_DATE - INTERVAL '14 days' THEN 'Previous Week'\n"
                    "                WHEN processingdate >= CURRENT_DATE - INTERVAL '30 days' THEN 'Current Month'\n"
                    "                WHEN processingdate >= CURRENT_DATE - INTERVAL '60 days' THEN 'Previous Month'\n"
                    "            END AS time_period\n"
                    "        FROM base_data\n"
                    "        WHERE processingdate >= CURRENT_DATE - INTERVAL '300 days'\n"
                    "    )\n"
                    "    SELECT * FROM time_based_data ...\n"
                    "13. For text comparisons, ALWAYS use these patterns:\n"
                    "    - Exact match: clean_topic = 'value' or clean_sentiment = 'value'\n"
                    "    - Partial match: clean_topic LIKE '%value%'\n"
                    "14. For calculations ALWAYS use:\n"
                    "    - NULLIF(value, 0) for division\n"
                    "    - COALESCE(value, 0) for NULL handling\n"
                    "    - ROUND(CAST(value AS NUMERIC), 2) for decimals\n"
                    "15. For filtering dates ALWAYS use:\n"
                    "    - WHERE processingdate >= CURRENT_DATE - INTERVAL 'X days'\n"
                    "16. For aggregations ALWAYS use:\n"
                    "    - COUNT(*) FILTER (WHERE condition) for conditional counting\n"
                    "    - SUM(CASE WHEN condition THEN 1 ELSE 0 END) for counting matches\n"
                    "17. Never use:\n"
                    "    - Raw tables directly (always go through base_data)\n"
                    "    - Raw topic or sentiment columns (always use clean_topic and clean_sentiment)\n"
                    "    - Calculations without CAST and ROUND\n"
                    "    - Division without NULLIF\n"
                    "    - Date comparisons without INTERVAL\n"
                    "18. ALWAYS include proper ordering:\n"
                    "    ORDER BY [columns] {ASC|DESC} NULLS LAST\n"
                    "19. For limiting results:\n"
                    "    LIMIT [number]\n"
                    "20. Make sure to answer the question using the same language used by the user to ask it.\n"
                    "21. CRITICAL REMINDERS:\n"
                    "    - EVERY query MUST start with WITH base_data AS (...)\n"
                    "    - NEVER try to reference base_data without defining it first\n"
                    "    - ALWAYS include the complete base_data CTE definition\n"
                    "    - COPY and PASTE the exact base_data CTE structure shown above\n"
                    "22. NEVER execute a DELETE, UPDATE, INSERT, DROP, or any other data modification statements.\n"
                    f"\nSchema of available tables:\n{available_tables}\n\n")

    enhanced_query = user_query

    # Add context object awareness
    if context_obj and context_obj.get_current_topic():
        context_obj.update_focus(user_query)
        current_topic = context_obj.get_current_topic()

        if any(phrase in user_query.lower() for phrase in ['first issue', 'the issue', 'that issue']):
            enhanced_query = user_query.replace(
                'first issue', f'"{current_topic}"'
            ).replace(
                'the issue', f'"{current_topic}"'
            ).replace(
                'that issue', f'"{current_topic}"'
            )

    # Add text context if available
    if text_context:
        base_context += "Previous Conversation Context:\n"
        base_context += f"{text_context}\n\n"
        base_context += "Please consider the above conversation context when answering the following question:\n"

    return f"{base_context} Please only use these tables in your SQL query and do not make up non-existing ones to answer the following question: {enhanced_query}"


def generate_db_response(question: str, text_context: Optional[str] = None, context_obj: Optional[ConversationContext] = None) -> str:
    """Generate new response using OpenAI with context length handling and retry logic"""
    def try_generate_response(attempt: int = 1) -> tuple[Optional[str], Optional[str], int]:
        start_time = time.time()

        # Get total word count and cleaned words
        words = question.lower().split()
        total_words = len(words)
        logger.info(f"Total words in question: {total_words}")

        # Count English words
        english_words = ['what', 'which', 'how', 'where', 'when', 'who', 'why', 'did', 'does', 'has', 'had', 'it', 'there', 'they', 'show',
                         'is', 'are', 'was', 'were', 'the', 'this', 'that', 'these', 'those', 'our', 'your', 'an', 'a', 'top', 'give', 'do']
        nb_en = sum(1 for word in words if word in english_words)
        logger.info(f"English words found: {
                    [word for word in words if word in english_words]}")

        # Count Dutch words
        dutch_words = ['wat', 'hoe', 'waarom', 'welke', 'kunnen', 'waar', 'wie', 'wanneer', 'onderwerp',
                       'kun', 'kunt', 'je', 'jij', 'u', 'bent', 'zijn', 'waar', 'wat', 'wie', 'hoe',
                       'waarom', 'wanneer', 'welk', 'welke', 'het', 'de', 'een', 'het', 'deze', 'dit',
                       'die', 'dat', 'mijn', 'uw', 'jullie', 'ons', 'onze', 'geen', 'niet', 'met',
                       'over', 'door', 'om', 'op', 'voor', 'na', 'bij', 'aan', 'in', 'uit', 'te',
                       'bedrijf', 'waarom', 'tevreden', 'graag', 'gaan', 'wordt', 'komen', 'zal']
        nb_dutch = sum(1 for word in words if word in dutch_words)
        logger.info(f"Dutch words found: {
                    [word for word in words if word in dutch_words]}")

        logger.info(f"number of Dutch words: {
                    nb_dutch}, number of English words: {nb_en}")

        # Set language based on majority
        is_dutch = nb_dutch > nb_en
        is_english = nb_en > nb_dutch
        logger.info(f"is_dutch: {is_dutch}")

        validator = DatabaseAgentValidator()
        db = get_db_connection()

        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=os.getenv('AI_ANALYZER_OPENAI_API_KEY'),
            model_name='gpt-3.5-turbo'
        )

        db_chain = SQLDatabaseSequentialChain.from_llm(
            llm,
            db,
            verbose=True,
            use_query_checker=True
        )

        processed_query = add_context_to_query(
            question, text_context, context_obj)

        # Add language instruction only for Dutch questions
        if is_dutch:
            language_instruction = """
            BELANGRIJK: Je bent een Nederlandstalige database-assistent. 
            Geef alle antwoorden volledig in het Nederlands, inclusief:
            - Nederlandse datum- en tijdnotatie (bijvoorbeeld: 24 december 2024 14:30)
            - Nederlandse getallen (gebruik punt voor duizendtallen, komma voor decimalen)
            - Nederlandse bewoordingen voor alle termen (bijvoorbeeld: 'Transcriptie ID' in plaats van 'Transcript ID')
            """
            processed_query = language_instruction + "\n" + processed_query

        sql_result = db_chain.run(processed_query)

        if not sql_result or sql_result.strip() == "":
            return None, "Empty result", start_time

        # Create and log system message for final formatting
        system_message = {
            "role": "system",
            "content": "Je bent een Nederlandse database-assistent. Geef alle antwoorden in correct Nederlands." if is_dutch else "You are a database assistant. Provide answers in English."
        }
        user_message = {
            "role": "user",
            "content": f"{'Herformuleer deze analyse in duidelijk Nederlands:' if is_dutch else 'Format this analysis:'}\n\n{sql_result}"
        }
        logger.info(f"Formatting with system message: {system_message}")
        logger.info(f"User message for formatting: {user_message}")

        final_result = llm.predict_messages(
            [system_message, user_message]).content
        response_time = int((time.time() - start_time) * 1000)
        return final_result, None, response_time

    # First attempt
    result, error, response_time = try_generate_response(1)

    if result:
        cache_manager.track_query_performance(
            query=question,
            was_answered=True,
            response_time=response_time,
            topic_category=detect_query_topic(question),
            tokens_used=estimate_tokens_used(result)
        )
        return result

    logger.warning(f"First query attempt failed: {error}. Retrying...")

    # Second attempt
    result, error, response_time = try_generate_response(2)

    if result:
        cache_manager.track_query_performance(
            query=question,
            was_answered=True,
            response_time=response_time,
            topic_category=detect_query_topic(question),
            tokens_used=estimate_tokens_used(result)
        )
        return result

    # Get words for Dutch detection in error messages
    words = question.lower().split()
    dutch_words = ['kun', 'kunt', 'je', 'jij', 'u', 'bent', 'zijn', 'waar', 'wat', 'wie', 'hoe',
                   'waarom', 'wanneer', 'welk', 'welke', 'het', 'de', 'een', 'het', 'deze', 'dit',
                   'die', 'dat', 'mijn', 'uw', 'jullie', 'ons', 'onze', 'geen', 'niet']
    is_dutch = sum(1 for word in words if word in dutch_words) > 0

    if error == "context_length_error":
        error_message = "Context length exceeded - Too much data requested"
        user_message = (
            "De opgevraagde tijdsperiode bevat te veel data om te verwerken. Probeer:\n"
            "1. De tijdsperiode te verkleinen (bijvoorbeeld laatste week i.p.v. laatste maand)\n"
            "2. De zoekcriteria te verfijnen\n"
            "3. De vraag op te splitsen in kleinere delen\n"
            "\nBijvoorbeeld, als u naar een jaar aan data vroeg, probeer dan een maand of kwartaal."
        ) if is_dutch else (
            "The requested time range contains too much data to process. Please try:\n"
            "1. Reducing the time range (e.g., last week instead of last month)\n"
            "2. Narrowing down your search criteria\n"
            "3. Breaking your question into smaller parts\n"
            "\nFor example, if you asked about a year of data, try asking about a month or a quarter instead."
        )
    else:
        error_message = "Invalid SQL query"
        user_message = "Geen geldige SQL-query gevonden" if is_dutch else "No valid SQL query found in the response"

    cache_manager.track_query_performance(
        query=question,
        was_answered=False,
        response_time=response_time,
        error_message=error_message,
        topic_category=detect_query_topic(question)
    )

    return user_message


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


def answer_question(question: str, conversation_id: Optional[str] = None, db_session=None):
    """Enhanced question answering with better context awareness"""
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

        # Get enhanced conversation context
        text_context, context_obj = None, None
        if conversation_id and db_session:
            text_context, context_obj = get_conversation_context(
                db_session, conversation_id)
            logger.info(f"Retrieved conversation context for ID {
                        conversation_id}")

        # Generate new response with enhanced context
        response = generate_db_response(question, text_context, context_obj)

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
