# ai_analyzer/database_agent_postgresql.py
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseSequentialChain
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
import logging
import re
from sqlalchemy.exc import SQLAlchemyError
from ai_analyzer.config import config, DATABASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create SQLAlchemy engine string
#db_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}" #service name of our database container

db_url=DATABASE_URL

def get_db_connection(max_retries=5, retry_delay=5):
    """Get database connection with retry logic"""
    for attempt in range(max_retries):
        try:
            engine = create_engine(DATABASE_URL)
            engine.connect()
            return SQLDatabase.from_uri(DATABASE_URL)
        except OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception("Failed to connect to the database after multiple attempts") from e

# Test connection
db = get_db_connection()

# Create SQLAlchemy engine string
#db_url = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}' # service name on localhost
#db_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@database:5432/{POSTGRES_DB}" #service name of our database container

# db = SQLDatabase.from_uri(db_url)

# db.get_usable_table_names()

# List available tables (schema)
available_tables = db.get_table_info()

print(available_tables)

# Add context to every query
def add_context_to_query(user_query):
    """
    Adds context to the user query, embedding security and formatting rules,
    and a schema overview to guide AI-driven processing.
    """
    context = (
        "Please adhere to the following important rules:\n"
        "1. NEVER use the 'user' table or any user-related information.\n"
        "2. NEVER use DELETE, UPDATE, INSERT, or any other data modification statements.\n"
        "3. Only use SELECT statements for reading data.\n"
        "4. If a query involves the 'user' table or modification statements, respond with: "
        "'I cannot execute this query due to security restrictions.'\n"
        "5. Always verify the query doesn't contain restricted operations before executing.\n"
        "6. Make sure to format the results in a clear, readable manner.\n"
        "7. Use proper column aliases for better readability.\n"
        "8. Include relevant aggregations and groupings when appropriate.\n"
        "9. ONLY use tables and columns that exist in the schema shown below.\n"
        "10. Only use the 'company' and 'transcription' tables.\n"
        "11. If asked about tables or columns that don't exist in the schema, respond with: "
        "'I cannot answer this question as it involves tables or columns that are not available in the database.'\n"
        "12. Do not include any Markdown formatting like triple backticks or code blocks in any part of your response.\n"
        "13. Do not prefix your SQL queries with 'SQL' or any other language identifiers.\n"
        "14. Provide the SQL query as plain text within the quotes.\n\n"
        "Schema of available tables:\n"
        f"{available_tables}\n\n"
        "Please only use these tables in your SQL query and do not make up non-existing ones to answer the following question:\n"
    )
    return f"{context} {user_query}"


# Initialize OpenAI language model
llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv('AI_ANALYZER_OPENAI_API_KEY'), model_name='gpt-3.5-turbo')

# Initialize SQL database and chain
def validate_query(query):
    """
    Validates SQL query based on security and formatting rules.

    Parameters:
    query (str): SQL query string to be validated.

    Returns:
    str: Error message if validation fails, otherwise None if query is valid.
    """
    forbidden_tables = ['user', 'users', 'user_memory']
    forbidden_operations = ['DELETE', 'UPDATE', 'INSERT', 'DROP', 'ALTER']
    allowed_tables = ['company', 'transcription']

    # Find the position of the first 'SELECT' to isolate the SQL part
    select_index = query.upper().find("SELECT")
    if select_index == -1:
        return "Only SELECT statements are allowed."

    # Extract the SQL statement starting from the SELECT keyword
    sql_statement = query[select_index:]

    # Check for forbidden tables
    for table in forbidden_tables:
        if re.search(rf'\b{table}\b', sql_statement, re.IGNORECASE):
            return "I cannot execute this query due to security restrictions."

    # Check for forbidden operations (whole words only)
    for operation in forbidden_operations:
        if re.search(rf'\b{operation}\b', sql_statement, re.IGNORECASE):
            return "I cannot execute this query due to security restrictions."

    # Ensure only allowed tables are used (matches table names after FROM or JOIN)
    tables_in_query = re.findall(r'\b(?:FROM|JOIN)\s+([A-Za-z_]+)', sql_statement, re.IGNORECASE)
    if any(table for table in tables_in_query if table not in allowed_tables):
        return "I cannot answer this question as it involves tables or columns that are not available in the database."

    # Query passed validation
    return None


def answer_question(question):
    try:
        db = SQLDatabase.from_uri(db_url)
        db_chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True, use_query_checker=True)

        logger.info(f"Received question: {question}")

        # Process the query by adding the context with available tables
        processed_query = add_context_to_query(question)
        
        # Validate the query before execution
        validation_error = validate_query(processed_query)
        print(validation_error )
        #if validation_error:
         #   return validation_error

        # Execute the question using the chain
        result = db_chain.run(processed_query)
        return result
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"An error occurred while processing your question: {e}"

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
