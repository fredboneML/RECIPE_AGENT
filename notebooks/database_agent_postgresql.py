import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseSequentialChain
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(find_dotenv())

# PostgreSQL connection details
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('DB_HOST')  
DB_PORT = os.getenv('DB_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')

# Create SQLAlchemy engine string
db_url = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}'

db = SQLDatabase.from_uri(db_url)

# List available tables (schema)
available_tables = db.get_table_info()

print(available_tables)

# Add context to every query
def add_context_to_query(user_query):
    context = f"Here is the schema of the available tables: {available_tables}. \
    Make sure to give the result in an easy readable human format! \
    Please only use these tables in your SQL query to answer the following question: "
    return f"{context} {user_query}"


# Initialize OpenAI language model
llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv('AI_ANALYZER_OPENAI_API_KEY'), model_name='gpt-3.5-turbo')

# Initialize SQL database and chain
def answer_question(question):
    try:
        # Create SQL database and chain using the new method
        db = SQLDatabase.from_uri(db_url)
        db_chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True, use_query_checker=True)

        logger.info(f"Received question: {question}")

        # Process the query by adding the context with available tables
        processed_query = add_context_to_query(question)

        
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
