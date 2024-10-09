import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseSequentialChain, SQLDatabaseChain
#from langchain_community.llms import SQLDatabase
from langchain.chat_models import ChatOpenAI
#from urllib.parse import quote_plus
#from langchain import SQLDatabase, SQLDatabaseChain
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(find_dotenv())

# PostgreSQL connection details
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')  
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Create SQLAlchemy engine string
db_url = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Initialize OpenAI language model
llm = ChatOpenAI(temperature=0, openai_api_key=os.environ['AI_ANALYZER_OPENAI_API_KEY'], model_name='gpt-3.5-turbo')

# Initialize SQLDatabase
db = SQLDatabase.from_uri(db_url)

# Initialize SQLDatabaseChain
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

def answer_question(question):
    logger.info(f"Received question: {question}")

    try:
        # Use SQLDatabaseChain to generate and execute SQL query
        result = db_chain.run(question)
        
        logger.info("Query executed successfully.")
        print("\nAnswer:")
        print(result)
        
        return result
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return f"An error occurred while processing your question: {str(e)}"

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
    question = "What are the top 10 topics discussed in all transcriptions?"
    answer_question(question)

    # Alternatively, iterate through all sample questions
    # for q in sample_questions:
    #     answer_question(q)
    #     print("\n" + "-"*50 + "\n")
