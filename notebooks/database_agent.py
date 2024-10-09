# database_agent.py
import os
from openai import OpenAI
import weaviate
from collections import Counter
from dotenv import load_dotenv, find_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

client_openai = OpenAI(
  api_key=os.environ['AI_ANALYZER_OPENAI_API_KEY'],  
)

# Connect to Weaviate instance
client = weaviate.Client(
    url="http://localhost:8090"
)

# Define the schema description
schema_description = """
Classes:
- Company:
    - company_id (string)
    - clid (string)
    - telephone_number (string)
- Transcription:
    - company_id (string)
    - processingdate (date)
    - transcription (text)
    - summary (text)
    - topic_parent_class (string)
    - topic (string)
    - sentiment_parent_class (string)
    - sentiment (string)
    - ofCompany (reference to Company)
"""

def generate_prompt(question):
    escaped_examples = """
Examples:
Question: "What are the most frequently discussed topics in calls with Company Y?"
GraphQL Query:
{
  Aggregate {
    Transcription(
      where: {
        path: ["ofCompany", "Company", "clid"],
        operator: Equal,
        valueString: "Company Y"
      }
    ) {
      groupedBy {
        path: ["topic"]
        value
      }
      topic {
        count
      }
    }
  }
}

Question: "Give me the top 10 topics?"
GraphQL Query:
{
  Aggregate {
    Transcription(limit: 10) {
      groupedBy {
        path: ["topic"]
        value
      }
      topic {
        count
      }
    }
  }
}
"""
    prompt = f"""
You are an expert data analyst who translates natural language questions into Weaviate GraphQL queries.

Schema:
{schema_description}

Guidelines:
- Analyze the question carefully to understand what data is being requested.
- Construct a valid Weaviate GraphQL query that can be executed against the database.
- Ensure the query aligns with the schema and uses the correct classes, properties, and data types.
- Use Weaviate's Aggregate API for aggregation queries.
- For top N queries, use the 'limit' directive at the Transcription level, not at the root level.
- Always include both 'groupedBy' and the specific field (e.g., 'topic') in the aggregation.
- In the 'groupedBy' clause, include both 'path' and 'value'.
- Do not include any explanations or additional text; only provide the GraphQL query.

{escaped_examples}

Question: "{question}"
GraphQL Query:
"""
    return prompt


def sanitize_query(query):
    """
    Clean up any extra characters and ensure proper structure.
    """
    # Remove backticks and clean up extra characters
    query = query.replace('`', '')
    
    # Remove any leading/trailing whitespace and newlines
    query = query.strip()
    
    # Ensure the query is wrapped in curly braces if not already
    if not query.startswith('{'):
        query = '{' + query
    if not query.endswith('}'):
        query = query + '}'
    
    return query
    

def get_graphql_query(question):
    prompt = generate_prompt(question)

    try:
        response = client_openai.chat.completions.create(
              model="gpt-4o",
              messages=[
                {"role": "system", "content": "You are an expert data analyst who translates natural language questions into Weaviate GraphQL queries."},
                {"role": "user", "content": prompt}
                ],
              max_tokens=500,
              temperature=0,
        )
        
        graphql_query = response.choices[0].message.content.strip()
        sanitized_query = sanitize_query(graphql_query)
        logger.info("GraphQL Query generated successfully.")
        return sanitized_query
    except Exception as e:
        logger.error(f"Error generating GraphQL query: {e}")
        return None

def execute_graphql_query(client, graphql_query):
    try:
        response = client.query.raw(graphql_query)
        logger.info("GraphQL Query executed successfully.")
        return response
    except Exception as e:
        logger.error(f"Error executing GraphQL query: {e}")
        return None

def generate_natural_language_answer(question, data):
    if not data or 'errors' in data:
        return "I'm sorry, I couldn't retrieve the data due to an error."

    data_str = str(data)

    prompt = f"""
You are an assistant that provides clear and concise answers to user questions based on the provided data.

Question: "{question}"

Data: {data_str}

Guidelines:
- Summarize the data to directly answer the question.
- If numerical data is provided, include relevant statistics or trends.
- Present the information in a human-readable format.
- Do not mention the data structure or formatting details.

Answer:
"""
    try:
        response = client_openai.chat.completions.create(
              model="gpt-4o",
              messages=[
                {"role": "system", "content": "You are an expert data analyst who translates natural language questions into Weaviate GraphQL queries."},
                {"role": "user", "content": prompt}
                ],
              max_tokens=500,
              temperature=0,
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info("Natural language answer generated successfully.")
        return answer
    except Exception as e:
        logger.error(f"Error generating natural language answer: {e}")
        return "I'm sorry, I couldn't process your request at this time."

def answer_question(question):
    logger.info(f"Received question: {question}")

    # Step 1: Get the GraphQL query from OpenAI
    graphql_query = get_graphql_query(question)
    if not graphql_query:
        logger.error("Failed to generate GraphQL query.")
        return

    print("\nGenerated GraphQL Query:")
    print(graphql_query)

    # Step 2: Execute the query against Weaviate
    data = execute_graphql_query(client, graphql_query)
    if data is None:
        logger.error("Failed to retrieve data.")
        return

    print("\nData Retrieved:")
    print(data)

    # Step 3: Generate a natural language answer
    answer = generate_natural_language_answer(question, data)
    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    # Sample questions to test the agent
    sample_questions = [
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
    question = "Give me the top 10 topics?"
    answer_question(question)

    # Alternatively, iterate through all sample questions
    # for q in sample_questions:
    #     answer_question(q)
    #     print("\n" + "-"*50 + "\n")
