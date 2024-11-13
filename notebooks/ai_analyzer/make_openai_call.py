import os
import openai
from openai import OpenAI
import json
from jinja2 import Template
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Initialize OpenAI client
client = OpenAI(
  api_key=os.environ['AI_ANALYZER_OPENAI_API_KEY'],  
)

def generate_prompt(data, prompt_template_path):

    """Generates the prompt using a template."""
    if not os.path.exists(prompt_template_path):
        raise FileNotFoundError(
            f"Prompt template not found at: {prompt_template_path}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Directory contents: {os.listdir(os.path.dirname(prompt_template_path))}"
        )

    with open(prompt_template_path, 'r') as file:
        template_string = file.read()

    template = Template(template_string)
    prompt = template.render(data)
    return prompt

def access_sentiment_topic(response, model_type):
    """Extract sentiment and topic from the API response."""
    if model_type == "chat":
        output_text = response.choices[0].message.content.strip()
    else:
        output_text = response.choices[0].text.strip()

    # Remove any backticks and code block formatting
    output_text = output_text.replace("```json", "").replace("```", "").strip()

    # Ensure the response is not empty
    if not output_text:
        raise ValueError("The API response is empty or invalid.")
    
    # Parse the JSON string into a Python dictionary
    try:
        json_output = json.loads(output_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e} \nResponse Text: {output_text}")
    
    # Extract the sentiment and topic from the dictionary
    sentiment = json_output.get("sentiment")
    topic = json_output.get("topic")
    summary = json_output.get("summary")

    result = {
        "sentiment": sentiment,
        "topic": topic,
        "summary": summary,
    }
    
    return result

# Define pricing per model (per 1K tokens)
pricing = {
    "gpt-4o-2024-08-06": {"prompt": 0.003750, "completion": 0.015000},  # GPT-4o with new pricing
    "gpt-4o-mini-2024-07-18": {"prompt": 0.000330, "completion": 0.001200},  # GPT-4o-mini with new pricing
    "gpt-3.5-turbo": {"prompt": 0.003000, "completion": 0.006000},  # GPT-3.5 pricing (chat model)
    "gpt-3.5-turbo-instruct": {"prompt": 0.001500, "completion": 0.002000},  #  GPT-3.5-turbo-instruct pricing
    "davinci-002": {"prompt": 0.012000, "completion": 0.012000},  # DaVinci pricing
    "babbage-002": {"prompt": 0.001600, "completion": 0.001600},  # Babbage pricing
}
# Helper function to determine model type (chat or completion)
def is_chat_model(model):
    chat_models = ["gpt-3.5-turbo", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]
    return model in chat_models

def calculate_cost(model, prompt_tokens, completion_tokens):
    """Calculates the cost based on prompt and completion tokens."""
    prompt_cost = (prompt_tokens / 1000) * pricing[model]["prompt"]
    completion_cost = (completion_tokens / 1000) * pricing[model]["completion"]
    total_cost = prompt_cost + completion_cost
    return total_cost

def make_openai_call(prompt, model):
    """Make an OpenAI API call and calculate its cost."""
    model_type = "chat" if is_chat_model(model) else "completion"
    
    if model_type == "chat":
        # Use ChatCompletion for chat models
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that does summarization, sentiment analysis, and topic modeling."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0
        )
    else:
        # Use Completion for completion models
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=500,
            temperature=0
        )

    # Extract token usage from the response
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    # Calculate the cost for this call
    cost = calculate_cost(model, prompt_tokens, completion_tokens)
    cost = round(cost, 6)

    # Extract sentiment and topic from the response
    sentiment_topic = access_sentiment_topic(response, model_type)
    sentiment = sentiment_topic["sentiment"]
    topic = sentiment_topic["topic"]
    summary = sentiment_topic["summary"]

    # Print the result and cost
    # print(f"model: {model}")
    # print(f"sentiment: {sentiment}")
    # print(f"topic: {topic}")
    # print(f"API Call Cost: ${cost:.5f}")
    # print(f"Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}")
    result = {
        "summary": summary,
        "topic": topic,
        "sentiment": sentiment,
        "cost": cost,
    }
    
    return result

