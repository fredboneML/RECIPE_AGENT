# ai_analyzer/make_openai_call.py
import os
import openai
from openai import OpenAI
import json
from jinja2 import Template
from ai_analyzer.config import config


# Initialize OpenAI client
client = OpenAI(
  api_key=config['AI_ANALYZER_OPENAI_API_KEY'],  
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

    # Check if required fields are present
    if not all([sentiment, topic, summary]):
        raise ValueError("Missing required fields in JSON response")

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


def is_chat_model(model):
    """Helper function to determine model type (chat or completion)"""
    chat_models = ["gpt-3.5-turbo", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]
    return model in chat_models

def calculate_cost(model, prompt_tokens, completion_tokens):
    """Calculates the cost based on prompt and completion tokens."""
    prompt_cost = (prompt_tokens / 1000) * pricing[model]["prompt"]
    completion_cost = (completion_tokens / 1000) * pricing[model]["completion"]
    return round(prompt_cost + completion_cost, 6)

def make_api_call(prompt, model, attempt=1, max_attempts=5):
    """Make a single API call with the given model and handle the response"""
    model_type = "chat" if is_chat_model(model) else "completion"
    
    try:
        if model_type == "chat":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that does summarization, sentiment analysis, and topic modeling. You MUST respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0
            )
        else:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=500,
                temperature=0
            )

        # Try to parse the response
        sentiment_topic = access_sentiment_topic(response, model_type)
        
        # Calculate costs
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cost = calculate_cost(model, prompt_tokens, completion_tokens)

        return {
            "success": True,
            "data": sentiment_topic,
            "cost": cost,
            "attempt": attempt
        }

    except (ValueError, json.JSONDecodeError) as e:
        print(f"Attempt {attempt} failed: {str(e)}")  # Add logging for debugging
        if attempt < max_attempts:
            # Add a small delay before retrying
            time.sleep(5)
            # Modify the prompt to emphasize JSON format requirement
            enhanced_prompt = f"{prompt}\n\nIMPORTANT: Your response MUST be in valid JSON format as shown in the example above. Do not include any other text or formatting."
            return make_api_call(enhanced_prompt, model, attempt + 1, max_attempts)
        else:
            return {
                "success": False,
                "error": str(e),
                "attempt": attempt
            }
    except Exception as e:
        print(f"Unexpected error on attempt {attempt}: {str(e)}")  # Add logging for debugging
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "attempt": attempt
        }

def make_openai_call(prompt, model):
    """Make OpenAI API call with retry logic and return the results"""
    result = make_api_call(prompt, model)
    
    if result["success"]:
        return {
            "summary": result["data"]["summary"],
            "topic": result["data"]["topic"],
            "sentiment": result["data"]["sentiment"],
            "cost": result["cost"],
            "attempts": result["attempt"]
        }
    else:
        raise ValueError(f"Failed to get valid response after multiple attempts. Last error: {result['error']}")
