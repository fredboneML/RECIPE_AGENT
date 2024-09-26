import openai

client = OpenAI(
  api_key=os.environ['AI_ANALYZER_OPENAI_API_KEY'],  
)

def generate_prompt(data, prompt_template_path):
    with open(prompt_template_path, 'r') as file:
        template_string = file.read()

    template = Template(template_string)
    prompt = template.render(data)
    return prompt

def access_sentiment_topic(response):
    # Access the 'choices' attribute and extract the JSON text from the first choice
    output_text = response.choices[0].text.strip()
    
    # Parse the JSON string into a Python dictionary
    json_output = json.loads(output_text)
    
    # Extract the sentiment and topic from the dictionary
    sentiment = json_output.get("sentiment")
    topic = json_output.get("topic")

    result = {
        "sentiment": sentiment,
        "topic": topic,
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
    
def calculate_cost(model, prompt_tokens, completion_tokens):
    """Calculates the cost based on prompt and completion tokens."""
    prompt_cost = (prompt_tokens / 1000) * pricing[model]["prompt"]
    completion_cost = (completion_tokens / 1000) * pricing[model]["completion"]
    total_cost = prompt_cost + completion_cost
    return total_cost

def make_openai_call(prompt, model):
    """Make an OpenAI API call and calculate its cost."""
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=50,
        temperature=0
    )

    # Extract token usage from the response
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    # Calculate the cost for this call
    cost = calculate_cost(model, prompt_tokens, completion_tokens)

    # Extract sentiment and topic from the response
    sentiment_topic = access_sentiment_topic(response)
    sentiment = sentiment_topic["sentiment"]
    topic = sentiment_topic["topic"]

    # Print the result and cost
    print(f"model: {model}")
    print(f"sentiment: {sentiment}")
    print(f"topic: {topic}")
    print(f"API Call Cost: ${cost:.5f}")
    print(f"Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}")
