import time
from ai_analyzer_old.make_openai_call import generate_prompt, make_openai_call
import pandas as pd
from datetime import datetime
import os


# Get the current date
current_date = datetime.now().date().strftime("%Y-%m-%d")

# Get the absolute path to the prompt template - adjusted for the correct directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPT_TEMPLATE_PATH = os.path.join(BASE_DIR, 'ai-analyzer', 'prompt_template', 'prompt_template_sentiment_topic.txt')
DATA_DIR = os.path.join(BASE_DIR, 'data')

def debug_paths():
    """Print debug information about paths"""
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"PROMPT_TEMPLATE_PATH: {PROMPT_TEMPLATE_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"File exists at PROMPT_TEMPLATE_PATH: {os.path.exists(PROMPT_TEMPLATE_PATH)}")
    print(f"Directory contents of {os.path.dirname(PROMPT_TEMPLATE_PATH)}: {os.listdir(os.path.dirname(PROMPT_TEMPLATE_PATH))}")

def make_openai_call_df(df, model="gpt-4o-mini-2024-07-18", n=None):
    # Print debug information
    debug_paths()
    
    if not os.path.exists(PROMPT_TEMPLATE_PATH):
        raise FileNotFoundError(
            f"Prompt template not found at: {PROMPT_TEMPLATE_PATH}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"BASE_DIR contents: {os.listdir(BASE_DIR)}"
        )

    if not n:
        n = len(df)

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Start the timer
    start_time = time.time()
        
    df_n = df.iloc[:n].copy().rename(columns={'summary': 'summary_old'})
    result_df = pd.DataFrame(columns=['summary', 'topic', 'sentiment', 'cost'])
    
    for i in range(len(df_n)):
        if i % 500 == 0:
            print(f'{i} rows from {len(df_n)} processed')
        data = {
            'transcript': df_n['transcription'][i],
        }
        prompt = generate_prompt(
            data=data,
            prompt_template_path=PROMPT_TEMPLATE_PATH
        )
        kwargs_2 = {
            "model": model,
            "prompt": prompt
        }

        result = make_openai_call(**kwargs_2)
        result_df_temp = pd.DataFrame([result])
        result_df = pd.concat([result_df, result_df_temp], axis=0)
    
    result_df = result_df.reset_index(drop=True)
    result_df = pd.concat([df_n, result_df], axis=1)
    
    # Calculate and print the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Total Elapsed time to process {n} rows: {elapsed_time} seconds")
    
    result_df = (result_df
                 .filter(items=[
                     'id',
                     'processingdate',
                     'transcription',
                     'summary',
                     'topic',
                     'sentiment'
                 ])
                 .rename(columns={'id': 'company_id'})
                )

    output_path = os.path.join(DATA_DIR, f'df_transcription__{current_date}.csv')
    result_df.to_csv(output_path, index=False)

    return result_df