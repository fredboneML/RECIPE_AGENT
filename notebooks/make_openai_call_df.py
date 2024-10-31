import time
from make_openai_call import generate_prompt, make_openai_call
import pandas as pd
from datetime import datetime


# Get the current date
current_date = datetime.now().date().strftime("%Y-%m-%d")
current_date

def make_openai_call_df(df, model="gpt-4o-mini-2024-07-18", n=None):
    if not n:
        n = len(df)

    # Start the timer
    start_time = time.time()
        
    df_n = df.iloc[:n].copy().rename(columns={'summary': 'summary_old'})
    result_df = pd.DataFrame(columns = ['summary', 'topic', 'sentiment', 'cost'])
    for i in range(len(df_n)):
        if i % 250 == 0:
            print(f'{i} rows from {len(df_n)} processed')
        data = {
                'transcript': df_n['transcription'][i],
                }
        prompt = generate_prompt(data=data, 
                                 prompt_template_path='../prompt_template/prompt_template_sentiment_topic.txt'
                                )
        kwargs_2 ={
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
    print("Total Elapsed time to process {n} rows:", elapsed_time, "seconds")
    
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

    result_df.to_csv(f'../data/df_transcription__{current_date}.csv', index=False)

    return result_df
