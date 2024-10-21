import requests
import pandas as pd
from datetime import datetime


# Get the current date
current_date = datetime.now().date().strftime("%Y-%m-%d")
current_date


def fetch_data_from_api(url, api_key, last_id, limit=10000):
    # Define the URL parameters (query parameters)
    params = {
        'last_id': last_id,
        'limit':limit
    }

    # Define the form data to send in the POST request
    form_data = {
        'api_key': api_key,

    }

    # Send the POST request
    response = requests.post(url, data=form_data, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Print the response data
        print('data successfully fetched')  
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}, Response: {response.text}")

    if len(response.json()) > 0:
        
        # Convert the response data into a pandas DataFrame
        df = pd.DataFrame(response.json()).rename(columns={'processtime': 'processingdate'})
        df_company = df[['id', 'clid', 'dst']].rename(columns={'dst': 'telephone_number'})
        df = df[['id', 'processingdate', 'transcription', 'summary']]
        
        df_company.to_csv(f'../data/df_company__{current_date}.csv', index=False)
        df.to_csv(f'../data/df__{current_date}.csv', index=False)
        print('data successfully saved') 
        print(f"""first id: {min(df['id'])}, last id: {max(df['id'])}""")
    else:
        print('No new data to be fetched')
