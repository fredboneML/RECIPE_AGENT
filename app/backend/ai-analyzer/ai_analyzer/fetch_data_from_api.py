# ai_analyzer/fetch_data_from_api.py
import requests
import pandas as pd
from datetime import datetime
import logging
import json
from ai_analyzer.config import config, DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fetch_data_from_api')

# Get the current date
current_date = datetime.now().date().strftime("%Y-%m-%d")

def fetch_data_from_api(url, api_key, last_id, limit=None):
    try:
        if limit is None:
            limit = config['LIMIT']
            
        params = {
            'last_id': last_id,
            'limit': limit
        }

        form_data = {
            'api_key': api_key,
        }

        logger.info(f"Sending request to {url} with last_id: {last_id}")
        logger.info(f"URL: {config['URL']}")
        logger.info(f"API_KEY: {config['API_KEY']}")
        logger.info(f"LIMIT: {config['LIMIT']} ")
        response = requests.post(url, data=form_data, params=params)

        if response.status_code == 200:
            data = response.json()
            logger.info(f"Response received. Status: {response.status_code}")
            logger.info(f"Response content type: {type(data)}")
            logger.info(f"Response structure: {json.dumps(data)[:200]}...")
            
            if not data:
                logger.info('No new data to be fetched')
                return
                
            try:
                df = pd.DataFrame.from_records(data if isinstance(data, list) else [data])
                logger.info(f"DataFrame columns: {df.columns.tolist()}")
                
                required_company_cols = ['id', 'clid', 'dst']
                if all(col in df.columns for col in required_company_cols):
                    df_company = df[required_company_cols].rename(columns={'dst': 'telephone_number'})
                    df_company.to_csv(f'{DATA_DIR}/df_company__{current_date}.csv', index=False)
                    logger.info(f'Company data saved with {len(df_company)} records')
                else:
                    logger.warning(f"Missing company columns. Available columns: {df.columns.tolist()}")
                
                if 'processtime' in df.columns:
                    df = df.rename(columns={'processtime': 'processingdate'})
                
                required_main_cols = ['id', 'processingdate', 'transcription']
                if all(col in df.columns for col in required_main_cols):
                    df_main = df[required_main_cols + ['summary'] if 'summary' in df.columns else required_main_cols]
                    df_main.to_csv(f'{DATA_DIR}/df__{current_date}.csv', index=False)
                    logger.info(f'Main data saved with {len(df_main)} records')
                else:
                    logger.warning(f"Missing main columns. Available columns: {df.columns.tolist()}")
                
                if 'id' in df.columns and len(df) > 0:
                    logger.info(f'Data processing complete. First id: {df["id"].min()}, last id: {df["id"].max()}')
                
            except Exception as e:
                logger.error(f"Error processing DataFrame: {str(e)}", exc_info=True)
                logger.error(f"Raw data sample: {str(data)[:500]}")
                raise
                
        else:
            logger.error(f"Failed to fetch data. Status code: {response.status_code}")
            logger.error(f"Response content: {response.text[:500]}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        fetch_data_from_api(config['URL'], config['API_KEY'], config.LAST_ID, config['LIMIT'])
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)