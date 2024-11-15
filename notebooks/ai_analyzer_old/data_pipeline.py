#!/usr/bin/env python3

import sys
import os

# Add the parent directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from datetime import datetime 
from dotenv import load_dotenv, find_dotenv
from ai_analyzer_old import config
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Integer
from ai_analyzer_old import fetch_data_from_api as fetch_data
from ai_analyzer_old.data_import_postgresql import run_data_import
import logging

import os
import pandas as pd
from ai_analyzer_old.make_openai_call_df import make_openai_call_df

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('data_import_postgresql')

# Load environment variables from .env file
load_dotenv(find_dotenv())

# PostgreSQL connection details
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('DB_HOST')  
DB_PORT = os.getenv('DB_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')

# API URL details
url = os.getenv('URL')
api_key = os.getenv('API_KEY')
limit = config.LIMIT

# Get the current date
current_date = datetime.now().date().strftime("%Y-%m-%d")

data_dir = config.DATA_DIR

# Base for ORM models
Base = declarative_base()

# Transcription table ORM model
class Transcription(Base):
    __tablename__ = 'transcription'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False)
    company_id = Column(String, nullable=False)
    processingdate = Column(DateTime, nullable=False)
    transcription = Column(String, nullable=False)
    summary = Column(String)
    topic = Column(String)
    sentiment = Column(String)

# Create a database engine
def create_db_engine():
    engine = create_engine(f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}')
    engine.connect()
    logger.info("Successfully connected to the database.")
    return engine

# Return the max id of the transcription table
def get_max_company_id(session):
    """
    Get the maximum company_id from the transcription table.
    Casts the company_id to numeric before sorting to ensure proper numerical ordering.
    
    Args:
        session: SQLAlchemy session object
    
    Returns:
        str: The maximum company_id, or None if the table is empty
    """
    try:
        # Cast company_id to numeric for proper sorting
        from sqlalchemy import cast, Numeric
        max_id = session.query(Transcription.company_id)\
            .order_by(cast(Transcription.company_id, Numeric).desc())\
            .first()
        
        if max_id:
            logger.info(f"Retrieved maximum company_id: {max_id[0]}")
            return max_id[0]
        else:
            logger.info("No transcription records found in the database")
            return None
    except Exception as e:
        logger.error(f"Error retrieving maximum company_id: {e}")
        return None
engine = create_db_engine()
Session = sessionmaker(bind=engine)
session = Session()

def delete_files_by_date(directory, target_date):
    """
    Delete all files in the specified directory that have a specific date pattern between '__' and '.csv'
    
    Args:
        directory (str): The directory path where to look for files
        target_date (str): The date in format 'YYYY-MM-DD' to match
        
    Returns:
        list: Names of deleted files
    """
    try:
        deleted_files = []
        # Get all files in directory
        for filename in os.listdir(directory):
            try:
                # Find the date pattern between '__' and '.csv'
                if '__' in filename and '.csv' in filename:
                    date_part = filename.split('__')[1].split('.csv')[0]
                    
                    # Check if the extracted date matches the target date
                    if date_part == target_date:
                        file_path = os.path.join(directory, filename)
                        os.remove(file_path)
                        deleted_files.append(filename)
                        logger.info(f"Deleted file: {filename}")
            
            except (IndexError, Exception) as e:
                logger.error(f"Error processing file {filename}: {e}")
                continue
                
        if not deleted_files:
            logger.info(f"No files found with date {target_date}")
        else:
            logger.info(f"Successfully deleted {len(deleted_files)} files")
            
        return deleted_files
        
    except Exception as e:
        logger.error(f"Error while deleting files: {e}")
        return []

#########################################################################################
#--------------------------------------- Pipeline ---------------------------------------
#########################################################################################

## 1rst Step of the pipeline ##
# Querying the max id of the transcription table
last_id = int(get_max_company_id(session=session))


## 2nd Step of the pipeline ##
# Fetching data using the last_id found in the database to make sure 
# We are only fetching new data

fetch_data.fetch_data_from_api(url, api_key, last_id, limit)

## 3rd Step of the pipeline ##
# Generating sentiment and topic of the new data that has been fetched

df__file_name = [file for file in os.listdir(data_dir) if file.startswith('df__')]
df__file_name = sorted(df__file_name,
                       key=lambda x: datetime.strptime(x.split('__')[1].split('.csv')[0], '%Y-%m-%d'),
                       reverse=True)
# The 3rd step should only be triggered if there are new data
if len(df__file_name) > 1:
    df__file_name = df__file_name[0]
    df = pd.read_csv(f'{data_dir}/{df__file_name}')
    make_openai_call_df(df=df, model="gpt-4o-mini-2024-07-18")

## 4th Step of the pipeline ##
    # Insert the generated topics and sentiments to the db
    run_data_import()
    
## 5th Step of the pipeline ##
    # Clean up (removing data inside the data folder after successful data import)
    delete_files_by_date(data_dir, current_date)

#########################################################################################
#--------------------------------------- End Pipeline -----------------------------------
#########################################################################################
