
# data_import.py

import os
import uuid
import weaviate
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Step 1: Connect to Weaviate Instance
client = weaviate.Client(url="http://localhost:8090")

if client.is_ready():
    logger.info("Weaviate is ready!")
else:
    logger.error("Weaviate is not ready. Please check the server.")
    exit(1)

# Step 2: Define the schema for 'Company' and 'Transcription' classes

company_class = {
    "class": "Company",
    "description": "A company entity",
    "vectorizer": "none",  # No need to vectorize company data
    "properties": [
        {
            "name": "company_id",
            "dataType": ["string"],
            "description": "Unique identifier for the company"
        },
        {
            "name": "clid",
            "dataType": ["string"],
            "description": "Company name (clid)"
        },
        {
            "name": "telephone_number",
            "dataType": ["string"],
            "description": "Company's telephone number"
        }
    ]
}

transcription_class = {
    "class": "Transcription",
    "description": "A call transcription",
    "vectorizer": "text2vec-transformers",
    "moduleConfig": {
        "text2vec-transformers": {
            "poolingStrategy": "masked_mean",
            "vectorizeClassName": False
        }
    },
    "properties": [
        {
            "name": "company_id",
            "dataType": ["string"],
            "description": "Unique identifier for the company"
        },
        {
            "name": "processingdate",
            "dataType": ["date"],
            "description": "Processing date of the call"
        },
        {
            "name": "transcription",
            "dataType": ["text"],
            "description": "Transcription of the call"
        },
        {
            "name": "summary",
            "dataType": ["text"],
            "description": "Summary of the call"
        },
        {
            "name": "topic_parent_class",
            "dataType": ["string"],
            "description": "Parent class for topic"
        },
        {
            "name": "topic",
            "dataType": ["string"],
            "description": "Topic of the call"
        },
        {
            "name": "sentiment_parent_class",
            "dataType": ["string"],
            "description": "Parent class for sentiment"
        },
        {
            "name": "sentiment",
            "dataType": ["string"],
            "description": "Sentiment of the call"
        },
        {
            "name": "ofCompany",
            "dataType": ["Company"],
            "description": "Reference to the company"
        }
    ]
}

# Step 3: Delete existing classes if they exist (optional)
try:
    client.schema.delete_all()
    logger.info("Existing schema deleted successfully.")
except Exception as e:
    logger.warning(f"Could not delete existing schema: {e}")

# Step 4: Create the classes in Weaviate
try:
    client.schema.create_class(company_class)
    logger.info("Company class created successfully.")
except Exception as e:
    logger.error(f"Error creating Company class: {e}")

try:
    client.schema.create_class(transcription_class)
    logger.info("Transcription class created successfully.")
except Exception as e:
    logger.error(f"Error creating Transcription class: {e}")

# Step 5: Load data into DataFrames

try:
    df_transcription = pd.read_csv('../data/df_transcription.csv')
    logger.info("df_transcription.csv loaded successfully.")
except FileNotFoundError:
    logger.error("df_transcription.csv not found. Please check the file path.")
    exit(1)
except Exception as e:
    logger.error(f"Error loading df_transcription.csv: {e}")
    exit(1)

try:
    df_company = pd.read_csv('../data/df_company.csv')
    df_company = df_company[df_company['company_id'].isin(list(set(df_transcription['company_id'])))]

    logger.info("df_company.csv loaded successfully.")
except FileNotFoundError:
    logger.error("df_company.csv not found. Please check the file path.")
    exit(1)
except Exception as e:
    logger.error(f"Error loading df_company.csv: {e}")
    exit(1)

# Step 6: Ensure data types are correct

# Convert 'company_id' and 'telephone_number' to strings in df_company
df_company['company_id'] = df_company['company_id'].astype(str)
df_company['telephone_number'] = df_company['telephone_number'].astype(str)
df_company['clid'] = df_company['clid'].astype(str)  # Ensure 'clid' is string

# Handle missing values if any
df_company = df_company.fillna('')

# Convert 'company_id' to string in df_transcription
df_transcription['company_id'] = df_transcription['company_id'].astype(str)
df_transcription['processingdate'] = pd.to_datetime(df_transcription['processingdate'], errors='coerce')

# Handle missing or invalid dates
df_transcription['processingdate'] = df_transcription['processingdate'].fillna(pd.Timestamp('1970-01-01'))

# Convert dates to ISO 8601 format
df_transcription['processingdate'] = df_transcription['processingdate'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Ensure text fields are strings
text_fields = ['transcription', 'summary', 'topic_parent_class', 'topic', 'sentiment_parent_class', 'sentiment']
for field in text_fields:
    df_transcription[field] = df_transcription[field].astype(str)

# Handle missing values if any
df_transcription = df_transcription.fillna('')

# Step 7: Import 'Company' data into Weaviate

for index, row in df_company.iterrows():
    properties = {
        'company_id': row['company_id'],
        'clid': row['clid'],
        'telephone_number': row['telephone_number']
    }
    try:
        client.data_object.create(
            data_object=properties,
            class_name='Company'
        )
        if (index + 1) % 100 == 0:
            logger.info(f"Imported {index + 1} companies.")
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        logger.error(f"Error creating Company object for company_id {row['company_id']}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error creating Company object for company_id {row['company_id']}: {e}")

logger.info("Completed importing Company data.")

# Step 8: Build a mapping from 'company_id' to Weaviate UUIDs

company_uuid_map = {}
# Get all Company objects with their 'company_id' and '_additional' 'id'
try:
    response = client.query.get('Company', ['company_id']).with_additional('id').do()
    for company in response['data']['Get']['Company']:
        company_uuid_map[company['company_id']] = company['_additional']['id']
    logger.info("Company UUID mapping created successfully.")
except Exception as e:
    logger.error(f"Error fetching Company UUIDs: {e}")
    exit(1)

# Verify that all company_ids in Transcription exist in Company
unique_company_ids_transcription = set(df_transcription['company_id'].unique())
missing_company_ids = unique_company_ids_transcription - set(company_uuid_map.keys())

if missing_company_ids:
    logger.warning("The following company_ids in Transcription are missing in Company:")
    for cid in missing_company_ids:
        logger.warning(f"- {cid}")
else:
    logger.info("All company_ids in Transcription exist in Company.")

# Step 9: Import 'Transcription' data into Weaviate

for index, row in df_transcription.iterrows():
    # Generate a unique UUID for the transcription
    transcription_uuid = str(uuid.uuid4())

    company_id = row['company_id']
    if company_id in company_uuid_map:
        company_uuid = company_uuid_map[company_id]
        company_beacon = f'weaviate://localhost/Company/{company_uuid}'
    else:
        logger.warning(f"Company ID {company_id} not found in company_uuid_map. Skipping transcription.")
        company_beacon = None

    properties = {
        'company_id': row['company_id'],
        'processingdate': row['processingdate'],
        'transcription': row['transcription'],
        'summary': row['summary'],
        'topic_parent_class': row['topic_parent_class'],
        'topic': row['topic'],
        'sentiment_parent_class': row['sentiment_parent_class'],
        'sentiment': row['sentiment'],
        'ofCompany': [
            {'beacon': company_beacon}
        ] if company_beacon else []
    }

    try:
        client.data_object.create(
            data_object=properties,
            class_name='Transcription',
            uuid=transcription_uuid,
            vector=None  # Vector will be generated automatically by the vectorizer
        )
        if (index + 1) % 100 == 0:
            logger.info(f"Imported {index + 1} transcriptions.")
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        logger.error(f"Error creating Transcription object for transcription_uuid {transcription_uuid}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error creating Transcription object for transcription_uuid {transcription_uuid}: {e}")

logger.info("Completed importing Transcription data.")

# Step 10: Verify data has been imported

# Query a sample of 'Company' data
try:
    result_company = client.query.get('Company', ['company_id', 'clid', 'telephone_number']).with_limit(5).do()
    print("Sample Company data:")
    print(result_company)
except Exception as e:
    logger.error(f"Error querying Company data: {e}")

# Query a sample of 'Transcription' data
try:
    result_transcription = client.query.get('Transcription', ['company_id', 'processingdate', 'transcription']).with_limit(5).do()
    print("Sample Transcription data:")
    print(result_transcription)
except Exception as e:
    logger.error(f"Error querying Transcription data: {e}")
