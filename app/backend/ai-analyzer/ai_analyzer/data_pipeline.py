#!/usr/bin/env python3
from sqlalchemy import create_engine, Column, String, DateTime, Integer, cast, Numeric, UniqueConstraint
from ai_analyzer.make_openai_call_df import make_openai_call_df
from ai_analyzer.data_import_postgresql import run_data_import, run_create_tables
from ai_analyzer import fetch_data_from_api as fetch_data
from ai_analyzer.config import config, DATABASE_URL, DATA_DIR
import pandas as pd
from sqlalchemy.orm import declarative_base, sessionmaker
import sys
import os
import uuid
from datetime import datetime
import logging
import importlib


def verify_packages():
    """Verify all required packages are installed"""
    required_packages = [
        'sqlalchemy',
        'pandas',
        'psycopg2',
        'openai',
        'dotenv',  # Changed from python-dotenv to dotenv
        'fastapi',
        'uvicorn'
    ]

    logger = logging.getLogger('data_pipeline')
    logger.info("Verifying required packages...")

    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing.append(package)
            logger.error(f"✗ {package} is missing")

    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        sys.exit(1)
    else:
        logger.info("All required packages are installed")


# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/cron.log')
    ]
)
logger = logging.getLogger('data_pipeline')

# Verify packages
verify_packages()

# Now import the packages after verification


# Get the current date
current_date = datetime.now().date().strftime("%Y-%m-%d")

# Base for ORM models
Base = declarative_base()

# Transcription table ORM model


class Transcription(Base):
    __tablename__ = 'transcription'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    transcription_id = Column(String, nullable=False)
    processingdate = Column(DateTime, nullable=False)
    transcription = Column(String, nullable=False)
    summary = Column(String)
    topic = Column(String)
    sentiment = Column(String)
    call_duration_secs = Column(Integer, nullable=True)
    tenant_code = Column(String, nullable=True)
    clid = Column(String, nullable=True)  # Caller ID
    telephone_number = Column(String, nullable=True)
    # e.g., 'IN', 'LOCAL', None, 'OUT'
    call_direction = Column(String, nullable=True)

    # Add a unique constraint on transcription_id and processingdate
    __table_args__ = (
        UniqueConstraint(
            'transcription_id',
            'processingdate',
            name='unique_company_transcription'
        ),
    )

# Create a database engine


def create_db_engine():
    engine = create_engine(DATABASE_URL)
    engine.connect()
    logger.info("Successfully connected to the database.")
    return engine

# Return the max id of the transcription table


def get_max_transcription_id(session, tenant_code):
    try:
        max_id = session.query(Transcription.transcription_id)\
            .filter(Transcription.tenant_code == tenant_code)\
            .order_by(cast(Transcription.transcription_id, Numeric).desc())\
            .first()

        if max_id:
            logger.info(f"Retrieved maximum transcription_id: {max_id[0]}")
            return max_id[0]
        else:
            logger.info("No transcription records found in the database")
            return config['LAST_ID']
    except Exception as e:
        logger.error(f"Error retrieving maximum transcription_id: {e}")
        return config['LAST_ID']


def delete_files_by_date(directory, target_date):
    try:
        deleted_files = []
        for filename in os.listdir(directory):
            try:
                if '__' in filename and '.csv' in filename:
                    date_part = filename.split('__')[1].split('.csv')[0]

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


if __name__ == "__main__":
    try:
        logger.info("Starting data pipeline execution")

        # Log the current configuration
        logger.info(f"Using database: {config['POSTGRES_DB']} on host: {
                    config['DB_HOST']}")

        # Create engine and session
        engine = create_db_engine()
        Session = sessionmaker(bind=engine)
        session = Session()

#########################################################################################
# --------------------------------------- Pipeline ---------------------------------------
#########################################################################################

        try:
            run_create_tables()
            # 1st Step: Querying the max id of the transcription table
            logger.info("Step 1: Querying max company ID")
            tenant_code = config['TENANT_CODE']
            last_id = get_max_transcription_id(
                session=session, tenant_code=config['TENANT_CODE'])
            if last_id is not None:
                last_id = int(last_id)
            else:
                last_id = config['LAST_ID']
            logger.info(f"Retrieved last_id: {last_id}")

            # 2nd Step: Fetching data using the last_id found in the database
            logger.info("Step 2: Fetching data from API")
            new_data = fetch_data.fetch_data_from_api(
                config['URL'], config['API_KEY'], last_id, config['TENANT_CODE'], config['LIMIT'])

            df__file_name = [file for file in os.listdir(
                DATA_DIR) if file.startswith('df__')]
            df__file_name = sorted(df__file_name,
                                   key=lambda x: datetime.strptime(
                                       x.split('__')[1].split('.csv')[0], '%Y-%m-%d'),
                                   reverse=True)

            logger.info(f"Files found: {df__file_name}")
            # Only process if there are new data files besides the initial one
            if new_data:

                # 3rd Step: Generating sentiment and topic of the new data
                logger.info("Step 3: Processing new data")
                logger.info("Found new data files to process")
                df__file_name = df__file_name[0]
                logger.info(f"Processing file: {df__file_name}")
                df = pd.read_csv(f'{DATA_DIR}/{df__file_name}')
                logger.info(f"Number of rows to process: {len(df)}")
                make_openai_call_df(df=df, model="gpt-4o-mini-2024-07-18")

                # 4th Step: Insert the generated topics and sentiments to the db
                logger.info("Step 4: Running data import")
                run_data_import()

                # 5th Step: Clean up
                logger.info("Step 5: Cleaning up files")
                deleted_files = delete_files_by_date(DATA_DIR, current_date)
                logger.info(f"Deleted files: {deleted_files}")
                logger.info("Pipeline execution completed successfully")
            else:
                logger.info("No new data files to process")

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise
#########################################################################################
# --------------------------------------- End Pipeline -----------------------------------
#########################################################################################
