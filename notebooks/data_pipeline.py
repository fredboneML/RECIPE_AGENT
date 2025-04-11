# ai_analyzer/data_pipeline.py
import sys
import os
import uuid
from datetime import datetime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, String, DateTime, text, cast, Numeric, UniqueConstraint
import logging
import pandas as pd
from ai_analyzer_old.config import config, DATABASE_URL, DATA_DIR
from ai_analyzer_old import fetch_data_from_api as fetch_data
from ai_analyzer_old.data_import_postgresql import run_data_import
from ai_analyzer_old.make_openai_call_df import make_openai_call_df
import time
import contextlib

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

# Get the current date
current_date = datetime.now().date().strftime("%Y-%m-%d")

# Base for ORM models
Base = declarative_base()


class Transcription(Base):
    __tablename__ = 'transcription'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String, nullable=False)
    processingdate = Column(DateTime, nullable=False)
    transcription = Column(String, nullable=False)
    summary = Column(String)
    topic = Column(String)
    sentiment = Column(String)

    __table_args__ = (
        UniqueConstraint(
            'company_id',
            'processingdate',
            name='unique_company_transcription'
        ),
    )


def create_db_engine():
    """Create a database engine with connection pooling."""
    return create_engine(
        DATABASE_URL,
        pool_size=3,  # Reduced from 5
        max_overflow=5,  # Reduced from 10
        pool_timeout=30,
        pool_pre_ping=True,
        pool_recycle=1800  # Recycle connections after 30 minutes
    )


def get_db_session():
    """Get a database session with proper cleanup."""
    engine = create_db_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def get_max_company_id(session):
    """Get the maximum company ID with improved error handling"""
    try:
        result = session.execute(
            text("SELECT MAX(CAST(company_id AS NUMERIC)) FROM transcription")
        ).scalar()

        if result is not None:
            logger.info(f"Retrieved maximum company_id: {result}")
            return int(result)
        else:
            logger.info("No transcription records found in the database")
            return config.get('LAST_ID', 0)
    except Exception as e:
        logger.error(f"Error retrieving maximum company_id: {e}")
        return config.get('LAST_ID', 0)


def safely_read_csv(file_path, max_retries=3, retry_delay=5):
    """Safely read CSV files with retries and proper resource management"""
    for attempt in range(max_retries):
        try:
            # Use a context manager to ensure proper file handling
            return pd.read_csv(file_path)
        except OSError as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1} to read {file_path} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to read {file_path} after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error reading {file_path}: {e}")
            raise


def delete_files_by_date(directory, target_date):
    """Delete files with improved error handling and logging"""
    try:
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist")
            return []

        deleted_files = []
        for filename in os.listdir(directory):
            try:
                if '__' in filename and '.csv' in filename:
                    date_part = filename.split('__')[1].split('.csv')[0]

                    if date_part == target_date:
                        file_path = os.path.join(directory, filename)
                        # Additional check before deletion
                        if os.path.exists(file_path):
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


def process_data():
    """Process data with proper connection handling."""
    with contextlib.closing(get_db_session()) as session:
        # Your data processing code here
        pass


def main():
    """Main pipeline execution with improved error handling and retries"""
    try:
        logger.info("Starting data pipeline execution")
        logger.info(
            f"Using database: {config['POSTGRES_DB']} on host: {config['DB_HOST']}")

        # Create engine and session with retries
        engine = create_db_engine()
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Step 1: Query max company ID
            logger.info("Step 1: Querying max company ID")
            last_id = int(get_max_company_id(session=session))
            logger.info(f"Retrieved last_id: {last_id}")

            # Step 2: Fetch new data
            logger.info("Step 2: Fetching data from API")
            fetch_data.fetch_data_from_api(
                config['URL'], config['API_KEY'], last_id, config['LIMIT'])

            # Find and sort data files
            df_files = [f for f in os.listdir(
                DATA_DIR) if f.startswith('df__')]
            df_files = sorted(
                df_files,
                key=lambda x: datetime.strptime(
                    x.split('__')[1].split('.csv')[0], '%Y-%m-%d'),
                reverse=True
            )

            if len(df_files) > 1:
                logger.info("Step 3: Processing new data")
                latest_file = df_files[0]
                logger.info(f"Processing file: {latest_file}")

                # Safely read the CSV file
                df = safely_read_csv(os.path.join(DATA_DIR, latest_file))
                logger.info(f"Number of rows to process: {len(df)}")

                # Process the data
                make_openai_call_df(df=df, model="gpt-4o-mini-2024-07-18")

                # Import processed data
                logger.info("Step 4: Running data import")
                success = run_data_import()
                if not success:
                    raise Exception("Data import failed")

                # Clean up
                logger.info("Step 5: Cleaning up files")
                deleted_files = delete_files_by_date(DATA_DIR, current_date)
                logger.info(f"Deleted files: {deleted_files}")
            else:
                logger.info("No new data files to process")

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
