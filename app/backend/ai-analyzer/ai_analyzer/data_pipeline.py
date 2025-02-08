#!/usr/bin/env python3
from sqlalchemy import create_engine, Column, String, DateTime, Integer, cast, Numeric, UniqueConstraint, text
from sqlalchemy.orm import declarative_base, sessionmaker
from ai_analyzer.make_openai_call_df import make_openai_call_df
from ai_analyzer import fetch_data_from_api as fetch_data
from ai_analyzer.config import config, DATABASE_URL, DATA_DIR
import pandas as pd
import sys
import os
import uuid
from datetime import datetime
import logging
from psycopg2 import connect
import time

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

# Create engine and session factories
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
)
Session = sessionmaker(bind=engine)

# Base for ORM models
Base = declarative_base()


class Transcription(Base):
    """Model for transcription data"""
    __tablename__ = 'transcription'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    transcription_id = Column(String, nullable=False)
    # Changed from processingdate
    processing_date = Column(DateTime, nullable=False)
    transcription = Column(String, nullable=False)
    summary = Column(String)
    topic = Column(String)
    sentiment = Column(String)
    call_duration_secs = Column(Integer, nullable=True)
    tenant_code = Column(String, nullable=False)
    clid = Column(String, nullable=True)
    telephone_number = Column(String, nullable=True)
    call_direction = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            'transcription_id',
            'processing_date',  # Changed from processingdate
            'tenant_code',
            name='unique_tenant_transcription'
        ),
    )


def wait_for_db(max_retries=10, delay=10):
    """Wait for database to be ready"""
    for i in range(max_retries):
        try:
            # Try direct PostgreSQL connection
            conn = connect(
                dbname=config['POSTGRES_DB'],
                user=config['POSTGRES_USER'],
                password=config['POSTGRES_PASSWORD'],
                host='database',
                port=5432
            )
            conn.close()
            logger.info("PostgreSQL connection successful")

            # Try SQLAlchemy connection
            with Session() as session:
                session.execute(text("SELECT 1"))
            logger.info("SQLAlchemy connection successful")
            return True

        except Exception as e:
            logger.error(
                f"Database connection attempt {i + 1}/{max_retries} failed: {str(e)}")
            if i < max_retries - 1:
                time.sleep(delay)
    return False


def get_max_transcription_id(session, tenant_code):
    """Get the maximum transcription ID for a tenant"""
    try:
        max_id = session.query(Transcription.transcription_id)\
            .filter(Transcription.tenant_code == tenant_code)\
            .order_by(cast(Transcription.transcription_id, Numeric).desc())\
            .first()

        if max_id:
            logger.info(f"Retrieved maximum transcription_id: {max_id[0]}")
            return int(max_id[0])
        else:
            logger.info("No transcription records found in the database")
            return config.get('LAST_ID', 0)

    except Exception as e:
        logger.error(f"Error retrieving maximum transcription_id: {e}")
        return config.get('LAST_ID', 0)


def delete_files_by_date(directory, target_date):
    """Delete processed files for a specific date"""
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

        if deleted_files:
            logger.info(f"Successfully deleted {len(deleted_files)} files")
        else:
            logger.info(f"No files found with date {target_date}")

        return deleted_files

    except Exception as e:
        logger.error(f"Error while deleting files: {e}")
        return []


def load_data_from_csv():
    """Load data from CSV files with proper error handling"""
    try:
        # Check for transcription files
        df_transcription_files = [
            file for file in os.listdir(DATA_DIR)
            if file.startswith('df_transcription__') and file.endswith('.csv')
        ]

        logger.info(f"Found transcription files: {df_transcription_files}")

        if not df_transcription_files:
            logger.info("No matching data files found. Skipping data import.")
            return pd.DataFrame()

        # Sort files by date
        df_transcription_files = sorted(
            df_transcription_files,
            key=lambda x: datetime.strptime(
                x.split('__')[1].split('.csv')[0], '%Y-%m-%d'),
            reverse=True
        )

        # Get the latest file
        latest_transcription = df_transcription_files[0]
        logger.info(f"Loading transcription file: {latest_transcription}")

        # Load the data
        df_transcription = pd.read_csv(f'{DATA_DIR}/{latest_transcription}')
        logger.info(f"Loaded {len(df_transcription)} transcription records")

        return df_transcription

    except Exception as e:
        logger.error(f"Error loading CSV files: {str(e)}", exc_info=True)
        return pd.DataFrame()


def process_data_pipeline(session, tenant_code):
    """Process data pipeline for a specific tenant"""
    try:
        logger.info(f"Processing pipeline for tenant: {tenant_code}")

        # Get last processed ID
        last_id = get_max_transcription_id(session, tenant_code)
        logger.info(f"Retrieved last_id: {last_id}")

        # Fetch new data from API
        new_data = fetch_data.fetch_data_from_api(
            config['URL'],
            config['API_KEY'],
            last_id,
            tenant_code,
            config['LIMIT']
        )

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

            # Load data
            df_transcription = load_data_from_csv()

            # Only proceed with import if we have data
            if not df_transcription.empty:
                logger.info(
                    f"Number of records in df_transcription: {len(df_transcription)}")

                # Insert processed data
                try:
                    # Check for existing records to avoid duplicates
                    existing_records = session.execute(
                        text("""
                            SELECT transcription_id, processing_date 
                            FROM transcription 
                            WHERE tenant_code = :tenant_code
                        """),
                        {"tenant_code": tenant_code}
                    ).fetchall()

                    # Create a set of existing records for faster lookup
                    existing_set = {
                        (str(rec.transcription_id),
                         pd.Timestamp(rec.processing_date))
                        for rec in existing_records
                    }

                    stmt = text("""
                        INSERT INTO transcription (
                            id, transcription_id, processing_date, transcription,
                            summary, topic, sentiment, call_duration_secs,
                            tenant_code, clid, telephone_number, call_direction
                        ) VALUES (
                            :id, :transcription_id, :processing_date, :transcription,
                            :summary, :topic, :sentiment, :call_duration_secs,
                            :tenant_code, :clid, :telephone_number, :call_direction
                        ) ON CONFLICT ON CONSTRAINT unique_tenant_transcription DO NOTHING
                    """)

                    successful_inserts = 0
                    for _, row in df_transcription.iterrows():
                        # Convert date for comparison
                        row_date = pd.to_datetime(row['processing_date'])

                        # Check if record already exists
                        if (str(row['transcription_id']), row_date) in existing_set:
                            continue

                        # Generate a unique UUID for the id field
                        values = {
                            'id': str(uuid.uuid4()),  # Generate unique ID
                            'transcription_id': str(row['transcription_id']),
                            'processing_date': row_date,
                            'transcription': str(row['transcription']),
                            'summary': str(row.get('summary', '')),
                            'topic': str(row.get('topic', '')),
                            'sentiment': str(row.get('sentiment', '')),
                            'tenant_code': tenant_code,
                            'call_duration_secs': int(row['call_duration_secs']) if pd.notna(row.get('call_duration_secs')) else None,
                            'clid': str(row.get('clid', '')),
                            'telephone_number': str(row.get('telephone_number', '')),
                            'call_direction': str(row.get('call_direction', ''))
                        }

                        session.execute(stmt, values)
                        successful_inserts += 1

                        # Commit every 100 records
                        if successful_inserts % 100 == 0:
                            session.commit()
                            logger.info(
                                f"Committed {successful_inserts} records")

                    # Final commit for remaining records
                    session.commit()
                    logger.info(
                        f"Successfully inserted {successful_inserts} new records for tenant {tenant_code}")

                except Exception as e:
                    logger.error(f"Error inserting data: {e}")
                    session.rollback()
                    raise

            # Cleanup processed files
            deleted_files = delete_files_by_date(DATA_DIR, current_date)
            logger.info(f"Cleaned up processed files: {deleted_files}")

        else:
            logger.info("No new data to process")

        return True

    except Exception as e:
        logger.error(
            f"Error in pipeline processing for tenant {tenant_code}: {e}")
        return False


def main():
    """Main function to run the data pipeline"""
    try:
        logger.info("Starting data pipeline execution")

        # Wait for database
        if not wait_for_db(max_retries=10, delay=10):
            logger.error("Failed to connect to database")
            sys.exit(1)

        # Create session and process each tenant
        with Session() as session:
            try:
                # Get all active tenants
                tenant_results = session.execute(text("""
                    SELECT tenant_code, tenant_code_alias 
                    FROM tenant_codes 
                    WHERE tenant_code IS NOT NULL
                """)).fetchall()

                logger.info(f"Found {len(tenant_results)} tenants to process")

                # Process each tenant
                for tenant_code, alias in tenant_results:
                    logger.info(f"Processing tenant: {tenant_code} ({alias})")
                    if not process_data_pipeline(session, tenant_code):
                        logger.error(
                            f"Failed to process tenant: {tenant_code}")

            except Exception as e:
                logger.error(f"Error during tenant processing: {e}")
                raise

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
