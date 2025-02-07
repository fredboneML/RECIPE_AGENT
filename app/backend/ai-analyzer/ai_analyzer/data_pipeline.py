#!/usr/bin/env python3
from sqlalchemy import create_engine, Column, String, DateTime, Integer, cast, Numeric, UniqueConstraint, text
from ai_analyzer.make_openai_call_df import make_openai_call_df
from sqlalchemy import create_engine, Column, Integer, String, text, func, and_
from ai_analyzer.data_import_postgresql import (
    run_data_import,
    run_create_tables,
    get_tenant_codes,
    init_db
)
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
from ai_analyzer.tenant_manager import setup_new_tenant_partition, setup_user_table_triggers
from psycopg2 import connect
import time
from sqlalchemy.orm import sessionmaker
from ai_analyzer.data_import_postgresql import check_user_exists, create_user, check_tenant_exists, initialize_default_tenant, check_restricted_tables_populated, populate_restricted_tables
# Create a database engine

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

# Create initial engine with explicit host and port
initial_connection_url = f"postgresql://{config['POSTGRES_USER']}:{
    config['POSTGRES_PASSWORD']}@database:5432/{config['POSTGRES_DB']}"

engine = create_engine(
    initial_connection_url,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
)
# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Session = sessionmaker(bind=engine)

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


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default='read_only')
    tenant_code = Column(String, nullable=False)

    def has_write_permission(self):
        return self.role in ['admin', 'write']


def wait_for_db(max_retries=10, delay=10):
    """Wait for database to be ready"""
    logger = logging.getLogger(__name__)

    for i in range(max_retries):
        try:
            # Try direct PostgreSQL connection first using database hostname
            conn = connect(
                dbname=config['POSTGRES_DB'],
                user=config['POSTGRES_USER'],
                password=config['POSTGRES_PASSWORD'],
                host='database',  # Use the service name from docker-compose
                port=5432  # Use internal port 5432
            )
            conn.close()
            logger.info("Direct PostgreSQL connection successful")

            # Then try SQLAlchemy connection
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            logger.info("SQLAlchemy connection successful")

            return True
        except Exception as e:
            logger.error(f"Database connection attempt {
                         i + 1}/{max_retries} failed: {str(e)}")
            if i < max_retries - 1:
                time.sleep(delay)
    return False


def create_db_engine():
    """Create and verify database engine"""
    try:
        # Create connection URL with explicit host and port
        connection_url = f"postgresql://{config['POSTGRES_USER']}:{
            config['POSTGRES_PASSWORD']}@database:5432/{config['POSTGRES_DB']}"

        new_engine = create_engine(
            connection_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True
        )
        new_engine.connect()
        logger.info("Successfully connected to the database")
        return new_engine
    except Exception as e:
        logger.error(f"Error creating database engine: {e}")
        raise


# get max id of the transcription table
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


def verify_tables_exist(engine):
    """Verify that all required tables exist in the database"""
    try:
        with engine.begin() as connection:
            # Check each table explicitly
            tables_to_check = [
                'transcription',
                'users',
                'tenant_codes',
                'user_memory',
                'restricted_tables'
            ]

            # Query to check table existence
            check_sql = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = :table_name;
            """)

            missing_tables = []
            for table in tables_to_check:
                result = connection.execute(
                    check_sql, {"table_name": table}).first()
                if not result:
                    missing_tables.append(table)

            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                return False

            logger.info("All required tables exist")
            return True

    except Exception as e:
        logger.error(f"Error verifying tables: {e}")
        return False


def process_data_pipeline(session, tenant_code):
    """Process data pipeline for a specific tenant"""
    try:
        logger.info(f"Processing pipeline for tenant: {tenant_code}")

        # 1. Get last processed ID for this tenant
        last_id = get_max_transcription_id(session, tenant_code)
        if last_id is not None:
            last_id = int(last_id)
        else:
            last_id = config['LAST_ID']
        logger.info(f"Retrieved last_id: {last_id}")

        # 2. Fetch new data from API
        new_data = fetch_data.fetch_data_from_api(
            config['URL'],
            config['API_KEY'],
            last_id,
            tenant_code,
            config['LIMIT']
        )

        if new_data:
            # Process new data
            df__file_name = sorted(
                [f for f in os.listdir(DATA_DIR) if f.startswith('df__')],
                key=lambda x: datetime.strptime(
                    x.split('__')[1].split('.csv')[0], '%Y-%m-%d'),
                reverse=True
            )[0]

            logger.info(f"Processing file: {df__file_name}")
            df = pd.read_csv(f'{DATA_DIR}/{df__file_name}')
            logger.info(f"Number of rows to process: {len(df)}")

            # Generate sentiment and topics
            make_openai_call_df(df=df, model="gpt-4o-mini-2024-07-18")

            # Import the processed data
            run_data_import()

            # Cleanup
            deleted_files = delete_files_by_date(DATA_DIR, current_date)
            logger.info(f"Deleted files: {deleted_files}")
        else:
            logger.info("No new data to process")

        return True

    except Exception as e:
        logger.error(f"Error in pipeline processing for tenant {
                     tenant_code}: {e}")
        return False


if __name__ == "__main__":
    #########################################################################################
    # --------------------------------------- Pipeline ---------------------------------------
    #########################################################################################
    try:
        logger.info("Starting data pipeline execution")

        # Wait for database
        if not wait_for_db(max_retries=10, delay=10):
            logger.error("Failed to connect to database")
            sys.exit(1)

        # Create engine and session
        engine = create_db_engine()
        SessionLocal = sessionmaker(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # First, verify and create users if needed
            logger.info("Checking and creating users if needed...")
            logger.info("Config values:")
            logger.info(f"ADMIN_USER: {config.get('ADMIN_USER', 'not set')}")
            logger.info(f"ADMIN_PASSWORD: {'set' if config.get(
                'ADMIN_PASSWORD') else 'not set'}")
            logger.info(f"READ_USER: {config.get('READ_USER', 'not set')}")
            logger.info(f"READ_USER_PASSWORD: {'set' if config.get(
                'READ_USER_PASSWORD') else 'not set'}")

            # Check and create admin user
            admin_user = config.get('ADMIN_USER')
            if admin_user and not check_user_exists(session, admin_user, config.get('TENANT_CODE')):
                logger.info(f"Creating admin user: {admin_user}")
                create_user(
                    session,
                    admin_user,
                    config.get('ADMIN_PASSWORD'),
                    tenant_code=config.get('TENANT_CODE'),
                    role='admin'
                )
            else:
                logger.info("Admin user already exists")

            # Check and create read user
            read_user = config.get('READ_USER')
            if read_user and not check_user_exists(session, read_user, config.get('TENANT_CODE')):
                logger.info(f"Creating read user: {read_user}")
                create_user(
                    session,
                    read_user,
                    config.get('READ_USER_PASSWORD'),
                    tenant_code=config.get('TENANT_CODE'),
                    role='read_only'
                )
            else:
                logger.info("Read user already exists")

            # Check and initialize default tenant
            if not check_tenant_exists(session, 'tientelecom'):
                logger.info("Initializing default tenant")
                initialize_default_tenant(session)
            else:
                logger.info("Default tenant already exists")

            # Check and populate restricted tables
            if not check_restricted_tables_populated(session):
                logger.info("Populating restricted tables")
                populate_restricted_tables(session)
            else:
                logger.info("Restricted tables already populated")

            # Initialize database and run data import
            success = run_data_import()

            if not success:
                logger.error("Failed to import initial data")
                sys.exit(1)

            # Verify tables exist
            if not verify_tables_exist(engine):
                logger.error("Required tables are missing")
                sys.exit(1)

            # Set up triggers
            setup_user_table_triggers(engine)

            # Get all tenant aliases
            tenant_list = get_tenant_codes(session)
            all_tenants = [tenant['alias'] for tenant in tenant_list]
            logger.info(f"Processing for tenants: {all_tenants}")

            # Verify final counts
            transcription_count = session.query(Transcription).count()
            user_count = session.query(User).count()
            logger.info(f"Number of records in Transcription table: {
                        transcription_count}")
            logger.info(f"Number of records in User table: {user_count}")

            # Process each tenant
            for tenant_code in all_tenants:
                process_data_pipeline(session, tenant_code)

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


#########################################################################################
# --------------------------------------- End Pipeline -----------------------------------
#########################################################################################
