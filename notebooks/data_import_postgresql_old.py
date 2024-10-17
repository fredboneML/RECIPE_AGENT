import uuid
import os
from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import logging
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# PostgreSQL connection details
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')  
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('data_import_postgresql')

# Base for ORM models
Base = declarative_base()

# Company table ORM model
class Company(Base):
    __tablename__ = 'company'
    company_id = Column(String, primary_key=True)
    clid = Column(String)
    telephone_number = Column(String)

# Transcription table ORM model
class Transcription(Base):
    __tablename__ = 'transcription'

    # Auto-generating the primary key id field
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), nullable=False)

    company_id = Column(String, nullable=False)
    processingdate = Column(DateTime, nullable=False)
    transcription = Column(String, nullable=False)
    summary = Column(String)
    topic_parent_class = Column(String)
    topic = Column(String)
    sentiment_parent_class = Column(String)
    sentiment = Column(String)
    

# Create a database engine
def create_db_engine():
    engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    engine.connect()
    logger.info("Successfully connected to the database.")
    return engine

# Load data from CSV
def load_data_from_csv():
    df_company = pd.read_csv('../data/df_company.csv')
    df_transcription = pd.read_csv('../data/df_transcription.csv')
    logger.info("df_company.csv and df_transcription.csv loaded successfully.")
    return df_company, df_transcription


# Insert data into the Company table
def insert_company_data(df_company, session):
    try:
        for index, row in df_company.iterrows():
            insert_stmt = insert(Company).values(
                company_id=str(row['company_id']),
                clid=str(row['clid']),
                telephone_number=str(row['telephone_number'])
            )
            # Handle duplicates by doing nothing if there's a conflict on 'company_id'
            do_nothing_stmt = insert_stmt.on_conflict_do_nothing(index_elements=['company_id'])
            session.execute(do_nothing_stmt)

        session.commit()
        logger.info("Completed importing Company data.")
    except Exception as e:
        logger.error(f"Error importing Company data: {e}")
        session.rollback()


# Insert data into the Transcription table
def insert_transcription_data(df_transcription, session):
    try:
        logger.info(f"Starting to import {len(df_transcription)} transcription records.")
        
        # Verify data types before insertion
        df_transcription['processingdate'] = pd.to_datetime(df_transcription['processingdate'])
        
        for index, row in df_transcription.iterrows():
            try:
                insert_stmt = insert(Transcription).values(
                    id=str(uuid.uuid4()),  # Generate a new UUID for each record
                    company_id=str(row['company_id']),
                    processingdate=row['processingdate'],
                    transcription=str(row['transcription']),
                    summary=str(row['summary']) if pd.notna(row['summary']) else None,
                    topic_parent_class=str(row['topic_parent_class']) if pd.notna(row['topic_parent_class']) else None,
                    topic=str(row['topic']) if pd.notna(row['topic']) else None,
                    sentiment_parent_class=str(row['sentiment_parent_class']) if pd.notna(row['sentiment_parent_class']) else None,
                    sentiment=str(row['sentiment']) if pd.notna(row['sentiment']) else None
                )
                do_nothing_stmt = insert_stmt.on_conflict_do_nothing(index_elements=['id'])
                session.execute(do_nothing_stmt)
                
                if (index + 1) % 1000 == 0:  # Commit every 1000 records
                    session.commit()
                    logger.info(f"Imported {index + 1} transcription records.")
            
            except Exception as e:
                logger.error(f"Error importing transcription record at index {index}: {e}")
                logger.error(f"Problematic row: {row}")
                session.rollback()
        
        session.commit()  # Final commit for any remaining records
        logger.info("Completed importing Transcription data.")
        
        # Verify the number of records inserted
        count = session.query(Transcription).count()
        logger.info(f"Total records in Transcription table after import: {count}")
        
    except Exception as e:
        logger.error(f"Error in transcription data import process: {e}")
        session.rollback()

def run_data_import():
    engine = create_db_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create tables if they don't exist
    Base.metadata.create_all(engine)
    logger.info("Database tables created successfully.")

    # Load data
    df_company, df_transcription = load_data_from_csv()
    
    # Log the number of records in the DataFrames
    logger.info(f"Number of records in df_company: {len(df_company)}")
    logger.info(f"Number of records in df_transcription: {len(df_transcription)}")

    # Insert company and transcription data
    insert_company_data(df_company, session)
    insert_transcription_data(df_transcription, session)
    
    # Verify the number of records in the database tables
    company_count = session.query(Company).count()
    transcription_count = session.query(Transcription).count()
    logger.info(f"Number of records in Company table: {company_count}")
    logger.info(f"Number of records in Transcription table: {transcription_count}")

    session.close()

if __name__ == '__main__':
    run_data_import()
