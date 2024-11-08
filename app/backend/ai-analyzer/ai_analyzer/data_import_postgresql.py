# ai_analyzer/data_import_postgresql.py
import uuid
import os
from sqlalchemy import (
    create_engine, 
    Column, 
    String, 
    DateTime, 
    ForeignKey, 
    Integer, 
    Text, 
    Boolean,
    UniqueConstraint 
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import logging
import hashlib
import datetime
from ai_analyzer.config import config, DATA_DIR, DATABASE_URL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('data_import_postgresql')

# Base for ORM models
Base = declarative_base()

# Transcription table ORM model with composite unique constraint
class Transcription(Base):
    __tablename__ = 'transcription'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String, nullable=False)
    processingdate = Column(DateTime, nullable=False)
    transcription = Column(String, nullable=False)
    summary = Column(String)
    topic = Column(String)
    sentiment = Column(String)

    # Add a unique constraint on company_id and processingdate
    __table_args__ = (
        UniqueConstraint(
            'company_id', 
            'processingdate', 
            name='unique_company_transcription'
        ),
    )

# User Memory table ORM model
class UserMemory(Base):
    __tablename__ = 'user_memory'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    conversation_id = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)

    def __init__(self, user_id, conversation_id, query, response, is_active=True):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.query = query
        self.response = response
        self.is_active = is_active


# Company table ORM model
class Company(Base):
    __tablename__ = 'company'
    company_id = Column(String, primary_key=True)
    clid = Column(String)
    telephone_number = Column(String)

# User table ORM model with role
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default='read_only')

    def has_write_permission(self):
        return self.role in ['admin', 'write']
    

# Function storing user memory
def store_user_memory(user_id, query, response):
    session = SessionLocal()
    try:
        user_memory = UserMemory(user_id=user_id, query=query, response=response)
        session.add(user_memory)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error storing user memory: {e}")
    finally:
        session.close()

# Function retrieving user memory
def get_user_memory(user_id):
    session = SessionLocal()
    try:
        memories = session.query(UserMemory).filter(UserMemory.user_id == user_id).order_by(UserMemory.timestamp).all()
        return memories
    except Exception as e:
        print(f"Error retrieving user memory: {e}")
        return []
    finally:
        session.close()


# Create a database engine
def create_db_engine():
    engine = create_engine(DATABASE_URL)
    engine.connect()
    logger.info("Successfully connected to the database.")
    return engine

# Create all tables
def create_tables(engine):
    Base.metadata.create_all(engine)
    logger.info("All database tables created successfully.")

# Create the data directory if it doesn't exist
data_dir = DATA_DIR

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Directory '{data_dir}' created.")
else:
    print(f"Directory '{data_dir}' already exists.")

# Load data from CSV

def load_data_from_csv():
    """Load data from CSV files with proper error handling"""
    try:
        # Check for both naming patterns for transcription files
        df_transcription_files = [
            file for file in os.listdir(data_dir) 
            if file.startswith('df_transcription__') and file.endswith('.csv')
        ]

        # Check for company files
        df_company_files = [
            file for file in os.listdir(data_dir) 
            if file.startswith('df_company__') and file.endswith('.csv')
        ]

        logger.info(f"Found transcription files: {df_transcription_files}")
        logger.info(f"Found company files: {df_company_files}")

        if not df_transcription_files or not df_company_files:
            logger.info("No matching data files found. Skipping data import.")
            return pd.DataFrame(), pd.DataFrame()

        # Sort files by date
        df_transcription_files = sorted(
            df_transcription_files,
            key=lambda x: datetime.datetime.strptime(x.split('__')[1].split('.csv')[0], '%Y-%m-%d'),
            reverse=True
        )
        
        df_company_files = sorted(
            df_company_files,
            key=lambda x: datetime.datetime.strptime(x.split('__')[1].split('.csv')[0], '%Y-%m-%d'),
            reverse=True
        )

        # Get the latest files
        latest_transcription = df_transcription_files[0]
        latest_company = df_company_files[0]

        logger.info(f"Loading transcription file: {latest_transcription}")
        logger.info(f"Loading company file: {latest_company}")

        # Load the data
        df_transcription = pd.read_csv(f'{data_dir}/{latest_transcription}')
        df_company = pd.read_csv(f'{data_dir}/{latest_company}')

        logger.info(f"Loaded {len(df_transcription)} transcription records")
        logger.info(f"Loaded {len(df_company)} company records")

        return df_company, df_transcription

    except Exception as e:
        logger.error(f"Error loading CSV files: {str(e)}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()
    

# Insert data into the Company table
def insert_company_data(df_company, session):
    try:
        logger.info(f"Starting to import {len(df_company)} company records.")
        for index, row in df_company.iterrows():
            insert_stmt = insert(Company).values(
                company_id=str(row['id']),
                clid=str(row['clid']),
                telephone_number=str(row['telephone_number'])
            )
            do_nothing_stmt = insert_stmt.on_conflict_do_nothing(index_elements=['company_id'])
            session.execute(do_nothing_stmt)

        session.commit()
        logger.info("Completed importing Company data.")
    except Exception as e:
        logger.error(f"Error importing Company data: {e}")
        session.rollback()

# Insert data into the Transcription table with uniqueness check
def insert_transcription_data(df_transcription, session):
    try:
        if df_transcription.empty:
            logger.warning("No transcription data to import")
            return

        logger.info(f"Starting to import {len(df_transcription)} transcription records.")
        
        # Convert processingdate to datetime
        df_transcription['processingdate'] = pd.to_datetime(df_transcription['processingdate'])
        
        # Group by company_id and get the latest record for each company
        df_transcription = df_transcription.sort_values('processingdate', ascending=False)
        df_transcription = df_transcription.drop_duplicates(subset=['company_id'], keep='first')
        
        logger.info(f"After removing duplicates: {len(df_transcription)} unique records")
        
        for index, row in df_transcription.iterrows():
            try:
                insert_stmt = insert(Transcription).values(
                    id=str(uuid.uuid4()),
                    company_id=str(row['company_id']),
                    processingdate=row['processingdate'],
                    transcription=str(row['transcription']),
                    summary=str(row['summary']) if pd.notna(row.get('summary')) else None,
                    topic=str(row['topic']) if pd.notna(row.get('topic')) else None,
                    sentiment=str(row['sentiment']) if pd.notna(row.get('sentiment')) else None
                )
                
                # Use on_conflict_do_nothing with the unique constraint
                do_nothing_stmt = insert_stmt.on_conflict_do_nothing(
                    constraint='unique_company_transcription'
                )
                session.execute(do_nothing_stmt)
                
                if (index + 1) % 1000 == 0:
                    session.commit()
                    logger.info(f"Imported {index + 1} transcription records.")
            
            except Exception as e:
                logger.error(f"Error importing transcription record at index {index}: {e}")
                logger.error(f"Problematic row: {row}")
                continue
        
        session.commit()
        logger.info("Completed importing Transcription data.")
        
        count = session.query(Transcription).count()
        logger.info(f"Total records in Transcription table after import: {count}")
        
    except Exception as e:
        logger.error(f"Error in transcription data import process: {e}")
        session.rollback()

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Update the create_user function to set role
def create_user(session, username, password, role='read_only'):
    try:
        # Check if the user already exists
        existing_user = session.query(User).filter_by(username=username).first()
        if existing_user:
            logger.info(f"User '{username}' already exists. Skipping creation.")
            return

        # Create new user with specified role
        hashed_password = hash_password(password)
        new_user = User(
            username=username, 
            password_hash=hashed_password,
            role=role
        )
        session.add(new_user)
        session.commit()
        logger.info(f"User '{username}' created successfully with role: {role}")
    except Exception as e:
        logger.error(f"Error creating user '{username}': {e}")
        session.rollback()


def run_data_import():
    """Run the data import process with proper error handling"""
    try:
        engine = create_db_engine()
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Create tables if they don't exist
            create_tables(engine)

            # Load data
            df_company, df_transcription = load_data_from_csv()
            
            # Only proceed with import if we have data
            if not df_company.empty or not df_transcription.empty:
                logger.info(f"Number of records in df_company: {len(df_company)}")
                logger.info(f"Number of records in df_transcription: {len(df_transcription)}")

                # Insert company and transcription data
                if not df_company.empty:
                    insert_company_data(df_company, session)
                if not df_transcription.empty:
                    insert_transcription_data(df_transcription, session)
            else:
                logger.info("No data to import.")
            
            # Create admin user with write permissions and regular user with read-only permissions
            logger.info("Config values:")
            logger.info(f"ADMIN_USER: {config.get('ADMIN_USER', 'not set')}")
            logger.info(f"ADMIN_PASSWORD: {'set' if config.get('ADMIN_PASSWORD') else 'not set'}")
            logger.info(f"READ_USER: {config.get('READ_USER', 'not set')}")
            logger.info(f"READ_USER_PASSWORD: {'set' if config.get('READ_USER_PASSWORD') else 'not set'}")
            create_user(
                    session, 
                    config.get('ADMIN_USER'), 
                    config.get('ADMIN_PASSWORD'), 
                    role='admin'
                )  # Admin user with write permissions
            
            create_user(
                    session, 
                    config.get('READ_USER'), 
                    config.get('READ_USER_PASSWORD'), 
                    role='read_only'
                ) # Regular user with read-only permissions
            
            # Verify the number of records in the database tables
            company_count = session.query(Company).count()
            transcription_count = session.query(Transcription).count()
            user_count = session.query(User).count()
            logger.info(f"Number of records in Company table: {company_count}")
            logger.info(f"Number of records in Transcription table: {transcription_count}")
            logger.info(f"Number of records in User table: {user_count}")

            return True
            
        finally:
            session.close()
        
    except Exception as e:
        logger.error(f"Error in run_data_import: {str(e)}")
        return False

if __name__ == '__main__':
    run_data_import()
