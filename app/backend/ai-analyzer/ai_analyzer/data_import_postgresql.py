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
    UniqueConstraint,
    Index  # Added Index import
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import logging
import hashlib
import datetime
from ai_analyzer.config import config, DATA_DIR, DATABASE_URL
from datetime import datetime as dt
from datetime import timezone
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('data_import_postgresql')

# Base for ORM models
Base = declarative_base()

# Transcription table ORM model with composite unique constraint


class Transcription(Base):
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
    tenant_code = Column(String, nullable=True)  # Made nullable
    clid = Column(String, nullable=True)
    telephone_number = Column(String, nullable=True)
    call_direction = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            'transcription_id',
            'processing_date',
            name='unique_company_transcription'
        ),
    )


class UserMemory(Base):
    __tablename__ = 'user_memory'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    conversation_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    # To maintain message order within conversation
    message_order = Column(Integer, nullable=False)
    expires_at = Column(DateTime, nullable=False)  # For 30-day retention

    __table_args__ = (
        UniqueConstraint('conversation_id', 'message_order',
                         name='unique_message_order'),
        Index('idx_user_conversations', user_id, conversation_id),
    )

    def __init__(self, user_id, conversation_id, query, response, title=None, message_order=0):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.query = query
        self.response = response
        self.title = title or self._generate_title(query)
        self.message_order = message_order
        self.expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=30)

    def _generate_title(self, query):
        # Generate a title from the first query of conversation
        return query[:50] + "..." if len(query) > 50 else query

# User table ORM model with role


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default='read_only')
    tenant_code = Column(String, nullable=False)

    def has_write_permission(self):
        return self.role in ['admin', 'write']


class RestrictedTable(Base):
    """
    Model to track restricted tables that should not be accessible
    through general queries or analytics
    """
    __tablename__ = 'restricted_tables'

    table_name = Column(String, primary_key=True)
    reason = Column(String, nullable=False)
    added_at = Column(DateTime, default=lambda: dt.now(timezone.utc))
    added_by = Column(String, nullable=False)

    def __repr__(self):
        return f"<RestrictedTable(table_name='{self.table_name}', reason='{self.reason}')>"


def populate_restricted_tables(session, added_by="system"):
    """
    Populate the restricted_tables table with default restricted tables
    """
    try:
        # Define default restricted tables and their reasons
        restricted_tables = [
            {
                "table_name": "users",
                "reason": "Contains sensitive user authentication and authorization data"
            },
            {
                "table_name": "query_cache",
                "reason": "Internal system table for query optimization"
            },
            {
                "table_name": "query_performance",
                "reason": "Internal system table for performance monitoring"
            },
            {
                "table_name": "user_memory",
                "reason": "Contains user conversation history and personal data"
            }
        ]

        # Add each restricted table
        for table in restricted_tables:
            # Check if entry already exists
            existing = session.query(RestrictedTable).filter_by(
                table_name=table["table_name"]
            ).first()

            if not existing:
                restricted_table = RestrictedTable(
                    table_name=table["table_name"],
                    reason=table["reason"],
                    added_by=added_by,
                    added_at=dt.now(timezone.utc)
                )
                session.add(restricted_table)
                logger.info(f"Added restricted table: {table['table_name']}")
            else:
                logger.info(f"Table {table['table_name']} already restricted")

        session.commit()
        return True

    except Exception as e:
        logger.error(f"Error populating restricted tables: {e}")
        session.rollback()
        return False


def is_table_restricted(session, table_name: str) -> bool:
    """
    Check if a table is restricted
    """
    try:
        return session.query(RestrictedTable).filter_by(
            table_name=table_name
        ).first() is not None
    except Exception as e:
        logger.error(f"Error checking if table is restricted: {e}")
        return False


def get_restricted_tables(session) -> list:
    """
    Get list of all restricted table names
    """
    try:
        restricted = session.query(RestrictedTable).all()
        return [table.table_name for table in restricted]
    except Exception as e:
        logger.error(f"Error getting restricted tables: {e}")
        return []


class TenantCode(Base):
    """
    Model to manage tenant codes and their aliases
    """
    __tablename__ = 'tenant_codes'

    tenant_code = Column(String, primary_key=True)
    tenant_code_alias = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=lambda: dt.now(timezone.utc))

    def __repr__(self):
        return f"<TenantCode(tenant_code='{self.tenant_code}', alias='{self.tenant_code_alias}')>"


def initialize_default_tenant(session) -> bool:
    """
    Initialize the default tenant (tientelecom)

    Args:
        session: SQLAlchemy session

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if default tenant exists
        existing = session.query(TenantCode).filter_by(
            tenant_code='tientelecom'
        ).first()

        if not existing:
            default_tenant = TenantCode(
                tenant_code='tientelecom',
                tenant_code_alias='tientelecom'
            )
            session.add(default_tenant)
            session.commit()
            logger.info(
                "Default tenant 'tientelecom' initialized successfully")
            return True
        else:
            logger.info("Default tenant 'tientelecom' already exists")
            return True

    except Exception as e:
        logger.error(f"Error initializing default tenant: {e}")
        session.rollback()
        return False


def get_tenant_codes(session) -> List[dict]:
    """
    Get list of all available tenant codes

    Args:
        session: SQLAlchemy session

    Returns:
        List[dict]: List of dictionaries containing tenant code information
    """
    try:
        tenants = session.query(TenantCode).all()
        return [
            {
                "tenant_code": tenant.tenant_code,
                "alias": tenant.tenant_code_alias,
                "created_at": tenant.created_at
            }
            for tenant in tenants
        ]
    except Exception as e:
        logger.error(f"Error retrieving tenant codes: {e}")
        return []


def get_tenant_by_code(session, code: str) -> Optional[TenantCode]:
    """
    Get tenant by tenant code or alias

    Args:
        session: SQLAlchemy session
        code: Tenant code or alias to look up

    Returns:
        Optional[TenantCode]: TenantCode object if found, None otherwise
    """
    try:
        return session.query(TenantCode).filter(
            (TenantCode.tenant_code == code) |
            (TenantCode.tenant_code_alias == code)
        ).first()
    except Exception as e:
        logger.error(f"Error retrieving tenant by code: {e}")
        return None


# conversation management functions

def store_conversation(session, user_id, conversation_id, query, response, title=None, message_order=None):
    """Store a conversation message"""
    try:
        if message_order is None:
            # Get the last message order for this conversation
            last_message = session.query(UserMemory)\
                .filter(UserMemory.conversation_id == conversation_id)\
                .order_by(UserMemory.message_order.desc())\
                .first()
            message_order = (last_message.message_order +
                             1) if last_message else 0

        memory = UserMemory(
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
            response=response,
            title=title,
            message_order=message_order
        )
        session.add(memory)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error storing conversation: {e}")
        return False


def get_user_conversations(session, user_id, limit=5):
    """Get user's recent conversations"""
    try:
        # Get distinct conversations with their latest message
        subquery = session.query(
            UserMemory.conversation_id,
            func.max(UserMemory.timestamp).label('max_timestamp')
        ).filter(
            UserMemory.user_id == user_id,
            UserMemory.is_active == True,
            UserMemory.expires_at > datetime.datetime.utcnow()
        ).group_by(UserMemory.conversation_id).subquery()

        conversations = session.query(UserMemory).join(
            subquery,
            and_(
                UserMemory.conversation_id == subquery.c.conversation_id,
                UserMemory.timestamp == subquery.c.max_timestamp
            )
        ).order_by(UserMemory.timestamp.desc()).limit(limit).all()

        return conversations
    except Exception as e:
        print(f"Error retrieving conversations: {e}")
        return []


def get_conversation_messages(session, conversation_id):
    """Get all messages in a conversation"""
    try:
        messages = session.query(UserMemory)\
            .filter(
                UserMemory.conversation_id == conversation_id,
                UserMemory.is_active == True,
                UserMemory.expires_at > datetime.datetime.utcnow()
        )\
            .order_by(UserMemory.message_order)\
            .all()
        return messages
    except Exception as e:
        print(f"Error retrieving conversation messages: {e}")
        return []


def cleanup_expired_conversations():
    """Cleanup expired conversations (can be run as a periodic task)"""
    try:
        session = SessionLocal()
        session.query(UserMemory)\
            .filter(UserMemory.expires_at <= datetime.datetime.utcnow())\
            .delete()
        session.commit()
    except Exception as e:
        print(f"Error cleaning up conversations: {e}")
        session.rollback()
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

        logger.info(f"Found transcription files: {df_transcription_files}")

        if not df_transcription_files:
            logger.info("No matching data files found. Skipping data import.")
            return pd.DataFrame(), pd.DataFrame()

        # Sort files by date
        df_transcription_files = sorted(
            df_transcription_files,
            key=lambda x: datetime.datetime.strptime(
                x.split('__')[1].split('.csv')[0], '%Y-%m-%d'),
            reverse=True
        )

        # Get the latest files
        latest_transcription = df_transcription_files[0]

        logger.info(f"Loading transcription file: {latest_transcription}")

        # Load the data
        df_transcription = pd.read_csv(f'{data_dir}/{latest_transcription}')

        logger.info(f"Loaded {len(df_transcription)} transcription records")

        return df_transcription

    except Exception as e:
        logger.error(f"Error loading CSV files: {str(e)}", exc_info=True)
        return pd.DataFrame()


def insert_transcription_data(df_transcription, session):
    try:
        if df_transcription.empty:
            logger.warning("No transcription data to import")
            return

        logger.info(f"Starting to import {
                    len(df_transcription)} transcription records.")

        # Convert processingdate to processing_date if needed
        if 'processingdate' in df_transcription.columns:
            df_transcription = df_transcription.rename(
                columns={'processingdate': 'processing_date'})

        # Convert processing_date to datetime
        df_transcription['processing_date'] = pd.to_datetime(
            df_transcription['processing_date'])

        # Group by transcription_id and get the latest record
        df_transcription = df_transcription.sort_values(
            'processing_date', ascending=False)
        df_transcription = df_transcription.drop_duplicates(
            subset=['transcription_id'], keep='first')

        for index, row in df_transcription.iterrows():
            try:
                insert_stmt = insert(Transcription).values(
                    id=str(uuid.uuid4()),
                    transcription_id=str(row['transcription_id']),
                    processing_date=row['processing_date'],
                    transcription=str(row['transcription']),
                    summary=str(row['summary']) if pd.notna(
                        row.get('summary')) else None,
                    topic=str(row['topic']) if pd.notna(
                        row.get('topic')) else None,
                    sentiment=str(row['sentiment']) if pd.notna(
                        row.get('sentiment')) else None,
                    tenant_code=str(row.get('tenant_code')) if pd.notna(
                        row.get('tenant_code')) else None,
                    call_duration_secs=int(row['call_duration_secs']) if pd.notna(
                        row.get('call_duration_secs')) else None,
                    clid=str(row.get('clid')) if pd.notna(
                        row.get('clid')) else None,
                    telephone_number=str(row.get('telephone_number')) if pd.notna(
                        row.get('telephone_number')) else None,
                    call_direction=str(row.get('call_direction')) if pd.notna(
                        row.get('call_direction')) else None
                )

                do_nothing_stmt = insert_stmt.on_conflict_do_nothing(
                    constraint='unique_company_transcription'
                )
                session.execute(do_nothing_stmt)

                if (index + 1) % 1000 == 0:
                    session.commit()
                    logger.info(f"Imported {index + 1} transcription records.")

            except Exception as e:
                logger.error(
                    f"Error importing transcription record at index {index}: {e}")
                logger.error(f"Problematic row: {row}")
                continue

        session.commit()
        logger.info("Completed importing Transcription data.")

    except Exception as e:
        logger.error(f"Error in transcription data import process: {e}")
        session.rollback()
# Hash password


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Update the create_user function to set role
def create_user(session, username, password, tenant_code, role='read_only'):
    try:
        # Check if the user already exists
        existing_user = session.query(
            User).filter_by(username=username, tenant_code=tenant_code).first()
        if existing_user:
            logger.info(
                f"User '{username}' already exists. Skipping creation.")
            return

        # Create new user with specified role
        hashed_password = hash_password(password)
        new_user = User(
            username=username,
            tenant_code=tenant_code,
            password_hash=hashed_password,
            role=role
        )
        session.add(new_user)
        session.commit()
        logger.info(
            f"User '{username}' created successfully with role: {role}")
    except Exception as e:
        logger.error(f"Error creating user '{username}': {e}")
        session.rollback()


def run_create_tables():
    """Create required tables if they don't exist"""
    try:
        engine = create_db_engine()

        try:
            # Create tables if they don't exist
            create_tables(engine)
            return True
        except Exception as e:
            logger.error(f"Error in run_create_tables: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error in run_create_tables: {str(e)}")
        return False


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
            df_transcription = load_data_from_csv()

            # Only proceed with import if we have data
            if not df_transcription.empty:
                logger.info(f"Number of records in df_transcription: {
                            len(df_transcription)}")

                # Insert transcription data
                if not df_transcription.empty:
                    insert_transcription_data(df_transcription, session)
            else:
                logger.info("No data to import.")

            # Create admin user with write permissions and regular user with read-only permissions
            logger.info("Config values:")
            logger.info(f"ADMIN_USER: {config.get('ADMIN_USER', 'not set')}")
            logger.info(f"ADMIN_PASSWORD: {'set' if config.get(
                'ADMIN_PASSWORD') else 'not set'}")
            logger.info(f"READ_USER: {config.get('READ_USER', 'not set')}")
            logger.info(f"READ_USER_PASSWORD: {'set' if config.get(
                'READ_USER_PASSWORD') else 'not set'}")
            create_user(
                session,
                config.get('ADMIN_USER'),
                config.get('ADMIN_PASSWORD'),
                tenant_code=config.get('TENANT_CODE'),
                role='admin'
            )  # Admin user with write permissions

            create_user(
                session,
                config.get('READ_USER'),
                config.get('READ_USER_PASSWORD'),
                tenant_code=config.get('TENANT_CODE'),
                role='read_only'
            )  # Regular user with read-only permissions

            # Populate restricted tables
            populate_restricted_tables(session)

            # Initialize default tenant
            initialize_default_tenant(session)

            # Verify the number of records in the database tables
            transcription_count = session.query(Transcription).count()
            user_count = session.query(User).count()
            logger.info(f"Number of records in Transcription table: {
                        transcription_count}")
            logger.info(f"Number of records in User table: {user_count}")

            return True

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error in run_data_import: {str(e)}")
        return False


if __name__ == '__main__':
    run_data_import()
