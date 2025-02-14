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
    MetaData,
    text,
    Boolean,
    UniqueConstraint,
    Index,
    DDL,
    event  # Added Index import
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import logging
import hashlib
import datetime
from ai_analyzer.config import config, DATA_DIR, DATABASE_URL
from typing import List, Optional
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime, timedelta, timezone

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('data_import_postgresql')

# Base for ORM models
Base = declarative_base()

# Create initial engine with explicit host and port


def get_database_url():
    """Get database URL with correct host and port"""
    return f"postgresql://{config['POSTGRES_USER']}:{config['POSTGRES_PASSWORD']}@database:5432/{config['POSTGRES_DB']}"


# Create the engine with the correct connection URL
engine = create_engine(
    get_database_url(),
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
)


def create_db_engine():
    """Create and verify database engine"""
    try:
        new_engine = create_engine(
            get_database_url(),
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True
        )
        new_engine.connect()
        logger.info(f"Successfully connected to the database at database:5432")
        return new_engine
    except Exception as e:
        logger.error(f"Error creating database engine: {e}")
        raise


# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Session = sessionmaker(bind=engine)


class Transcription(Base):
    """
    Transcription model for existing partitioned table.
    This model is used for querying only, not table creation.
    """
    __tablename__ = 'transcription'
    __table_args__ = {
        'extend_existing': True  # Only use extend_existing
    }

    id = Column(String, primary_key=True)
    transcription_id = Column(String, nullable=False)
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


def check_user_exists(session, username, tenant_code):
    """Check if a user already exists"""
    return session.query(User).filter_by(
        username=username,
        tenant_code=tenant_code
    ).first() is not None


def check_tenant_exists(session, tenant_code):
    """Check if a tenant already exists"""
    return session.query(TenantCode).filter_by(
        tenant_code=tenant_code
    ).first() is not None


def check_restricted_tables_populated(session):
    """Check if restricted tables are already populated"""
    return session.query(RestrictedTable).count() > 0


def create_partition_for_tenant(connection, tenant_code):
    """Create partition and related objects for a tenant"""
    try:
        # First check if partition exists
        check_sql = text("""
            SELECT EXISTS (
                SELECT FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = :partition_name
            );
        """)

        exists = connection.execute(
            check_sql,
            {"partition_name": f"transcription_{tenant_code}"}
        ).scalar()

        if not exists:
            # Create partition
            connection.execute(text(f"""
                CREATE TABLE transcription_{tenant_code}
                PARTITION OF transcription
                FOR VALUES IN ('{tenant_code}');

                CREATE INDEX idx_{tenant_code}_processing_date
                ON transcription_{tenant_code}(processing_date);

                CREATE INDEX idx_{tenant_code}_topic
                ON transcription_{tenant_code}(topic);

                CREATE INDEX idx_{tenant_code}_sentiment
                ON transcription_{tenant_code}(sentiment);
            """))

            # Create role if not exists
            connection.execute(text(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT FROM pg_roles
                        WHERE rolname = 'tenant_{tenant_code}'
                    ) THEN
                        CREATE ROLE tenant_{tenant_code}
                        LOGIN PASSWORD 'secure_{tenant_code}_password';
                    END IF;
                END $$;

                GRANT SELECT ON transcription_{tenant_code}
                TO tenant_{tenant_code};
            """))

            logger.info(f"Created partition for tenant: {tenant_code}")

        return True

    except Exception as e:
        logger.error(f"Error creating partition for tenant {tenant_code}: {e}")
        raise e


def ensure_tenant_partition(engine, tenant_code):
    """Ensure partition exists for tenant"""
    try:
        with engine.begin() as connection:
            return create_partition_for_tenant(connection, tenant_code)

    except Exception as e:
        logger.error(f"Error ensuring partition for tenant {tenant_code}: {e}")
        raise e

# Modified User model with tenant support


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default='read_only')
    tenant_code = Column(String, nullable=False)

    def has_write_permission(self):
        return self.role in ['admin', 'write']

# Function to create user with tenant role


def create_tenant_user(session, username, password, tenant_code, role='read_only'):
    try:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Create tenant role and partition if they don't exist
        session.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'tenant_{tenant_code}') THEN
                CREATE ROLE tenant_{tenant_code} LOGIN PASSWORD 'secure_{tenant_code}_password';
                GRANT SELECT ON transcription_{tenant_code} TO tenant_{tenant_code};
            END IF;
        END
        $$;
        """)

        # Create user
        new_user = User(
            username=username,
            password_hash=hashed_password,
            role=role,
            tenant_code=tenant_code
        )
        session.add(new_user)
        session.commit()

        return True
    except Exception as e:
        session.rollback()
        raise e

# Function to authenticate user and set role


def authenticate_user(session, username, password, tenant_code):
    try:
        user = session.query(User).filter_by(
            username=username,
            tenant_code=tenant_code
        ).first()

        if user and user.password_hash == hashlib.sha256(password.encode()).hexdigest():
            # Set role to tenant role
            session.execute(f"SET ROLE tenant_{tenant_code}")
            return user

        return None
    except Exception as e:
        raise e


class UserMemory(Base):
    """Model for storing conversation history"""
    __tablename__ = 'user_memory'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    conversation_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    message_order = Column(Integer, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    # Using JSON type for followup questions
    followup_questions = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint('conversation_id', 'message_order',
                         name='unique_message_order'),
    )

    def __init__(self, user_id, conversation_id, query, response, title=None, message_order=0, followup_questions=None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.query = query
        self.response = response
        self.title = title or self._generate_title(query)
        self.message_order = message_order
        self.expires_at = datetime.now(timezone.utc) + timedelta(days=30)
        self.followup_questions = followup_questions if followup_questions else []

    def _generate_title(self, query):
        return query[:50] + "..." if len(query) > 50 else query


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
    """Populate the restricted_tables table using existing session"""
    try:
        # Use existing session instead of creating new connection
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

        for table in restricted_tables:
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
    """Initialize the default tenant (tientelecom)"""
    try:
        # Use existing session instead of creating new connection
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

# In main.py and data_import_postgresql.py:
def store_conversation(session, user_id, conversation_id, query, response, message_order=None, followup_questions=None):
    """Store a conversation message with followup questions"""
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
            title=generate_title(query),
            message_order=message_order,
            followup_questions=followup_questions if followup_questions else []
        )
        session.add(memory)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error storing conversation: {e}")
        return False


def generate_title(query: str) -> str:
    """Generate a title from the first query of conversation"""
    return query[:50] + "..." if len(query) > 50 else query


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


# Create all tables


def create_tables(engine):
    """Create all tables in the correct order"""
    try:
        # First ensure we're in the public schema
        with engine.begin() as connection:
            connection.execute(text("SET search_path TO public"))

            # Create a temporary MetaData without the Transcription table
            temp_metadata = MetaData(schema='public')

            # Create all non-transcription tables first
            for table in Base.metadata.sorted_tables:
                if table.name != 'transcription':
                    table.tometadata(temp_metadata)

            # Create tables
            temp_metadata.create_all(bind=engine)

            # Create the partitioned transcription table
            connection.execute(
                text("DROP TABLE IF EXISTS transcription CASCADE;"))

            # Create partitioned table
            partition_sql = """
            CREATE TABLE transcription (
                id VARCHAR NOT NULL,
                transcription_id VARCHAR NOT NULL,
                processing_date TIMESTAMP NOT NULL,
                transcription TEXT NOT NULL,
                summary TEXT,
                topic VARCHAR,
                sentiment VARCHAR,
                call_duration_secs INTEGER,
                tenant_code VARCHAR NOT NULL,
                clid VARCHAR,
                telephone_number VARCHAR,
                call_direction VARCHAR,
                PRIMARY KEY (id, tenant_code),
                CONSTRAINT unique_tenant_transcription 
                    UNIQUE(transcription_id, processing_date, tenant_code)
            ) PARTITION BY LIST (tenant_code);
            """

            connection.execute(text(partition_sql))

            # Create initial partition for default tenant
            initial_tenant_sql = """
            CREATE TABLE transcription_tientelecom 
            PARTITION OF transcription
            FOR VALUES IN ('tientelecom');
            
            CREATE INDEX idx_tientelecom_processing_date 
            ON transcription_tientelecom(processing_date);
            
            CREATE INDEX idx_tientelecom_topic 
            ON transcription_tientelecom(topic);
            
            CREATE INDEX idx_tientelecom_sentiment 
            ON transcription_tientelecom(sentiment);
            """

            connection.execute(text(initial_tenant_sql))

            # Create role for default tenant
            role_sql = """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT FROM pg_roles 
                    WHERE rolname = 'tenant_tientelecom'
                ) THEN
                    CREATE ROLE tenant_tientelecom LOGIN PASSWORD 'secure_tientelecom_password';
                END IF;
            END $$;
            
            GRANT SELECT ON transcription_tientelecom TO tenant_tientelecom;
            """

            connection.execute(text(role_sql))

            # Verify tables were created
            verify_sql = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
            """

            tables = connection.execute(text(verify_sql)).fetchall()
            logger.info(f"Created tables: {[t[0] for t in tables]}")

        return True

    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


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


# Modified function to insert data with tenant partitioning
def insert_transcription_data(df_transcription, session, tenant_code):
    """Insert transcription data with proper tenant isolation and duplicate checking"""
    try:
        if df_transcription.empty:
            logger.warning("No transcription data to import")
            return

        # Ensure partition exists
        ensure_tenant_partition(session.bind, tenant_code)

        logger.info(f"Starting to import {
                    len(df_transcription)} transcription records for tenant {tenant_code}")

        # Set tenant_code for all records
        df_transcription['tenant_code'] = tenant_code

        # Convert processingdate to processing_date if needed
        if 'processingdate' in df_transcription.columns:
            df_transcription = df_transcription.rename(
                columns={'processingdate': 'processing_date'})

        # Convert processing_date to datetime
        df_transcription['processing_date'] = pd.to_datetime(
            df_transcription['processing_date'])

        # First, get existing records for this tenant to avoid duplicates
        existing_records = session.query(
            Transcription.transcription_id,
            Transcription.processing_date
        ).filter(
            Transcription.tenant_code == tenant_code
        ).all()

        # Create a set for efficient lookup
        existing_set = {(str(rec.transcription_id), rec.processing_date)
                        for rec in existing_records}

        # Filter out existing records
        new_records = []
        for _, row in df_transcription.iterrows():
            key = (str(row['transcription_id']), row['processing_date'])
            if key not in existing_set:
                new_records.append(row)

        # Convert to DataFrame
        new_df = pd.DataFrame(new_records) if new_records else pd.DataFrame()

        if new_df.empty:
            logger.info(f"No new records to import for tenant {tenant_code}")
            return True

        logger.info(
            f"Found {len(new_df)} new records to import for tenant {tenant_code}")

        # Process records in batches
        batch_size = 100
        successful_imports = 0

        for batch_start in range(0, len(new_df), batch_size):
            batch_end = min(batch_start + batch_size, len(new_df))
            batch = new_df.iloc[batch_start:batch_end]

            with session.begin_nested():  # Create savepoint
                try:
                    for _, row in batch.iterrows():
                        try:
                            # Prepare values with proper NULL handling
                            values = {
                                'id': str(uuid.uuid4()),
                                'transcription_id': str(row['transcription_id']),
                                'processing_date': row['processing_date'],
                                'transcription': str(row['transcription']),
                                'summary': str(row['summary']) if pd.notna(row.get('summary')) else None,
                                'topic': str(row['topic']) if pd.notna(row.get('topic')) else None,
                                'sentiment': str(row['sentiment']) if pd.notna(row.get('sentiment')) else None,
                                'tenant_code': tenant_code,
                                'call_duration_secs': int(row['call_duration_secs']) if pd.notna(row.get('call_duration_secs')) else None,
                                'clid': str(row.get('clid')) if pd.notna(row.get('clid')) else None,
                                'telephone_number': str(row.get('telephone_number')) if pd.notna(row.get('telephone_number')) else None,
                                'call_direction': str(row.get('call_direction')) if pd.notna(row.get('call_direction')) else None
                            }

                            # Insert with ON CONFLICT DO NOTHING as a safety net
                            insert_stmt = insert(Transcription).values(values)
                            do_nothing_stmt = insert_stmt.on_conflict_do_nothing(
                                constraint='unique_tenant_transcription'
                            )
                            session.execute(do_nothing_stmt)
                            successful_imports += 1

                        except Exception as e:
                            logger.error(f"Error importing record: {e}")
                            logger.error(f"Problematic row: {row}")
                            continue

                    session.commit()
                    logger.info(f"Successfully imported batch of {
                                batch_end - batch_start} records")

                except Exception as batch_error:
                    logger.error(f"Error processing batch: {batch_error}")
                    session.rollback()
                    continue

        logger.info(f"Import completed. Successfully imported {
                    successful_imports} new records for tenant {tenant_code}")
        return True

    except Exception as e:
        logger.error(f"Error in transcription data import process: {e}")
        session.rollback()
        return False

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


# Modified run_create_tables function
def run_create_tables():
    """Create required tables if they don't exist"""
    try:
        engine = create_engine(DATABASE_URL)

        # Create all tables with the new ordering
        create_tables(engine)

        return True

    except Exception as e:
        logger.error(f"Error in run_create_tables: {str(e)}")
        return False

# Initialize the database


def init_db():
    """Initialize the database with proper table creation order"""
    try:
        # Use get_database_url() for consistent connection settings
        connection_url = get_database_url()
        logger.info(f"Connecting to database at database:5432")

        engine = create_engine(
            connection_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True
        )

        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Use single session for all operations
            initialize_default_tenant(session)
            populate_restricted_tables(session)

            create_user(
                session,
                config.get('ADMIN_USER'),
                config.get('ADMIN_PASSWORD'),
                tenant_code='tientelecom',
                role='admin'
            )

            create_user(
                session,
                config.get('READ_USER'),
                config.get('READ_USER_PASSWORD'),
                tenant_code='tientelecom',
                role='read_only'
            )

            logger.info("Database initialized successfully")
            return True

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


def run_data_import():
    """Run the data import process with proper error handling"""
    try:

        # Use get_database_url() for consistent connection settings
        engine = create_engine(
            get_database_url(),
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True
        )
        Session = sessionmaker(bind=engine)
        session = Session()

        logger.info(
            "Created database session with explicit connection parameters")

        try:
            # Create tables if they don't exist
            # init_db()

            # Load data
            df_transcription = load_data_from_csv()

            # Only proceed with import if we have data
            if not df_transcription.empty:
                logger.info(f"Number of records in df_transcription: {
                            len(df_transcription)}")

                # Insert transcription data
                if not df_transcription.empty:
                    insert_transcription_data(
                        df_transcription, session, config.get('TENANT_CODE'))
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
