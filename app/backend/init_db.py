#!/usr/bin/env python3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
import time
import logging
import hashlib
import pandas as pd
from datetime import datetime
from ai_analyzer.config import config, DATA_DIR
from ai_analyzer.make_openai_call_df import make_openai_call_df
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_init')


def wait_for_db(engine, max_retries=30, delay=2):
    """Wait for database to be ready"""
    for attempt in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.info("Database is ready!")
                return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.info(
                    f"Database not ready (attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(
                    f"Failed to connect to database after {max_retries} attempts")
                raise


def create_tables(engine):
    """Create all required tables"""
    with engine.begin() as connection:
        # Set search path
        connection.execute(text("SET search_path TO public"))

        # Create users table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR NOT NULL UNIQUE,
                password_hash VARCHAR NOT NULL,
                role VARCHAR NOT NULL DEFAULT 'read_only',
                tenant_code VARCHAR NOT NULL
            )
        """))

        # Create tenant_codes table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS tenant_codes (
                tenant_code VARCHAR PRIMARY KEY,
                tenant_code_alias VARCHAR NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # Create user_memory table
        # Create user_memory table with followup_questions column
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS user_memory (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                conversation_id VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                message_order INTEGER NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                followup_questions JSONB,
                CONSTRAINT unique_message_order UNIQUE(conversation_id, message_order)
            )
        """))

        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_user_conversations 
            ON user_memory(user_id, conversation_id);
            
            CREATE INDEX IF NOT EXISTS idx_conversation_order
            ON user_memory(conversation_id, message_order);
        """))

        # Create restricted_tables table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS restricted_tables (
                table_name VARCHAR PRIMARY KEY,
                reason VARCHAR NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                added_by VARCHAR NOT NULL
            )
        """))

        # Create query_cache table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id SERIAL PRIMARY KEY,
                query_hash VARCHAR UNIQUE NOT NULL,
                query_text VARCHAR NOT NULL,
                response VARCHAR NOT NULL,
                last_record_count INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        # Create query_performance table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS query_performance (
                id SERIAL PRIMARY KEY,
                query_text VARCHAR NOT NULL,
                was_answered BOOLEAN NOT NULL,
                error_message VARCHAR,
                response_time INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                topic_category VARCHAR,
                tokens_used INTEGER
            )
        """))

        # Create partitioned transcription table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS transcription (
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
            ) PARTITION BY LIST (tenant_code)
        """))


def initialize_default_tenant(session):
    """Initialize the default tenant"""
    session.execute(text("""
        INSERT INTO tenant_codes (tenant_code, tenant_code_alias)
        VALUES ('tientelecom', 'tientelecom')
        ON CONFLICT (tenant_code) DO NOTHING
    """))

    # Create partition for default tenant
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS transcription_tientelecom 
        PARTITION OF transcription
        FOR VALUES IN ('tientelecom')
    """))

    # Create indexes
    session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_tientelecom_processing_date 
        ON transcription_tientelecom(processing_date);
        
        CREATE INDEX IF NOT EXISTS idx_tientelecom_topic 
        ON transcription_tientelecom(topic);
        
        CREATE INDEX IF NOT EXISTS idx_tientelecom_sentiment 
        ON transcription_tientelecom(sentiment)
    """))

    session.commit()


"get_generated_password", "generated_passwords"


def populate_restricted_tables(session):
    """Populate restricted tables"""
    restricted_tables = [
        ("users", "Contains sensitive user authentication data"),
        ("query_cache", "Internal system table for query optimization"),
        ("query_performance", "Internal system table for performance monitoring"),
        ("user_memory", "Contains user conversation history and personal data"),
        ("get_generated_password", "Temporally User passwords"),
        ("generated_passwords", "User passwords")
    ]

    for table, reason in restricted_tables:
        session.execute(text("""
            INSERT INTO restricted_tables (table_name, reason, added_by)
            VALUES (:table, :reason, 'system')
            ON CONFLICT (table_name) DO NOTHING
        """), {"table": table, "reason": reason})

    session.commit()


def create_users(session, config):
    """Create admin and read-only users"""
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()

    if config.get('ADMIN_USER') and config.get('ADMIN_PASSWORD'):
        session.execute(text("""
            INSERT INTO users (username, password_hash, role, tenant_code)
            VALUES (:username, :password_hash, 'admin', 'tientelecom')
            ON CONFLICT (username) DO NOTHING
        """), {
            "username": config['ADMIN_USER'],
            "password_hash": hash_password(config['ADMIN_PASSWORD'])
        })

    if config.get('READ_USER') and config.get('READ_USER_PASSWORD'):
        session.execute(text("""
            INSERT INTO users (username, password_hash, role, tenant_code)
            VALUES (:username, :password_hash, 'read_only', 'tientelecom')
            ON CONFLICT (username) DO NOTHING
        """), {
            "username": config['READ_USER'],
            "password_hash": hash_password(config['READ_USER_PASSWORD'])
        })

    session.commit()


def setup_permanent_triggers(session):
    """Set up permanent triggers for both user management and tenant management"""
    try:
        # 1. Set up user management triggers
        session.execute(text("""
            -- Create a table to temporarily store generated passwords
            CREATE TABLE IF NOT EXISTS generated_passwords (
                username VARCHAR PRIMARY KEY,
                password VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Updated function to generate random password
            CREATE OR REPLACE FUNCTION generate_default_password()
            RETURNS VARCHAR AS $$
            BEGIN
                RETURN substr(replace(gen_random_uuid()::text, '-', ''), 1, 12);
            END;
            $$ LANGUAGE plpgsql;

            -- Updated function to hash password
            CREATE OR REPLACE FUNCTION hash_password(password VARCHAR)
            RETURNS VARCHAR AS $$
            BEGIN
                RETURN encode(sha256(password::bytea), 'hex');
            END;
            $$ LANGUAGE plpgsql;

            -- Updated user management trigger function
            CREATE OR REPLACE FUNCTION manage_user_fields()
            RETURNS TRIGGER AS $$
            DECLARE
                generated_pwd VARCHAR;
            BEGIN
                -- Set ID if not provided
                IF NEW.id IS NULL THEN
                    NEW.id = nextval('users_id_seq');
                END IF;
                
                -- Set role if not provided
                IF NEW.role IS NULL THEN
                    NEW.role = 'read_only';
                END IF;
                
                -- Generate and hash password if not provided
                IF NEW.password_hash IS NULL THEN
                    -- Generate new password
                    generated_pwd := generate_default_password();
                    
                    -- Store the generated password temporarily
                    INSERT INTO generated_passwords (username, password)
                    VALUES (NEW.username, generated_pwd)
                    ON CONFLICT (username) 
                    DO UPDATE SET password = EXCLUDED.password, created_at = CURRENT_TIMESTAMP;
                    
                    -- Hash the password for storage
                    NEW.password_hash = hash_password(generated_pwd);
                ELSE
                    NEW.password_hash = hash_password(NEW.password_hash);
                END IF;
                
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;

            -- Recreate the trigger
            DROP TRIGGER IF EXISTS user_fields_trigger ON users;
            CREATE TRIGGER user_fields_trigger
                BEFORE INSERT ON users
                FOR EACH ROW
                EXECUTE FUNCTION manage_user_fields();

            -- Function to get the generated password
            CREATE OR REPLACE FUNCTION get_generated_password(p_username VARCHAR)
            RETURNS TABLE (password VARCHAR, created_at TIMESTAMP) AS $$
            BEGIN
                RETURN QUERY
                SELECT gp.password, gp.created_at
                FROM generated_passwords gp
                WHERE gp.username = p_username;
                
                -- Delete the password after retrieving it
                DELETE FROM generated_passwords WHERE username = p_username;
            END;
            $$ LANGUAGE plpgsql;
         """))

        # 2. Set up tenant management triggers - Fixed version
        session.execute(text("""
            CREATE OR REPLACE FUNCTION create_tenant_role(tenant_code text)
            RETURNS void AS $create_role$
            BEGIN
                IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'tenant_' || tenant_code) THEN
                    EXECUTE format('CREATE ROLE tenant_%I LOGIN PASSWORD %L',
                        tenant_code,
                        'secure_' || tenant_code || '_password');
                END IF;
            END;
            $create_role$ LANGUAGE plpgsql;

            CREATE OR REPLACE FUNCTION tenant_management_trigger()
            RETURNS TRIGGER AS $tenant_trigger$
            BEGIN
                -- Create partition for the new tenant
                EXECUTE format(
                    'CREATE TABLE IF NOT EXISTS transcription_%I PARTITION OF transcription FOR VALUES IN (%L)',
                    NEW.tenant_code, NEW.tenant_code
                );

                -- Create indexes for the new partition
                EXECUTE format(
                    'CREATE INDEX IF NOT EXISTS idx_%I_processing_date ON transcription_%I(processing_date)',
                    NEW.tenant_code, NEW.tenant_code
                );
                
                EXECUTE format(
                    'CREATE INDEX IF NOT EXISTS idx_%I_topic ON transcription_%I(topic)',
                    NEW.tenant_code, NEW.tenant_code
                );
                
                EXECUTE format(
                    'CREATE INDEX IF NOT EXISTS idx_%I_sentiment ON transcription_%I(sentiment)',
                    NEW.tenant_code, NEW.tenant_code
                );

                -- Create role for the new tenant
                PERFORM create_tenant_role(NEW.tenant_code);

                -- Grant permissions
                EXECUTE format(
                    'GRANT SELECT ON transcription_%I TO tenant_%I',
                    NEW.tenant_code, NEW.tenant_code
                );

                RETURN NEW;
            END;
            $tenant_trigger$ LANGUAGE plpgsql;

            -- Create tenant management trigger
            DROP TRIGGER IF EXISTS tenant_management_trigger ON tenant_codes;
            CREATE TRIGGER tenant_management_trigger
                AFTER INSERT ON tenant_codes
                FOR EACH ROW
                EXECUTE FUNCTION tenant_management_trigger()
        """))

        session.commit()
        logger.info(
            "Successfully set up permanent triggers for user and tenant management")
        return True

    except Exception as e:
        logger.error(f"Error setting up permanent triggers: {e}")
        session.rollback()
        raise


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


def insert_transcription_data(df_transcription, session, tenant_code):
    """Insert transcription data with proper tenant isolation"""
    try:
        if df_transcription.empty:
            logger.warning("No transcription data to import")
            return True

        logger.info(
            f"Starting to import {len(df_transcription)} transcription records for tenant {tenant_code}")

        # Set tenant_code for all records
        df_transcription['tenant_code'] = tenant_code

        # Convert processingdate to processing_date if needed
        if 'processingdate' in df_transcription.columns:
            df_transcription = df_transcription.rename(
                columns={'processingdate': 'processing_date'})

        # Convert processing_date to datetime
        df_transcription['processing_date'] = pd.to_datetime(
            df_transcription['processing_date'])

        # Process records in batches
        batch_size = 100
        successful_imports = 0

        for batch_start in range(0, len(df_transcription), batch_size):
            batch_end = min(batch_start + batch_size, len(df_transcription))
            batch = df_transcription.iloc[batch_start:batch_end]

            with session.begin_nested():  # Create savepoint
                try:
                    for _, row in batch.iterrows():
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
                        session.execute(stmt, values)
                        successful_imports += 1

                    session.commit()
                    logger.info(
                        f"Successfully imported batch of {batch_end - batch_start} records")

                except Exception as batch_error:
                    logger.error(f"Error processing batch: {batch_error}")
                    session.rollback()
                    continue

        logger.info(
            f"Import completed. Successfully imported {successful_imports} new records for tenant {tenant_code}")
        return True

    except Exception as e:
        logger.error(f"Error in transcription data import process: {e}")
        session.rollback()
        return False


def create_tenant_partition(session, tenant_code):
    """Create partition and role for a specific tenant"""
    try:
        # Create partition
        session.execute(text(f"""
            CREATE TABLE IF NOT EXISTS transcription_{tenant_code} 
            PARTITION OF transcription
            FOR VALUES IN ('{tenant_code}')
        """))

        # Create indexes
        session.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{tenant_code}_processing_date 
            ON transcription_{tenant_code}(processing_date);
            
            CREATE INDEX IF NOT EXISTS idx_{tenant_code}_topic 
            ON transcription_{tenant_code}(topic);
            
            CREATE INDEX IF NOT EXISTS idx_{tenant_code}_sentiment 
            ON transcription_{tenant_code}(sentiment)
        """))

        # Create role and grant permissions
        session.execute(text(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT FROM pg_roles 
                    WHERE rolname = 'tenant_{tenant_code}'
                ) THEN
                    CREATE ROLE tenant_{tenant_code} LOGIN PASSWORD 'secure_{tenant_code}_password';
                END IF;
            END $$;
            
            GRANT SELECT ON transcription_{tenant_code} TO tenant_{tenant_code}
        """))

        session.commit()
        logger.info(
            f"Successfully created partition for tenant: {tenant_code}")
        return True

    except Exception as e:
        logger.error(f"Error creating partition for tenant {tenant_code}: {e}")
        session.rollback()
        return False


def initialize_tenants_from_data(session, df_transcription):
    """Initialize tenants based on unique tenant codes in the data"""
    try:
        if df_transcription.empty:
            logger.warning("No data provided for tenant initialization")
            return True

        # Get unique tenant codes from the data
        unique_tenants = df_transcription['tenant_code'].unique()
        logger.info(f"Found unique tenant codes in data: {unique_tenants}")

        for tenant_code in unique_tenants:
            # Add tenant to tenant_codes table
            session.execute(text("""
                INSERT INTO tenant_codes (tenant_code, tenant_code_alias)
                VALUES (:tenant_code, :tenant_code)
                ON CONFLICT (tenant_code) DO NOTHING
            """), {"tenant_code": tenant_code})

            # Create partition for the tenant
            create_tenant_partition(session, tenant_code)

        session.commit()
        return True

    except Exception as e:
        logger.error(f"Error initializing tenants from data: {e}")
        session.rollback()
        return False


def import_initial_data(session):
    """Import initial data with multi-tenant support"""
    try:
        # Load data
        df_transcription = load_data_from_csv()

        if not df_transcription.empty:
            logger.info(
                f"Number of records in df_transcription: {len(df_transcription)}")

            # Initialize tenants from the data
            initialize_tenants_from_data(session, df_transcription)

            # Group data by tenant_code and insert into respective partitions
            for tenant_code, tenant_data in df_transcription.groupby('tenant_code'):
                logger.info(
                    f"Importing {len(tenant_data)} records for tenant: {tenant_code}")
                insert_transcription_data(tenant_data, session, tenant_code)
        else:
            logger.info("No data to import.")

        return True

    except Exception as e:
        logger.error(f"Error importing initial data: {e}")
        return False


def main():
    """Main function to initialize the database"""
    try:
        # Get database configuration from environment
        db_config = {
            'POSTGRES_USER': os.getenv('POSTGRES_USER'),
            'POSTGRES_PASSWORD': os.getenv('POSTGRES_PASSWORD'),
            'POSTGRES_DB': os.getenv('POSTGRES_DB'),
            'DB_HOST': os.getenv('DB_HOST', 'database'),
            'DB_PORT': os.getenv('DB_PORT', '5432'),
            'ADMIN_USER': os.getenv('ADMIN_USER'),
            'ADMIN_PASSWORD': os.getenv('ADMIN_PASSWORD'),
            'READ_USER': os.getenv('READ_USER'),
            'READ_USER_PASSWORD': os.getenv('READ_USER_PASSWORD'),
            'TENANT_CODE': os.getenv('TENANT_CODE', 'tientelecom'),
        }

        # Create database URL
        db_url = f"postgresql://{db_config['POSTGRES_USER']}:{db_config['POSTGRES_PASSWORD']}@{db_config['DB_HOST']}:{db_config['DB_PORT']}/{db_config['POSTGRES_DB']}"

        # Create engine
        engine = create_engine(db_url)

        # Wait for database to be ready
        wait_for_db(engine)

        # Create session factory
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Create all tables
            create_tables(engine)

            # Initialize default tenant
            initialize_default_tenant(session)

            # Populate restricted tables
            populate_restricted_tables(session)

            # Create users
            create_users(session, db_config)

            # Set up permanent triggers
            setup_permanent_triggers(session)

            # Import initial data with multi-tenant support
            import_initial_data(session)

            logger.info("Database initialization completed successfully!")

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise


if __name__ == "__main__":
    main()
