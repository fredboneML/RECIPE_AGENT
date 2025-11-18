#!/usr/bin/env python3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
import time
import logging
import hashlib
from ai_analyzer.config import config, DATA_DIR

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
                role VARCHAR NOT NULL DEFAULT 'read_only'
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
                question TEXT NOT NULL,
                sql TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '7 days'),
                hash_key VARCHAR(64) NOT NULL,
                execution_time_ms INTEGER,
                result_count INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_query_cache_expires ON query_cache(expires_at);
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
            INSERT INTO users (username, password_hash, role)
            VALUES (:username, :password_hash, 'admin')
            ON CONFLICT (username) DO NOTHING
        """), {
            "username": config['ADMIN_USER'],
            "password_hash": hash_password(config['ADMIN_PASSWORD'])
        })

    if config.get('READ_USER') and config.get('READ_USER_PASSWORD'):
        session.execute(text("""
            INSERT INTO users (username, password_hash, role)
            VALUES (:username, :password_hash, 'read_only')
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

        session.commit()
        logger.info(
            "Successfully set up permanent triggers for user management")
        return True

    except Exception as e:
        logger.error(f"Error setting up permanent triggers: {e}")
        session.rollback()
        raise


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
            'READ_USER_PASSWORD': os.getenv('READ_USER_PASSWORD')
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

            # Populate restricted tables
            populate_restricted_tables(session)

            # Create users
            create_users(session, db_config)

            # Set up permanent triggers
            setup_permanent_triggers(session)

            logger.info("Database initialization completed successfully!")

        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise


if __name__ == "__main__":
    main()
