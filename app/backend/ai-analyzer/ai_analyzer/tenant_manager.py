from typing import Optional
import hashlib
import uuid
import logging
from sqlalchemy import text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('tenant_manager')


def setup_new_tenant_partition(engine, tenant_code: str) -> bool:
    """
    Set up partition, indexes, and role for a new tenant.
    Call this function after a new tenant is added to tenant_codes table.
    """
    try:
        with engine.begin() as connection:
            # Check if partition exists
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
                """))

                # Create indexes
                connection.execute(text(f"""
                    CREATE INDEX idx_{tenant_code}_processing_date
                    ON transcription_{tenant_code}(processing_date);

                    CREATE INDEX idx_{tenant_code}_topic
                    ON transcription_{tenant_code}(topic);

                    CREATE INDEX idx_{tenant_code}_sentiment
                    ON transcription_{tenant_code}(sentiment);
                """))

                # Create role
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

                logger.info(
                    f"Created partition, indexes and role for tenant: {tenant_code}")
                return True

            return True

    except Exception as e:
        logger.error(f"Error setting up tenant {tenant_code}: {e}")
        return False


def setup_user_table_triggers(engine):
    """
    Set up triggers on the users table for automatic field generation.
    Run this once during database initialization.
    """
    try:
        with engine.begin() as connection:
            # Create function to generate random password
            connection.execute(text("""
                CREATE OR REPLACE FUNCTION generate_default_password()
                RETURNS VARCHAR AS $func$
                BEGIN
                    RETURN encode(gen_random_bytes(12), 'hex');
                END;
                $func$ LANGUAGE plpgsql;
            """))

            # Create function to hash password
            connection.execute(text("""
                CREATE OR REPLACE FUNCTION hash_password(password VARCHAR)
                RETURNS VARCHAR AS $func$
                BEGIN
                    RETURN encode(sha256(password::bytea), 'hex');
                END;
                $func$ LANGUAGE plpgsql;
            """))

            # Create trigger function for user management
            connection.execute(text("""
                CREATE OR REPLACE FUNCTION manage_user_fields()
                RETURNS TRIGGER AS $func$
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
                        NEW.password_hash = generate_default_password();
                        RAISE NOTICE 'Generated password for user %: %', 
                            NEW.username, NEW.password_hash;
                        NEW.password_hash = hash_password(NEW.password_hash);
                    ELSE
                        NEW.password_hash = hash_password(NEW.password_hash);
                    END IF;
                    
                    RETURN NEW;
                END;
                $func$ LANGUAGE plpgsql;
            """))

            # Create trigger
            connection.execute(text("""
                DROP TRIGGER IF EXISTS user_fields_trigger ON users;
                CREATE TRIGGER user_fields_trigger
                    BEFORE INSERT ON users
                    FOR EACH ROW
                    EXECUTE FUNCTION manage_user_fields();
            """))

            # Create trigger function for tenant partition management
            connection.execute(text("""
                CREATE OR REPLACE FUNCTION manage_tenant_partition()
                RETURNS TRIGGER AS $func$
                DECLARE
                    role_name text;
                    role_password text;
                BEGIN
                    -- Set up role name and password
                    role_name := 'tenant_' || NEW.tenant_code;
                    role_password := 'secure_' || NEW.tenant_code || '_password';
                    
                    -- Create partition
                    EXECUTE format(
                        'CREATE TABLE IF NOT EXISTS transcription_%I PARTITION OF transcription FOR VALUES IN (%L)',
                        NEW.tenant_code, NEW.tenant_code
                    );
                    
                    -- Create indexes
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
                    
                    -- Create role if it doesn't exist
                    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = role_name) THEN
                        EXECUTE format(
                            'CREATE ROLE %I LOGIN PASSWORD %L',
                            role_name, role_password
                        );
                    END IF;
                    
                    -- Grant permissions
                    EXECUTE format(
                        'GRANT SELECT ON transcription_%I TO %I',
                        NEW.tenant_code, role_name
                    );
                    
                    RETURN NEW;
                END;
                $func$ LANGUAGE plpgsql;
            """))

            # Create trigger for tenant_codes table
            connection.execute(text("""
                DROP TRIGGER IF EXISTS tenant_partition_trigger ON tenant_codes;
                CREATE TRIGGER tenant_partition_trigger
                    AFTER INSERT ON tenant_codes
                    FOR EACH ROW
                    EXECUTE FUNCTION manage_tenant_partition();
            """))

            logger.info("Successfully set up database triggers")
            return True

    except Exception as e:
        logger.error(f"Error setting up triggers: {e}")
        return False
