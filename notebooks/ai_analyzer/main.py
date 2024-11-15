# ai_analyzer/main.py
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError
import hashlib
from ai_analyzer.database_agent_postgresql import answer_question, get_db_connection
from ai_analyzer.data_import_postgresql import run_data_import
from ai_analyzer.fetch_data_from_api import fetch_data_from_api
from ai_analyzer.make_openai_call_df import make_openai_call_df
import time
import logging
from psycopg2 import connect
from ai_analyzer.config import config, DATABASE_URL, DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def wait_for_db(max_retries=10, delay=10):
    """Wait for database to be ready with better error handling"""
    logger = logging.getLogger(__name__)
    
    for i in range(max_retries):
        try:
            # Try direct PostgreSQL connection first
            conn = connect(
                dbname=config['POSTGRES_DB'],
                user=config['POSTGRES_USER'],
                password=config['POSTGRES_PASSWORD'],
                host=config['DB_HOST'],
                port=config['DB_PORT']
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
            logger.error(f"Database connection attempt {i + 1}/{max_retries} failed: {str(e)}")
            if i < max_retries - 1:  # Don't sleep on the last attempt
                time.sleep(delay)
    return False

# User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default='read_only')

    def has_write_permission(self):
        return self.role in ['admin', 'write']

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://37.97.226.251:3000",  
        "http://192.168.2.132:3000", 
        "http://172.21.0.4:3000", 
        "http://172.21.0.3:3000",
        "http://frontend_app:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Hashing passwords
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Add startup event to handle initial data loading
@app.on_event("startup")
async def startup_event():
    """Startup event handler with proper error handling"""
    try:
        # Wait for database with more retries
        if not wait_for_db(max_retries=10, delay=10):
            logger.error("Failed to connect to database during startup")
            return

        # Run data import
        logger.info("Starting data import...")
        success = run_data_import()
        
        if success:
            logger.info("Data import completed successfully")
        else:
            logger.warning("Data import completed with warnings (no data files found)")
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")


# Health check endpoint with detailed debugging
@app.get("/health")
async def health_check():
    try:
        # Try direct PostgreSQL connection first
        conn = connect(
            dbname=config['POSTGRES_DB'],
            user=config['POSTGRES_USER'],
            password=config['POSTGRES_PASSWORD'],
            host=config['DB_HOST'],
            port=config['DB_PORT']
        )
        conn.close()
        logger.info("PostgreSQL connection successful")
        
        # Then try SQLAlchemy connection with proper text() wrapper
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        logger.info("SQLAlchemy connection successful")
        
        return {
            "status": "healthy",
            "database": {
                "host": config['DB_HOST'],
                "port": config['DB_PORT'],
                "name": config['POSTGRES_DB'],
                "user": config['POSTGRES_USER']
            }
        }
    except Exception as e:
        error_msg = f"Health check failed: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "unhealthy",
            "error": error_msg,
            "database_config": {
                "host": config['DB_HOST'],
                "port": config['DB_PORT'],
                "name": config['POSTGRES_DB'],
                "user": config['POSTGRES_USER']
            }
        }


# API to handle login
@app.post("/api/login")
async def login(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        username = data['username']
        password = data['password']
        
        user = db.query(User).filter(User.username == username).first()
        if user and user.password_hash == hash_password(password):
            return {
                "success": True,
                "role": user.role,
                "username": user.username,
                "permissions": {
                    "canWrite": user.has_write_permission()
                }
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

# Updated API to handle queries
@app.post("/api/query")
async def query(request: Request):
    data = await request.json()
    question = data['query']
    
    try:
        answer = answer_question(question)
        return {"result": answer}
    except Exception as e:
        return {"error": True, "message": str(e)}

# Optional: API to add a new user
@app.post("/api/add_user")
async def add_user(request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_db)):
    # Check if current user has admin privileges
    if not current_user.has_write_permission():
        raise HTTPException(status_code=403, detail="Insufficient permissions")
        
    data = await request.json()
    username = data['username']
    password = data['password']
    role = data.get('role', 'read_only')  # Default to read_only if not specified
    
    # Validate role
    if role not in ['admin', 'write', 'read_only']:
        raise HTTPException(status_code=400, detail="Invalid role specified")
    
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    new_user = User(
        username=username,
        password_hash=hash_password(password),
        role=role
    )
    db.add(new_user)
    db.commit()
    return {"success": True, "message": "User added successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)