from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv
import hashlib
from .database_agent_postgresql import answer_question

# Load environment variables
load_dotenv()

# PostgreSQL connection details
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('DB_HOST')  
DB_PORT = os.getenv('DB_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')

# SQLAlchemy setup
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# API to handle login
@app.post("/api/login")
async def login(request: Request, db: Session = Depends(get_db)):
    logger.info("Login attempt received")
    data = await request.json()
    username = data['username']
    password = data['password']
    
    logger.info(f"Login attempt for user: {username}")
    
    user = db.query(User).filter(User.username == username).first()
    if user and user.password_hash == hash_password(password):
        logger.info(f"Successful login for user: {username}")
        return {"success": True}
    else:
        logger.warning(f"Failed login attempt for user: {username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")


# API to handle queries
@app.post("/api/query")
async def query(request: Request):
    data = await request.json()
    question = data['query']
    
    try:
        answer = answer_question(question)
        return {"result": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: API to add a new user
@app.post("/api/add_user")
async def add_user(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    username = data['username']
    password = data['password']
    
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    new_user = User(username=username, password_hash=hash_password(password))
    db.add(new_user)
    db.commit()
    return {"success": True, "message": "User added successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)