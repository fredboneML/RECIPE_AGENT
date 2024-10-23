from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError
import os
from dotenv import load_dotenv
import hashlib
from ai_analyzer.database_agent_postgresql import answer_question, get_db_connection
from ai_analyzer.data_import_postgresql import run_data_import
from ai_analyzer.fetch_data_from_api import fetch_data_from_api
from ai_analyzer.make_openai_call_df import make_openai_call_df
from ai_analyzer import config
from datetime import datetime
import pandas as pd
from ai_analyzer import config

data_dir =config.DATA_DIR

# Load environment variables
load_dotenv()

# PostgreSQL connection details
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('DB_HOST')  
DB_PORT = os.getenv('DB_PORT')
POSTGRES_DB = os.getenv('POSTGRES_DB')

URL = os.getenv('URL')
API_KEY = os.getenv('API_KEY')


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

### Pipeline ###
# 1. Fetch data
fetch_data_from_api(url=URL, api_key=API_KEY, 
                    last_id=config.LAST_ID, limit=config.LIMIT)

# 2.  Generating sentiment and topic
df__file_name = [file for file in os.listdir(data_dir) if file.startswith('df__')]
df__file_name = sorted(df__file_name,
                       key=lambda x: datetime.strptime(x.split('__')[1].split('.csv')[0], '%Y-%m-%d'),
                       reverse=True)
print(df__file_name)
df__file_name = df__file_name[0]
df = pd.read_csv(f'{data_dir}/{df__file_name}')

df = make_openai_call_df(df=df, model="gpt-4o-mini-2024-07-18")

# 3. Load data to db
run_data_import()

###############

# Test endpoint
@app.get("/api/test")
async def test():
    print("Test endpoint hit!")
    return {"message": "Backend is reachable"}


# API to handle login
@app.post("/api/login")
async def login(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    username = data['username']
    password = data['password']
    
    user = db.query(User).filter(User.username == username).first()
    if user and user.password_hash == hash_password(password):
        return {"success": True}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

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