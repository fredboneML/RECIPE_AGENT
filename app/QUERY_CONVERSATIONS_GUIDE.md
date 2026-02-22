# Guide: Querying User Conversations from Database

This guide shows you how to connect to the PostgreSQL database and download conversations for a specific user.

## Database Connection Details

- **Host**: `localhost` (from your machine) or `database` (from within Docker network)
- **Port**: `5433` (mapped from container port 5432)
- **Database**: Value from `POSTGRES_DB` environment variable
- **User**: Value from `POSTGRES_USER` environment variable
- **Password**: Value from `POSTGRES_PASSWORD` environment variable

## Important: Understanding User IDs

**The `user_id` in the `user_memory` table is the numeric user ID (from the `users` table) converted to a string, NOT the username!**

For example:
- Username: `admin` → user_id in database: `"1"` (if admin is user ID 1)
- Username: `readonly` → user_id in database: `"2"` (if readonly is user ID 2)

**To find conversations by username, use the `find_user_conversations.sh` script instead!**

## Method 1: Using the Provided Scripts (Easiest)

### Option A: Find by Username (Easiest - Recommended!)

This script finds the user ID from the username automatically:

```bash
# Make the script executable (if not already)
chmod +x find_user_conversations.sh

# List all users and their conversation counts
./find_user_conversations.sh

# Query conversations by username and export to CSV
./find_user_conversations.sh admin csv

# Export to JSON
./find_user_conversations.sh readonly json

# Display in table format
./find_user_conversations.sh admin table
```

### Option B: Docker-based script (By numeric user ID)

This script uses the Docker container's environment variables, but requires the numeric user ID:

```bash
# Make the script executable (if not already)
chmod +x query_user_conversations_docker.sh

# First, list available users to find the ID
./query_user_conversations_docker.sh

# Query conversations for a user (using numeric ID like "1", "2", etc.)
./query_user_conversations_docker.sh 1 csv

# Export to JSON
./query_user_conversations_docker.sh 1 json

# Display in table format
./query_user_conversations_docker.sh 1 table
```

### Option B: Direct connection script (Requires credentials)

This script connects directly to the database and requires credentials:

```bash
# First, set your credentials (or create a .env file)
export POSTGRES_USER=your_user
export POSTGRES_PASSWORD=your_password
export POSTGRES_DB=your_database

# Or create a .env file in the app directory with:
# POSTGRES_USER=your_user
# POSTGRES_PASSWORD=your_password
# POSTGRES_DB=your_database

# Make the script executable
chmod +x query_user_conversations.sh

# Query conversations for a user and export to CSV
./query_user_conversations.sh <user_id> csv

# Export to JSON
./query_user_conversations.sh <user_id> json

# Display in table format
./query_user_conversations.sh <user_id> table
```

## Method 2: Direct psql Connection

### Connect from your local machine:

```bash
# Set password as environment variable
export PGPASSWORD="${POSTGRES_PASSWORD}"

# Connect to database
psql -h localhost -p 5433 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"
```

### Once connected, run these SQL queries:

#### List all users with conversations:
```sql
SELECT DISTINCT user_id, COUNT(*) as conversation_count
FROM user_memory
WHERE is_active = TRUE
GROUP BY user_id
ORDER BY conversation_count DESC;
```

#### Get all conversations for a specific user:
```sql
SELECT 
    conversation_id,
    message_order,
    title,
    query,
    response,
    timestamp,
    followup_questions
FROM user_memory
WHERE user_id = 'YOUR_USER_ID' AND is_active = TRUE
ORDER BY conversation_id, message_order;
```

#### Export to CSV:
```sql
\COPY (
    SELECT 
        conversation_id,
        message_order,
        title,
        query,
        response,
        timestamp,
        followup_questions
    FROM user_memory
    WHERE user_id = 'YOUR_USER_ID' AND is_active = TRUE
    ORDER BY conversation_id, message_order
) TO '/tmp/conversations.csv' WITH CSV HEADER;
```

#### Export to JSON:
```sql
SELECT json_agg(
    json_build_object(
        'conversation_id', conversation_id,
        'message_order', message_order,
        'title', title,
        'query', query,
        'response', response,
        'timestamp', timestamp,
        'followup_questions', followup_questions
    ) ORDER BY conversation_id, message_order
)
FROM user_memory
WHERE user_id = 'YOUR_USER_ID' AND is_active = TRUE;
```

## Method 3: Using Docker Exec

### Connect via Docker container:

```bash
# Execute psql inside the postgres container
docker exec -it postgres_db psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"
```

Or run a one-liner query:

```bash
docker exec -it postgres_db psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "
SELECT 
    conversation_id,
    message_order,
    title,
    LEFT(query, 100) as query_preview,
    LEFT(response, 100) as response_preview,
    timestamp
FROM user_memory
WHERE user_id = 'YOUR_USER_ID' AND is_active = TRUE
ORDER BY conversation_id, message_order;
"
```

### Export via Docker:

```bash
# Export to CSV
docker exec -it postgres_db psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "\COPY (
    SELECT conversation_id, message_order, title, query, response, timestamp, followup_questions
    FROM user_memory
    WHERE user_id = 'YOUR_USER_ID' AND is_active = TRUE
    ORDER BY conversation_id, message_order
) TO STDOUT WITH CSV HEADER" > conversations_export.csv
```

## Method 4: Using Python Script

Create a Python script to query and export:

```python
#!/usr/bin/env python3
import os
import csv
import json
from sqlalchemy import create_engine, text
from datetime import datetime

# Database connection
db_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5433/{os.getenv('POSTGRES_DB')}"
engine = create_engine(db_url)

# Query conversations for a user
user_id = "YOUR_USER_ID"

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT 
            conversation_id,
            message_order,
            title,
            query,
            response,
            timestamp,
            followup_questions
        FROM user_memory
        WHERE user_id = :user_id AND is_active = TRUE
        ORDER BY conversation_id, message_order
    """), {"user_id": user_id})
    
    conversations = []
    for row in result:
        conversations.append({
            'conversation_id': row[0],
            'message_order': row[1],
            'title': row[2],
            'query': row[3],
            'response': row[4],
            'timestamp': row[5].isoformat() if row[5] else None,
            'followup_questions': row[6]
        })
    
    # Export to JSON
    with open(f'conversations_{user_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(conversations)} conversations to JSON")
```

## Understanding the user_memory Table Schema

The `user_memory` table has the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `user_id` | VARCHAR | User identifier |
| `conversation_id` | VARCHAR | Conversation identifier |
| `title` | VARCHAR | Conversation title |
| `query` | TEXT | User's question/query |
| `response` | TEXT | System's response |
| `timestamp` | TIMESTAMP | When the message was created |
| `is_active` | BOOLEAN | Whether the message is active |
| `message_order` | INTEGER | Order of message in conversation |
| `expires_at` | TIMESTAMP | When the message expires |
| `followup_questions` | JSONB | Follow-up questions (JSON) |

## Useful Queries

### Get conversation statistics:
```sql
SELECT 
    user_id,
    COUNT(DISTINCT conversation_id) as total_conversations,
    COUNT(*) as total_messages,
    MIN(timestamp) as first_message,
    MAX(timestamp) as last_message
FROM user_memory
WHERE is_active = TRUE
GROUP BY user_id
ORDER BY total_messages DESC;
```

### Get a specific conversation thread:
```sql
SELECT 
    message_order,
    title,
    query,
    response,
    timestamp
FROM user_memory
WHERE conversation_id = 'SPECIFIC_CONVERSATION_ID'
ORDER BY message_order;
```

### Find conversations containing specific keywords:
```sql
SELECT 
    user_id,
    conversation_id,
    message_order,
    query,
    response,
    timestamp
FROM user_memory
WHERE is_active = TRUE
  AND (query ILIKE '%keyword%' OR response ILIKE '%keyword%')
ORDER BY timestamp DESC;
```

## Notes

- Replace `YOUR_USER_ID` with the actual user ID you want to query
- The `is_active = TRUE` filter ensures you only get active conversations
- Conversations are ordered by `conversation_id` and `message_order` to maintain chronological order
- The `followup_questions` field is stored as JSONB and may contain structured data
