#!/bin/bash

# Script to query and export user conversations from the database
# Usage: ./query_user_conversations.sh <user_id> [output_format]

USER_ID=${1:-""}
OUTPUT_FORMAT=${2:-"csv"}  # csv, json, or table

# Try to load .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Database connection details
DB_HOST="localhost"
DB_PORT="5433"
DB_USER="${POSTGRES_USER}"
DB_PASSWORD="${POSTGRES_PASSWORD}"
DB_NAME="${POSTGRES_DB}"

# Check if credentials are set
if [ -z "$DB_USER" ] || [ -z "$DB_PASSWORD" ] || [ -z "$DB_NAME" ]; then
    echo "Error: Database credentials not found!"
    echo ""
    echo "Please set the following environment variables:"
    echo "  - POSTGRES_USER"
    echo "  - POSTGRES_PASSWORD"
    echo "  - POSTGRES_DB"
    echo ""
    echo "You can either:"
    echo "  1. Create a .env file in the app directory with these variables"
    echo "  2. Export them in your shell:"
    echo "     export POSTGRES_USER=your_user"
    echo "     export POSTGRES_PASSWORD=your_password"
    echo "     export POSTGRES_DB=your_database"
    echo ""
    echo "Or use Docker exec method instead:"
    echo "  docker exec -it postgres_db psql -U \${POSTGRES_USER} -d \${POSTGRES_DB} -c \"SELECT ...\""
    exit 1
fi

if [ -z "$USER_ID" ]; then
    echo "Usage: $0 <user_id> [output_format]"
    echo ""
    echo "Examples:"
    echo "  $0 'user123' csv"
    echo "  $0 'user123' json"
    echo "  $0 'user123' table"
    echo ""
    echo "Available users:"
    if [ -n "$DB_USER" ] && [ -n "$DB_PASSWORD" ] && [ -n "$DB_NAME" ]; then
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT DISTINCT user_id FROM user_memory ORDER BY user_id;" 2>/dev/null || echo "Could not connect to database. Try using Docker method instead."
    else
        echo "Database credentials not set. Use Docker method:"
        echo "  docker exec -it postgres_db psql -U \${POSTGRES_USER} -d \${POSTGRES_DB} -c \"SELECT DISTINCT user_id FROM user_memory ORDER BY user_id;\""
    fi
    exit 1
fi

# Export PGPASSWORD for psql
export PGPASSWORD="$DB_PASSWORD"

echo "Querying conversations for user: $USER_ID"
echo "=========================================="
echo ""

case "$OUTPUT_FORMAT" in
    csv)
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_FILE="conversations_${USER_ID}_${TIMESTAMP}.csv"
        echo "Exporting to CSV format..."
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "\COPY (
            SELECT 
                conversation_id,
                message_order,
                title,
                query,
                response,
                timestamp,
                followup_questions
            FROM user_memory
            WHERE user_id = '$USER_ID' AND is_active = TRUE
            ORDER BY conversation_id, message_order
        ) TO STDOUT WITH CSV HEADER" > "$OUTPUT_FILE"
        echo "Saved to: $OUTPUT_FILE"
        ;;
    json)
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_FILE="conversations_${USER_ID}_${TIMESTAMP}.json"
        echo "Exporting to JSON format..."
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -F"," -c "
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
            WHERE user_id = '$USER_ID' AND is_active = TRUE
        " | jq '.' > "$OUTPUT_FILE"
        echo "Saved to: $OUTPUT_FILE"
        ;;
    table)
        echo "Displaying conversations in table format:"
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
            SELECT 
                conversation_id,
                message_order,
                title,
                LEFT(query, 50) as query_preview,
                LEFT(response, 50) as response_preview,
                timestamp
            FROM user_memory
            WHERE user_id = '$USER_ID' AND is_active = TRUE
            ORDER BY conversation_id, message_order
            LIMIT 50;
        "
        echo ""
        echo "To see full query/response, use CSV or JSON export format."
        ;;
    *)
        echo "Invalid output format. Use: csv, json, or table"
        exit 1
        ;;
esac
