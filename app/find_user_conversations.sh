#!/bin/bash

# Helper script to find user conversations by username
# Usage: ./find_user_conversations.sh <username> [output_format]

USERNAME=${1:-""}
OUTPUT_FORMAT=${2:-"csv"}

# Try to load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Get database credentials
DB_USER=$(docker exec postgres_db printenv POSTGRES_USER 2>/dev/null || echo "${POSTGRES_USER:-postgres}")
DB_NAME=$(docker exec postgres_db printenv POSTGRES_DB 2>/dev/null || echo "${POSTGRES_DB:-recipe_agent}")

if [ -z "$USERNAME" ]; then
    echo "Usage: $0 <username> [output_format]"
    echo ""
    echo "Examples:"
    echo "  $0 'admin' csv"
    echo "  $0 'readonly' json"
    echo ""
    echo "Available users:"
    docker exec -i postgres_db psql -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            u.id,
            u.username,
            u.role,
            COUNT(DISTINCT um.conversation_id) as conversation_count,
            COUNT(um.id) as total_messages
        FROM users u
        LEFT JOIN user_memory um ON um.user_id = CAST(u.id AS VARCHAR) AND um.is_active = TRUE
        GROUP BY u.id, u.username, u.role
        ORDER BY u.id;
    "
    exit 1
fi

# Find user ID from username
USER_ID=$(docker exec -i postgres_db psql -U "$DB_USER" -d "$DB_NAME" -t -A -c "
    SELECT CAST(id AS VARCHAR) 
    FROM users 
    WHERE username = '$USERNAME';
" 2>/dev/null | tr -d '[:space:]')

if [ -z "$USER_ID" ]; then
    echo "Error: User '$USERNAME' not found!"
    echo ""
    echo "Available users:"
    docker exec -i postgres_db psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT id, username, role FROM users ORDER BY id;"
    exit 1
fi

echo "Found user: $USERNAME (ID: $USER_ID)"
echo "=========================================="
echo ""

# Now use the user_id to query conversations
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

case "$OUTPUT_FORMAT" in
    csv)
        OUTPUT_FILE="conversations_${USERNAME}_${TIMESTAMP}.csv"
        echo "Exporting to CSV format..."
        docker exec -i postgres_db psql -U "$DB_USER" -d "$DB_NAME" -c "\COPY (
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
        OUTPUT_FILE="conversations_${USERNAME}_${TIMESTAMP}.json"
        echo "Exporting to JSON format..."
        docker exec -i postgres_db psql -U "$DB_USER" -d "$DB_NAME" -t -A -F"," -c "
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
        docker exec -it postgres_db psql -U "$DB_USER" -d "$DB_NAME" -c "
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
