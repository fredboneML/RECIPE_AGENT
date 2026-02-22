#!/bin/bash

# Script to query and export user conversations using Docker exec
# This method uses the database credentials from the Docker container
# Usage: ./query_user_conversations_docker.sh <user_id> [output_format]

USER_ID=${1:-""}
OUTPUT_FORMAT=${2:-"csv"}  # csv, json, or table

# Check if Docker container is running
if ! docker ps | grep -q postgres_db; then
    echo "Error: postgres_db container is not running!"
    echo "Start it with: docker-compose up -d database"
    exit 1
fi

if [ -z "$USER_ID" ]; then
    echo "Usage: $0 <user_id> [output_format]"
    echo ""
    echo "Note: user_id is the numeric ID from the users table, not the username!"
    echo "      It's stored as a string in user_memory.user_id (e.g., '1', '2', '3')"
    echo ""
    echo "Examples:"
    echo "  $0 '1' csv"
    echo "  $0 '2' json"
    echo "  $0 '3' table"
    echo ""
    echo "Available users with conversations:"
    DB_USER=$(docker exec postgres_db printenv POSTGRES_USER 2>/dev/null || echo "${POSTGRES_USER:-postgres}")
    DB_NAME=$(docker exec postgres_db printenv POSTGRES_DB 2>/dev/null || echo "${POSTGRES_DB:-recipe_agent}")
    
    echo ""
    echo "User IDs with conversations (from user_memory table):"
    docker exec -i postgres_db psql -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT DISTINCT 
            um.user_id,
            u.username,
            COUNT(*) as conversation_count
        FROM user_memory um
        LEFT JOIN users u ON um.user_id = CAST(u.id AS VARCHAR)
        WHERE um.is_active = TRUE
        GROUP BY um.user_id, u.username
        ORDER BY um.user_id;
    " 2>/dev/null || {
        echo "Could not get user list. Showing just user_id values:"
        docker exec -i postgres_db psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT DISTINCT user_id FROM user_memory WHERE is_active = TRUE ORDER BY user_id;"
    }
    echo ""
    echo "All users in the system:"
    docker exec -i postgres_db psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT id, username, role FROM users ORDER BY id;" 2>/dev/null || echo "Could not retrieve users table"
    exit 1
fi

# Try to get credentials from container environment, fallback to environment variables
DB_USER=$(docker exec postgres_db printenv POSTGRES_USER 2>/dev/null || echo "${POSTGRES_USER:-postgres}")
DB_NAME=$(docker exec postgres_db printenv POSTGRES_DB 2>/dev/null || echo "${POSTGRES_DB:-recipe_agent}")

echo "Querying conversations for user: $USER_ID"
echo "Using database: $DB_NAME (user: $DB_USER)"
echo "=========================================="
echo ""

case "$OUTPUT_FORMAT" in
    csv)
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_FILE="conversations_${USER_ID}_${TIMESTAMP}.csv"
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
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_FILE="conversations_${USER_ID}_${TIMESTAMP}.json"
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
