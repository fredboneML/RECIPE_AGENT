#!/bin/bash

# Script to detect and recover from backend hanging issues
# Usage: ./detect_hanging_backend.sh [container_name] [timeout_seconds]

CONTAINER_NAME=${1:-backend_app}
TIMEOUT_SECONDS=${2:-300}  # Default 5 minutes
LOG_FILE="backend_hanging_detector.log"

echo "$(date): Starting backend hang detection for $CONTAINER_NAME" >> $LOG_FILE

# Function to check if backend is hanging
is_backend_hanging() {
    # Check if there's a "Fetching 5 files" log followed by no logs for more than X seconds
    local last_log_time=$(docker logs --tail 50 $CONTAINER_NAME 2>&1 | grep -m 1 "Fetching 5 files" | awk '{print $1}')
    
    if [ -z "$last_log_time" ]; then
        return 1  # No fetching log found, not hanging
    fi
    
    # Check if there are any logs after the "Fetching 5 files" log
    local logs_after=$(docker logs --since "$last_log_time" $CONTAINER_NAME 2>&1 | grep -v "Fetching 5 files" | wc -l)
    
    if [ "$logs_after" -lt 2 ]; then
        # If there are no logs after fetching for a while, it might be hanging
        echo "$(date): Possible hang detected. Last activity: $last_log_time" >> $LOG_FILE
        return 0  # Hanging detected
    fi
    
    return 1  # Not hanging
}

# Function to restart the container
restart_container() {
    echo "$(date): Restarting $CONTAINER_NAME due to hanging" >> $LOG_FILE
    docker restart $CONTAINER_NAME
    echo "$(date): Container restarted" >> $LOG_FILE
}

# Function to check resource usage
check_resources() {
    echo "$(date): Checking container resources" >> $LOG_FILE
    docker stats $CONTAINER_NAME --no-stream >> $LOG_FILE
}

# Function to collect diagnostic information
collect_diagnostics() {
    echo "$(date): Collecting diagnostic information" >> $LOG_FILE
    
    # Get container logs
    echo "==== RECENT LOGS ====" >> $LOG_FILE
    docker logs --tail 100 $CONTAINER_NAME >> $LOG_FILE 2>&1
    
    # Get network connections
    echo "==== NETWORK CONNECTIONS ====" >> $LOG_FILE
    docker exec $CONTAINER_NAME netstat -tunap 2>/dev/null >> $LOG_FILE || echo "netstat not available" >> $LOG_FILE
    
    # Get process list
    echo "==== PROCESS LIST ====" >> $LOG_FILE
    docker exec $CONTAINER_NAME ps aux 2>/dev/null >> $LOG_FILE || echo "ps not available" >> $LOG_FILE
    
    # Get memory info
    echo "==== MEMORY INFO ====" >> $LOG_FILE
    docker exec $CONTAINER_NAME cat /proc/meminfo 2>/dev/null >> $LOG_FILE || echo "meminfo not available" >> $LOG_FILE
}

# Main monitoring loop
while true; do
    if is_backend_hanging; then
        echo "$(date): Backend appears to be hanging" >> $LOG_FILE
        check_resources
        collect_diagnostics
        restart_container
    fi
    sleep 60  # Check every minute
done
