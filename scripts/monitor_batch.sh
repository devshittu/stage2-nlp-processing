#!/bin/bash
#
# Batch Processing Monitor Script
# Usage: ./scripts/monitor_batch.sh [job_id]
#
# If no job_id provided, uses the last submitted job
#
# Configuration:
#   BATCH_SAVE_RESULTS=false       - Don't save results to files
#   BATCH_DISPLAY_RESULTS=false    - Don't display results in terminal
#   BATCH_OUTPUT_DIR=/custom/path  - Custom output directory
#   BATCH_POLL_INTERVAL=60         - Custom polling interval (seconds)
#
# Examples:
#   ./scripts/monitor_batch.sh                          # Normal mode (save + display)
#   BATCH_SAVE_RESULTS=false ./scripts/monitor_batch.sh # Display only
#   BATCH_DISPLAY_RESULTS=false ./scripts/monitor_batch.sh # Save only
#

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load configuration
source "$SCRIPT_DIR/config.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get job ID
if [ $# -eq 0 ]; then
    if [ -f /tmp/last_batch_job_id.txt ]; then
        JOB_ID=$(cat /tmp/last_batch_job_id.txt)
        echo -e "${YELLOW}Using last submitted job: $JOB_ID${NC}"
    else
        echo -e "${RED}Error: No job ID provided and no recent job found${NC}"
        echo "Usage: $0 <job_id>"
        exit 1
    fi
else
    JOB_ID="$1"
fi

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}BATCH PROCESSING MONITOR${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "Job ID:     ${GREEN}$JOB_ID${NC}"
echo -e "Started at: ${GREEN}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo ""

# Poll interval in seconds (from config)
POLL_INTERVAL=$BATCH_POLL_INTERVAL

# Function to get checkpoint-based progress (more accurate than logs)
get_checkpoint_progress() {
    docker exec nlp-celery-worker python3 -c "
from src.core.checkpoint_manager import CheckpointManager
import json
checkpoint_mgr = CheckpointManager()
progress = checkpoint_mgr.get_progress('$JOB_ID')
print(json.dumps(progress) if progress else '{}')
" 2>/dev/null || echo "{}"
}

# Function to count processed documents from logs (fallback)
count_processed() {
    docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" | wc -l
}

# Function to get latest processed documents
get_latest_processed() {
    docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" | tail -5
}

# Initial count
LAST_COUNT=$(count_processed)
ITERATION=0

while true; do
    ITERATION=$((ITERATION + 1))

    # Get job status from API
    RESPONSE=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID 2>/dev/null)
    STATUS=$(echo "$RESPONSE" | jq -r '.status' 2>/dev/null || echo "UNKNOWN")

    # Get checkpoint progress (more accurate than logs)
    CHECKPOINT_DATA=$(get_checkpoint_progress)
    if [ "$CHECKPOINT_DATA" != "{}" ]; then
        CURRENT_COUNT=$(echo "$CHECKPOINT_DATA" | jq -r '.processed // 0')
        FAILED_COUNT=$(echo "$CHECKPOINT_DATA" | jq -r '.failed // 0')
        TOTAL_DOCS=$(echo "$CHECKPOINT_DATA" | jq -r '.total_documents // 0')
        CHECKPOINT_STATUS=$(echo "$CHECKPOINT_DATA" | jq -r '.status // "UNKNOWN"')
    else
        # Fallback to log counting
        CURRENT_COUNT=$(count_processed)
        FAILED_COUNT=0
        TOTAL_DOCS=0
        CHECKPOINT_STATUS="UNKNOWN"
    fi

    DOCS_THIS_INTERVAL=$((CURRENT_COUNT - LAST_COUNT))
    LAST_COUNT=$CURRENT_COUNT

    # Calculate rate
    if [ $DOCS_THIS_INTERVAL -gt 0 ]; then
        RATE=$(echo "scale=2; $DOCS_THIS_INTERVAL / ($POLL_INTERVAL / 60)" | bc)
    else
        RATE="0.00"
    fi

    # Display progress with checkpoint info
    TIMESTAMP=$(date '+%H:%M:%S')
    if [ $TOTAL_DOCS -gt 0 ]; then
        PROGRESS_PCT=$(echo "scale=1; ($CURRENT_COUNT / $TOTAL_DOCS) * 100" | bc)
        echo -e "${CYAN}[$ITERATION]${NC} $TIMESTAMP | Status: ${YELLOW}$STATUS${NC} | Progress: ${GREEN}$CURRENT_COUNT/$TOTAL_DOCS${NC} (${PROGRESS_PCT}%) | Failed: ${RED}$FAILED_COUNT${NC} | Rate: ${BLUE}$RATE docs/min${NC}"
    else
        echo -e "${CYAN}[$ITERATION]${NC} $TIMESTAMP | Status: ${YELLOW}$STATUS${NC} | Processed: ${GREEN}$CURRENT_COUNT${NC} | Failed: ${RED}$FAILED_COUNT${NC} | Rate: ${BLUE}$RATE docs/min${NC} (+$DOCS_THIS_INTERVAL)"
    fi

    # Check if job completed
    if [ "$STATUS" = "SUCCESS" ]; then
        echo ""
        echo -e "${GREEN}======================================================================${NC}"
        echo -e "${GREEN}✓ JOB COMPLETED SUCCESSFULLY${NC}"
        echo -e "${GREEN}======================================================================${NC}"
        echo -e "Completed at: ${GREEN}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
        echo -e "Total documents: ${GREEN}$CURRENT_COUNT${NC}"
        echo ""

        # Get results file path (empty if saving disabled)
        RESULTS_FILE=$(get_results_file_path "$JOB_ID")

        # Save and/or display results based on configuration
        save_and_display_json \
            "$RESPONSE" \
            "$RESULTS_FILE" \
            "Results" \
            "Results Summary" \
            '{
                job_id,
                status,
                result: {
                    success_count: .result.success_count,
                    error_count: .result.error_count,
                    storylines: (.result.storylines | length)
                }
            }'

        break
    elif [ "$STATUS" = "FAILURE" ]; then
        echo ""
        echo -e "${RED}======================================================================${NC}"
        echo -e "${RED}✗ JOB FAILED${NC}"
        echo -e "${RED}======================================================================${NC}"
        echo -e "Failed at: ${RED}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
        echo -e "Documents processed before failure: ${YELLOW}$CURRENT_COUNT${NC}"
        echo ""

        # Show error details
        ERROR=$(echo "$RESPONSE" | jq -r '.error // "Unknown error"')
        echo -e "${RED}Error: $ERROR${NC}"
        echo ""

        # Get error file path (empty if saving disabled)
        ERROR_FILE=$(get_error_file_path "$JOB_ID")

        # Save and/or display error details based on configuration
        save_and_display_json \
            "$RESPONSE" \
            "$ERROR_FILE" \
            "Error details" \
            "" \
            '.'

        break
    fi

    sleep $POLL_INTERVAL
done

echo ""
echo -e "${BLUE}Latest processed documents:${NC}"
get_latest_processed

echo ""
echo -e "${YELLOW}To analyze results, run:${NC}"
echo -e "  ./scripts/analyze_batch.sh $JOB_ID"
