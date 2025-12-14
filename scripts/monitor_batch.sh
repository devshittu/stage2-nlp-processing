#!/bin/bash
#
# Batch Processing Monitor Script
# Usage: ./scripts/monitor_batch.sh [job_id]
#
# If no job_id provided, uses the last submitted job
#

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

# Poll interval in seconds
POLL_INTERVAL=30

# Function to count processed documents from logs
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

    # Count processed documents
    CURRENT_COUNT=$(count_processed)
    DOCS_THIS_INTERVAL=$((CURRENT_COUNT - LAST_COUNT))
    LAST_COUNT=$CURRENT_COUNT

    # Calculate rate
    if [ $DOCS_THIS_INTERVAL -gt 0 ]; then
        RATE=$(echo "scale=2; $DOCS_THIS_INTERVAL / ($POLL_INTERVAL / 60)" | bc)
    else
        RATE="0.00"
    fi

    # Display progress
    TIMESTAMP=$(date '+%H:%M:%S')
    echo -e "${CYAN}[$ITERATION]${NC} $TIMESTAMP | Status: ${YELLOW}$STATUS${NC} | Processed: ${GREEN}$CURRENT_COUNT${NC} | Rate: ${BLUE}$RATE docs/min${NC} (+$DOCS_THIS_INTERVAL)"

    # Check if job completed
    if [ "$STATUS" = "SUCCESS" ]; then
        echo ""
        echo -e "${GREEN}======================================================================${NC}"
        echo -e "${GREEN}✓ JOB COMPLETED SUCCESSFULLY${NC}"
        echo -e "${GREEN}======================================================================${NC}"
        echo -e "Completed at: ${GREEN}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
        echo -e "Total documents: ${GREEN}$CURRENT_COUNT${NC}"
        echo ""

        # Save final results
        RESULTS_FILE="batch_results_${JOB_ID}.json"
        echo "$RESPONSE" | jq '.' > "$RESULTS_FILE"
        echo -e "${BLUE}Results saved to: ${GREEN}$RESULTS_FILE${NC}"

        # Show results summary
        echo ""
        echo -e "${BLUE}Results Summary:${NC}"
        echo "$RESPONSE" | jq '{
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

        # Save error details
        ERROR_FILE="batch_error_${JOB_ID}.json"
        echo "$RESPONSE" | jq '.' > "$ERROR_FILE"
        echo -e "${BLUE}Error details saved to: ${YELLOW}$ERROR_FILE${NC}"

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
