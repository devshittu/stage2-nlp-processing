#!/bin/bash
#
# Batch Processing Submission Script
# Usage: ./scripts/submit_batch.sh <input_file.jsonl> [batch_id]
#
# Example:
#   ./scripts/submit_batch.sh data/processed_articles_2025-10-20.jsonl batch_2025-10-20
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Input file required${NC}"
    echo "Usage: $0 <input_file.jsonl> [batch_id]"
    echo ""
    echo "Example:"
    echo "  $0 data/processed_articles_2025-10-20.jsonl batch_2025-10-20"
    exit 1
fi

INPUT_FILE="$1"
BATCH_ID="${2:-batch_$(date +%Y%m%d_%H%M%S)}"

# Validate input file
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Count documents
DOC_COUNT=$(wc -l < "$INPUT_FILE")
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}BATCH PROCESSING SUBMISSION${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo -e "Input file:     ${GREEN}$INPUT_FILE${NC}"
echo -e "Document count: ${GREEN}$DOC_COUNT${NC}"
echo -e "Batch ID:       ${GREEN}$BATCH_ID${NC}"
echo ""

# Check if Docker services are running
echo -e "${YELLOW}Checking Docker services...${NC}"
if ! docker ps --filter "name=nlp-orchestrator" --format "{{.Names}}" | grep -q "nlp-orchestrator"; then
    echo -e "${RED}Error: nlp-orchestrator service is not running${NC}"
    echo "Start services with: docker compose up -d"
    exit 1
fi

# Check orchestrator health
HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null || echo "error")
if [ "$HEALTH" != "ok" ]; then
    echo -e "${RED}Error: Orchestrator service is not healthy${NC}"
    echo "Check status with: curl http://localhost:8000/health | jq"
    exit 1
fi
echo -e "${GREEN}✓ All services healthy${NC}"
echo ""

# Create payload
PAYLOAD_FILE="/tmp/batch_payload_$(date +%s).json"
echo -e "${YELLOW}Creating batch payload...${NC}"
cat "$INPUT_FILE" | jq -s "{documents: ., batch_id: \"$BATCH_ID\"}" > "$PAYLOAD_FILE"

PAYLOAD_SIZE=$(du -h "$PAYLOAD_FILE" | cut -f1)
echo -e "${GREEN}✓ Payload created: $PAYLOAD_SIZE${NC}"
echo ""

# Submit batch
echo -e "${YELLOW}Submitting batch job...${NC}"
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/documents/batch \
    -H "Content-Type: application/json" \
    -d @"$PAYLOAD_FILE")

# Parse response
SUCCESS=$(echo "$RESPONSE" | jq -r '.success' 2>/dev/null || echo "false")

if [ "$SUCCESS" = "true" ]; then
    JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}✓ BATCH SUBMITTED SUCCESSFULLY${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    echo ""
    echo -e "Job ID:    ${GREEN}$JOB_ID${NC}"
    echo -e "Batch ID:  ${GREEN}$BATCH_ID${NC}"
    echo -e "Documents: ${GREEN}$DOC_COUNT${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Monitor progress:  ${YELLOW}./scripts/monitor_batch.sh $JOB_ID${NC}"
    echo -e "  2. Check job status:  ${YELLOW}curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq${NC}"
    echo -e "  3. View logs:         ${YELLOW}docker logs nlp-celery-worker --follow${NC}"
    echo ""

    # Save job info
    echo "$JOB_ID" > /tmp/last_batch_job_id.txt
    echo "$RESPONSE" | jq '.' > "/tmp/batch_submission_${JOB_ID}.json"

    # Cleanup payload
    rm -f "$PAYLOAD_FILE"
else
    echo -e "${RED}======================================================================${NC}"
    echo -e "${RED}✗ BATCH SUBMISSION FAILED${NC}"
    echo -e "${RED}======================================================================${NC}"
    echo ""
    echo -e "${RED}Response:${NC}"
    echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
    rm -f "$PAYLOAD_FILE"
    exit 1
fi
