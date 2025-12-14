#!/bin/bash
#
# Pause Batch Processing Script
# Usage: ./scripts/pause_batch.sh <job_id>
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ $# -lt 1 ]; then
    if [ -f /tmp/last_batch_job_id.txt ]; then
        JOB_ID=$(cat /tmp/last_batch_job_id.txt)
        echo -e "${YELLOW}Using last submitted job: $JOB_ID${NC}"
    else
        echo -e "${RED}Error: No job ID provided${NC}"
        echo "Usage: $0 <job_id>"
        exit 1
    fi
else
    JOB_ID="$1"
fi

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}PAUSE BATCH PROCESSING${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "Job ID: ${GREEN}$JOB_ID${NC}"
echo ""

# Pause via checkpoint
docker exec nlp-celery-worker python3 << PYTHON_SCRIPT
from src.core.checkpoint_manager import CheckpointManager

checkpoint_mgr = CheckpointManager()
success = checkpoint_mgr.pause("$JOB_ID")

if success:
    print("✓ Batch processing will pause after current document completes")
    checkpoint = checkpoint_mgr.load_checkpoint("$JOB_ID")
    if checkpoint:
        print(f"Status: {checkpoint.status}")
        print(f"Processed: {checkpoint.processed_documents}/{checkpoint.total_documents}")
        print(f"Failed: {checkpoint.failed_documents}")
else:
    print("✗ Failed to pause batch (job may not exist or already completed)")
    exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}✓ PAUSE SIGNAL SENT${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Note:${NC} Processing will pause after the current document finishes."
    echo -e "${YELLOW}To resume:${NC} ./scripts/resume_batch.sh $JOB_ID"
else
    echo ""
    echo -e "${RED}Failed to pause batch${NC}"
    exit 1
fi
