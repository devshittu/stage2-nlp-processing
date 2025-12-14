#!/bin/bash
#
# Resume Batch Processing Script
# Usage: ./scripts/resume_batch.sh <job_id>
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
echo -e "${BLUE}RESUME BATCH PROCESSING${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "Job ID: ${GREEN}$JOB_ID${NC}"
echo ""

# Get checkpoint status
CHECKPOINT_INFO=$(docker exec nlp-celery-worker python3 << PYTHON_SCRIPT
from src.core.checkpoint_manager import CheckpointManager
import json

checkpoint_mgr = CheckpointManager()
checkpoint = checkpoint_mgr.load_checkpoint("$JOB_ID")

if checkpoint:
    print(json.dumps({
        "status": checkpoint.status,
        "processed": checkpoint.processed_documents,
        "failed": checkpoint.failed_documents,
        "total": checkpoint.total_documents,
        "remaining": checkpoint.total_documents - checkpoint.processed_documents - checkpoint.failed_documents
    }))
else:
    print("{}")
PYTHON_SCRIPT
)

if [ "$CHECKPOINT_INFO" = "{}" ]; then
    echo -e "${RED}✗ No checkpoint found for job $JOB_ID${NC}"
    exit 1
fi

STATUS=$(echo "$CHECKPOINT_INFO" | jq -r '.status')
PROCESSED=$(echo "$CHECKPOINT_INFO" | jq -r '.processed')
FAILED=$(echo "$CHECKPOINT_INFO" | jq -r '.failed')
TOTAL=$(echo "$CHECKPOINT_INFO" | jq -r '.total')
REMAINING=$(echo "$CHECKPOINT_INFO" | jq -r '.remaining')

echo "Current Status:"
echo "  Status:     $STATUS"
echo "  Processed:  $PROCESSED"
echo "  Failed:     $FAILED"
echo "  Remaining:  $REMAINING / $TOTAL"
echo ""

if [ "$STATUS" != "PAUSED" ]; then
    echo -e "${YELLOW}⚠ Warning: Batch is not paused (status: $STATUS)${NC}"
    echo "Resume only works for PAUSED batches."
    exit 1
fi

echo -e "${YELLOW}Resuming batch processing...${NC}"
echo ""

# Resume by resubmitting with same job_id (will load checkpoint)
# This requires getting the original documents and resubmitting
echo -e "${RED}Note: Automatic resume not yet fully implemented${NC}"
echo -e "${YELLOW}To resume manually:${NC}"
echo "  1. Re-run the original submit command with the same batch_id"
echo "  2. The system will automatically detect the checkpoint and resume"
echo ""
echo "Alternatively, update checkpoint status to RUNNING:"

docker exec nlp-celery-worker python3 << PYTHON_SCRIPT
from src.core.checkpoint_manager import CheckpointManager

checkpoint_mgr = CheckpointManager()
success = checkpoint_mgr.resume("$JOB_ID")

if success:
    print("✓ Checkpoint status updated to RUNNING")
    print("Resubmit the batch to continue processing")
else:
    print("✗ Failed to update checkpoint")
    exit(1)
PYTHON_SCRIPT
