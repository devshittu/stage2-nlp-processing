#!/bin/bash
#
# Check Batch Processing Checkpoint Status
# Usage: ./scripts/checkpoint_status.sh [job_id]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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
echo -e "${BLUE}CHECKPOINT STATUS${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "Job ID: ${GREEN}$JOB_ID${NC}"
echo ""

# Get checkpoint progress
docker exec nlp-celery-worker python3 << PYTHON_SCRIPT
from src.core.checkpoint_manager import CheckpointManager
from datetime import datetime

checkpoint_mgr = CheckpointManager()
progress = checkpoint_mgr.get_progress("$JOB_ID")

if not progress:
    print("❌ No checkpoint found for job $JOB_ID")
    exit(1)

# Print progress
print(f"{'Status:':<20} {progress['status']}")
print(f"{'Batch ID:':<20} {progress['batch_id']}")
print(f"{'Total Documents:':<20} {progress['total_documents']}")
print(f"{'Processed:':<20} {progress['processed']} ({progress['progress_percentage']:.1f}%)")
print(f"{'Failed:':<20} {progress['failed']}")
print(f"{'Remaining:':<20} {progress['remaining']}")
print(f"{'Created:':<20} {progress['created_at']}")
print(f"{'Updated:':<20} {progress['updated_at']}")
print("")

# Load full checkpoint for detailed info
checkpoint = checkpoint_mgr.load_checkpoint("$JOB_ID")
if checkpoint:
    print("Recent Processed Documents (last 10):")
    for doc_id in checkpoint.processed_doc_ids[-10:]:
        print(f"  ✓ {doc_id}")

    if checkpoint.failed_doc_ids:
        print("")
        print(f"Failed Documents ({len(checkpoint.failed_doc_ids)}):")
        for doc_id in checkpoint.failed_doc_ids[:10]:
            print(f"  ✗ {doc_id}")
        if len(checkpoint.failed_doc_ids) > 10:
            print(f"  ... and {len(checkpoint.failed_doc_ids) - 10} more")
PYTHON_SCRIPT
