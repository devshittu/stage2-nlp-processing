#!/bin/bash
#
# Batch Processing Analysis Script
# Usage: ./scripts/analyze_batch.sh [job_id]
#
# Generates comprehensive metrics and analysis from batch processing logs
#
# Configuration:
#   BATCH_SAVE_ANALYSIS=false      - Don't save analysis files
#   BATCH_DISPLAY_ANALYSIS=false   - Don't display analysis in terminal
#   BATCH_OUTPUT_DIR=/custom/path  - Custom output directory
#
# Examples:
#   ./scripts/analyze_batch.sh                           # Normal mode (save + display)
#   BATCH_SAVE_ANALYSIS=false ./scripts/analyze_batch.sh # Display only
#   BATCH_DISPLAY_ANALYSIS=false ./scripts/analyze_batch.sh # Save only
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
    else
        echo -e "${RED}Error: No job ID provided${NC}"
        echo "Usage: $0 <job_id>"
        exit 1
    fi
else
    JOB_ID="$1"
fi

# Get analysis directory (configurable)
OUTPUT_DIR=$(get_analysis_dir_path)
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}BATCH PROCESSING ANALYSIS${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "Job ID:      ${GREEN}$JOB_ID${NC}"
if [ "$BATCH_SAVE_ANALYSIS" = "true" ]; then
    echo -e "Output dir:  ${GREEN}$OUTPUT_DIR${NC}"
else
    echo -e "Mode:        ${YELLOW}Display only (not saving)${NC}"
fi
echo ""

# Extract processed documents from logs
echo -e "${YELLOW}Extracting processing logs...${NC}"
docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" > "$OUTPUT_DIR/processed_docs.log"
TOTAL_PROCESSED=$(wc -l < "$OUTPUT_DIR/processed_docs.log")
echo -e "${GREEN}✓ Found $TOTAL_PROCESSED processed documents${NC}"

# Get API response
echo -e "${YELLOW}Retrieving job status from API...${NC}"
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq '.' > "$OUTPUT_DIR/api_response.json"
STATUS=$(jq -r '.status' < "$OUTPUT_DIR/api_response.json")
echo -e "${GREEN}✓ Job status: $STATUS${NC}"

# Extract errors if any
echo -e "${YELLOW}Checking for errors...${NC}"
docker logs nlp-celery-worker 2>&1 | grep -i "ERROR\|FAILURE\|Exception" > "$OUTPUT_DIR/errors.log" 2>/dev/null || true
ERROR_COUNT=$(wc -l < "$OUTPUT_DIR/errors.log" 2>/dev/null || echo "0")
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $ERROR_COUNT error entries${NC}"
else
    echo -e "${GREEN}✓ No errors found${NC}"
fi

# Generate Python analysis
echo -e "${YELLOW}Generating detailed metrics...${NC}"

cat > "$OUTPUT_DIR/analyze.py" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import re
import json
from datetime import datetime
from collections import defaultdict

# Read processed documents log
with open('processed_docs.log', 'r') as f:
    logs = f.readlines()

print("="*80)
print("COMPREHENSIVE BATCH PROCESSING METRICS")
print("="*80)
print()

# Extract timestamps and document IDs
timestamps = []
doc_ids = []
for line in logs:
    ts_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+', line)
    doc_match = re.search(r'Document (doc_\d+)', line)
    if ts_match and doc_match:
        ts = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
        timestamps.append(ts)
        doc_ids.append(doc_match.group(1))

total_docs = len(timestamps)
print(f"📊 OVERALL STATISTICS")
print(f"{'='*80}")
print(f"Successfully processed:          {total_docs}")
print()

# Time analysis
if len(timestamps) > 1:
    start_time = timestamps[0]
    end_time = timestamps[-1]
    total_time = (end_time - start_time).total_seconds()

    print(f"⏱️  TIMING ANALYSIS")
    print(f"{'='*80}")
    print(f"Start time:                      {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:                        {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total processing time:           {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"Processing rate:                 {total_docs/(total_time/60):.2f} docs/minute")
    print(f"Average time per document:       {total_time/total_docs:.1f} seconds")
    print()

    # Calculate intervals
    intervals = []
    for i in range(1, len(timestamps)):
        interval = (timestamps[i] - timestamps[i-1]).total_seconds()
        intervals.append(interval)

    print(f"📈 PROCESSING TIME DISTRIBUTION")
    print(f"{'='*80}")
    print(f"Minimum interval:                {min(intervals):.1f} seconds")
    print(f"Maximum interval:                {max(intervals):.1f} seconds")
    print(f"Median interval:                 {sorted(intervals)[len(intervals)//2]:.1f} seconds")
    print(f"Average interval:                {sum(intervals)/len(intervals):.1f} seconds")
    print()

    # Parallel processing detection
    simultaneous = sum(1 for i in intervals if i < 1)
    print(f"🔄 PARALLELIZATION")
    print(f"{'='*80}")
    print(f"Simultaneous completions:        {simultaneous} ({simultaneous/len(intervals)*100:.1f}%)")
    print(f"Sequential completions:          {len(intervals)-simultaneous}")
    print()

    # Time buckets
    fast = sum(1 for i in intervals if i < 10)
    normal = sum(1 for i in intervals if 10 <= i < 40)
    slow = sum(1 for i in intervals if i >= 40)

    print(f"Processing speed distribution:")
    print(f"  Fast (<10s):                   {fast} ({fast/len(intervals)*100:.1f}%)")
    print(f"  Normal (10-40s):               {normal} ({normal/len(intervals)*100:.1f}%)")
    print(f"  Slow (≥40s):                   {slow} ({slow/len(intervals)*100:.1f}%)")
    print()

# Document coverage
print(f"📄 DOCUMENT COVERAGE")
print(f"{'='*80}")
print(f"First document:                  {doc_ids[0]}")
print(f"Last document:                   {doc_ids[-1]}")
print()

# Save summary
summary = {
    "total_processed": total_docs,
    "start_time": start_time.isoformat() if timestamps else None,
    "end_time": end_time.isoformat() if timestamps else None,
    "processing_time_seconds": total_time if len(timestamps) > 1 else 0,
    "processing_rate_per_minute": total_docs/(total_time/60) if len(timestamps) > 1 else 0,
    "average_seconds_per_doc": total_time/total_docs if total_docs > 0 else 0,
}

with open('summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print()
print("Summary saved to: summary.json")
PYTHON_SCRIPT

# Run Python analysis
cd "$OUTPUT_DIR"
python3 analyze.py > metrics_report.txt

# Display or save based on configuration
if [ "$BATCH_DISPLAY_ANALYSIS" = "true" ]; then
    cat metrics_report.txt
fi

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}✓ ANALYSIS COMPLETE${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""

if [ "$BATCH_SAVE_ANALYSIS" = "true" ]; then
    echo -e "${BLUE}Generated files in: ${GREEN}$OUTPUT_DIR${NC}"
    ls -lh "$OUTPUT_DIR"
    echo ""
    echo -e "${YELLOW}View full report:${NC} cat $OUTPUT_DIR/metrics_report.txt"
    echo -e "${YELLOW}View summary:${NC}     cat $OUTPUT_DIR/summary.json | jq"
else
    echo -e "${YELLOW}Analysis displayed above (not saved to disk)${NC}"
    echo -e "${YELLOW}To save analysis, run with:${NC} BATCH_SAVE_ANALYSIS=true $0 $JOB_ID"
fi
