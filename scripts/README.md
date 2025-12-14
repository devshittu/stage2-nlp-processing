# Batch Processing Scripts

Comprehensive scripts for submitting, monitoring, and analyzing batch NLP processing jobs via Docker.

## 📁 Scripts Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `submit_batch.sh` | Submit a batch processing job | `./scripts/submit_batch.sh <input.jsonl> [batch_id]` |
| `monitor_batch.sh` | Monitor job progress in real-time | `./scripts/monitor_batch.sh [job_id]` |
| `analyze_batch.sh` | Generate comprehensive metrics | `./scripts/analyze_batch.sh [job_id]` |

---

## 🚀 Quick Start

### 1. Ensure Docker Services Are Running

```bash
# Start all services
docker compose up -d

# Verify services are healthy
docker ps --filter "name=nlp-"

# Check orchestrator health
curl http://localhost:8000/health | jq
```

Expected output:
```json
{
  "status": "ok",
  "services": {
    "ner_service": {"status": "ok"},
    "dp_service": {"status": "ok"},
    "event_llm_service": {"status": "ok"},
    "storage": {"status": "ok"}
  }
}
```

---

### 2. Submit a Batch Job

```bash
# Submit batch with automatic batch ID
./scripts/submit_batch.sh data/processed_articles_2025-10-20.jsonl

# Submit batch with custom batch ID
./scripts/submit_batch.sh data/processed_articles_2025-10-20.jsonl my_batch_001
```

**Output:**
```
======================================================================
✓ BATCH SUBMITTED SUCCESSFULLY
======================================================================

Job ID:    e39848d9-04e8-42d6-a5ac-48793f03edcc
Batch ID:  batch_2025-10-20
Documents: 253

Next steps:
  1. Monitor progress:  ./scripts/monitor_batch.sh e39848d9-04e8-42d6-a5ac-48793f03edcc
  2. Check job status:  curl -s http://localhost:8000/api/v1/jobs/e39848d9-04e8-42d6-a5ac-48793f03edcc | jq
  3. View logs:         docker logs nlp-celery-worker --follow
```

---

### 3. Monitor Job Progress

```bash
# Monitor specific job
./scripts/monitor_batch.sh e39848d9-04e8-42d6-a5ac-48793f03edcc

# Monitor last submitted job (auto-detected)
./scripts/monitor_batch.sh
```

**Live Output:**
```
======================================================================
BATCH PROCESSING MONITOR
======================================================================
Job ID:     e39848d9-04e8-42d6-a5ac-48793f03edcc
Started at: 2025-12-14 13:05:00

[1] 13:05:30 | Status: PROGRESS | Processed: 15 | Rate: 1.80 docs/min (+15)
[2] 13:06:00 | Status: PROGRESS | Processed: 33 | Rate: 2.16 docs/min (+18)
[3] 13:06:30 | Status: PROGRESS | Processed: 48 | Rate: 1.80 docs/min (+15)
...
```

Press `Ctrl+C` to stop monitoring (job continues running).

---

### 4. Analyze Results

```bash
# Analyze specific job
./scripts/analyze_batch.sh e39848d9-04e8-42d6-a5ac-48793f03edcc

# Analyze last job
./scripts/analyze_batch.sh
```

**Output:**
- Creates directory `batch_analysis_YYYYMMDD_HHMMSS/` with:
  - `metrics_report.txt` - Full metrics report
  - `summary.json` - JSON summary
  - `processed_docs.log` - Processing timeline
  - `api_response.json` - Full API response
  - `errors.log` - Any errors encountered

---

## 📊 Manual Docker Commands

### Submit Batch (Manual Method)

```bash
# Step 1: Prepare payload
cat data/processed_articles_2025-10-20.jsonl | \
  jq -s '{documents: ., batch_id: "batch_2025-10-20"}' > /tmp/batch_payload.json

# Step 2: Submit to orchestrator API
curl -X POST http://localhost:8000/api/v1/documents/batch \
  -H "Content-Type: application/json" \
  -d @/tmp/batch_payload.json | jq '.'

# Response:
# {
#   "success": true,
#   "batch_id": "batch_2025-10-20",
#   "job_id": "e39848d9-04e8-42d6-a5ac-48793f03edcc",
#   "document_count": 253,
#   "message": "Batch processing started..."
# }
```

### Monitor Job (Manual Method)

```bash
# Set your job ID
export JOB_ID="e39848d9-04e8-42d6-a5ac-48793f03edcc"

# Check job status via API
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq '{status, progress}'

# Count processed documents from logs
docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" | wc -l

# View latest processed documents
docker logs nlp-celery-worker 2>&1 | grep "✓ Document" | tail -10

# Watch processing in real-time
docker logs nlp-celery-worker --follow
```

### Continuous Monitoring Loop

```bash
# Monitor progress every 30 seconds
export JOB_ID="e39848d9-04e8-42d6-a5ac-48793f03edcc"

while true; do
  STATUS=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq -r '.status')
  COUNT=$(docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" | wc -l)

  echo "[$(date '+%H:%M:%S')] Status: $STATUS | Processed: $COUNT documents"

  if [ "$STATUS" = "SUCCESS" ] || [ "$STATUS" = "FAILURE" ]; then
    echo "Job completed with status: $STATUS"
    break
  fi

  sleep 30
done
```

---

## 📁 Retrieve Results

### Find Output Files

```bash
# List output files inside container
docker exec nlp-orchestrator ls -lht /app/data/

# Find today's output
docker exec nlp-orchestrator find /app/data -name "extracted_events_*.jsonl" -mtime -1
```

### Copy Results from Container

```bash
# Copy specific file
docker cp nlp-orchestrator:/app/data/extracted_events_2025-12-14.jsonl ./data/

# Copy all JSONL files
docker exec nlp-orchestrator sh -c 'tar czf - /app/data/*.jsonl' | tar xzf - -C ./data/
```

### Get Results via API

```bash
# Get complete job results
export JOB_ID="e39848d9-04e8-42d6-a5ac-48793f03edcc"

curl -s http://localhost:8000/api/v1/jobs/$JOB_ID > batch_results.json

# View summary
jq '{job_id, status, result: {success_count, error_count, storylines}}' batch_results.json

# Extract storylines
jq '.result.storylines' batch_results.json > storylines.json

# Extract events (if available in result)
jq '.result.events' batch_results.json > events.json
```

---

## 🔧 Troubleshooting

### Services Not Running

```bash
# Check service status
docker compose ps

# Start services
docker compose up -d

# View logs if services fail to start
docker compose logs orchestrator
docker compose logs celery-worker
```

### Job Stuck in PROGRESS

```bash
# Check if worker is alive
docker logs nlp-celery-worker --tail 50

# Check for processing activity
docker logs nlp-celery-worker 2>&1 | grep "✓ Document" | tail -20

# Restart worker if frozen
docker compose restart celery-worker
```

### Job Failed with Error

```bash
export JOB_ID="your-job-id"

# Get error details
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq '.error'

# Check worker logs for exceptions
docker logs nlp-celery-worker 2>&1 | grep -i "error\|exception" | tail -20

# Check individual service logs
docker logs nlp-ner-service --tail 50
docker logs nlp-dp-service --tail 50
docker logs nlp-event-llm-service --tail 50
```

### Out of Memory

```bash
# Check container memory usage
docker stats --no-stream

# Restart services to free memory
docker compose restart

# Process smaller batches
# Split your input file into chunks of 50-100 documents
split -l 100 data/large_file.jsonl data/chunk_
```

---

## ⚙️ Configuration

### Timeout Settings

The batch processing timeout has been set to **10 hours** to handle large batches:

**Config file:** `config/settings.yaml`
```yaml
celery:
  task_time_limit: 36000      # 10 hours hard limit
  task_soft_time_limit: 32400 # 9 hours soft limit
```

To change timeout:
1. Edit `config/settings.yaml`
2. Rebuild and restart: `docker compose down && docker compose build && docker compose up -d`

### Processing Optimization

**Increase worker count** (requires more CPU/GPU):
```yaml
# docker compose.yml
services:
  celery-worker:
    deploy:
      replicas: 3  # Run 3 workers in parallel
```

**Adjust worker concurrency**:
```bash
# In docker compose.yml, update celery-worker command:
command: celery -A src.core.celery_tasks worker --loglevel=info --concurrency=4
```

---

## 📊 Expected Performance

Based on testing with 253 documents:

| Metric | Value |
|--------|-------|
| **Processing rate** | ~2.0-2.5 docs/minute |
| **Average time per doc** | ~25-30 seconds |
| **Time for 253 docs** | ~2 hours |
| **Time for 100 docs** | ~45-50 minutes |
| **Success rate** | 100% (no errors) |

**Performance varies based on:**
- Document length (longer = slower)
- Event complexity (more events = slower LLM processing)
- Available GPU memory
- System load

---

## 🎯 Best Practices

1. **Always monitor the first few documents** to ensure processing works correctly
2. **Use batch IDs** to track different runs (e.g., `batch_2025-12-14_production`)
3. **Keep input files under 500 documents** for easier management
4. **Save job IDs** for later analysis
5. **Check logs regularly** during long-running jobs
6. **Backup output files** as they're generated

---

## 📚 Additional Resources

- **API Documentation:** `http://localhost:8000/api/v1/docs` (when services running)
- **Health Check:** `curl http://localhost:8000/health | jq`
- **Service Logs:** `docker compose logs -f [service-name]`
- **Configuration:** `config/settings.yaml`

---

## 🐛 Common Issues

### "Connection refused" error
```bash
# Services not started
docker compose up -d
```

### "TimeLimitExceeded" error
```bash
# Increase timeout in config/settings.yaml
# Or split batch into smaller chunks
```

### No output file generated
```bash
# Check if job completed successfully
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq '.status'

# If job failed/timed out, results weren't persisted
# Re-run with increased timeout
```

---

## 💡 Quick Tips

```bash
# Tail all service logs simultaneously
docker compose logs -f

# Watch document count in real-time
watch -n 10 'docker logs nlp-celery-worker 2>&1 | grep "✓ Document" | wc -l'

# Get processing rate (docs/minute)
# Count docs at start and after 5 minutes, then calculate

# Kill a stuck job
docker compose restart celery-worker

# Clear all data and restart fresh
docker compose down -v
docker compose up -d
```

---

**Need help?** Check the main project README or examine the service logs for detailed error messages.
