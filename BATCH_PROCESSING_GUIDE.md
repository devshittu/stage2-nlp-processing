# Batch Processing Guide - Quick Reference

This guide provides step-by-step instructions for running batch NLP processing jobs using Docker.

## 🎯 Quick Start (3 Simple Steps)

### Step 1: Start Docker Services

```bash
docker compose up -d
```

Wait ~30 seconds for services to initialize, then verify:

```bash
curl http://localhost:8000/health | jq
```

You should see `"status": "ok"`.

---

### Step 2: Submit Your Batch

```bash
./scripts/submit_batch.sh data/your_input_file.jsonl
```

This will output a **Job ID** - save it for monitoring.

Example output:
```
✓ BATCH SUBMITTED SUCCESSFULLY
Job ID:    e39848d9-04e8-42d6-a5ac-48793f03edcc
Documents: 253
```

---

### Step 3: Monitor Progress

```bash
./scripts/monitor_batch.sh e39848d9-04e8-42d6-a5ac-48793f03edcc
```

Or monitor the last submitted job:
```bash
./scripts/monitor_batch.sh
```

**That's it!** The script will monitor until completion and save results automatically.

---

## 📋 Complete Docker Command Reference

If you prefer to run commands manually without scripts:

### 1. Prepare Batch Payload

```bash
# Create JSON payload from your JSONL file
cat data/processed_articles_2025-10-20.jsonl | \
  jq -s '{documents: ., batch_id: "batch_2025-12-14"}' > /tmp/batch_payload.json

# Verify payload size
ls -lh /tmp/batch_payload.json
```

### 2. Submit to Docker API

```bash
# Submit batch job to orchestrator
curl -X POST http://localhost:8000/api/v1/documents/batch \
  -H "Content-Type: application/json" \
  -d @/tmp/batch_payload.json | jq '.'
```

**Save the `job_id` from the response!**

Example response:
```json
{
  "success": true,
  "batch_id": "batch_2025-12-14",
  "job_id": "e39848d9-04e8-42d6-a5ac-48793f03edcc",
  "document_count": 253,
  "message": "Batch processing started. Track progress with job_id..."
}
```

### 3. Monitor Job Status

```bash
# Set your job ID (replace with actual ID from step 2)
export JOB_ID="e39848d9-04e8-42d6-a5ac-48793f03edcc"

# Check status via API
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq '{status, progress}'

# Count processed documents
docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" | wc -l

# View latest 10 processed documents
docker logs nlp-celery-worker 2>&1 | grep "✓ Document" | tail -10
```

### 4. Watch Progress Live

```bash
# Method 1: Watch Docker logs
docker logs nlp-celery-worker --follow

# Method 2: Auto-refresh document count every 30 seconds
watch -n 30 'docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" | wc -l'

# Method 3: Monitoring loop (paste this into terminal)
while true; do
  STATUS=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq -r '.status')
  COUNT=$(docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" | wc -l)
  echo "[$(date '+%H:%M:%S')] Status: $STATUS | Processed: $COUNT documents"

  if [ "$STATUS" = "SUCCESS" ] || [ "$STATUS" = "FAILURE" ]; then
    echo "Job completed: $STATUS"
    break
  fi

  sleep 30
done
```

### 5. Retrieve Results

```bash
# Get final job results
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID > batch_results.json

# View results summary
cat batch_results.json | jq '{
  job_id,
  status,
  result: {
    success_count: .result.success_count,
    error_count: .result.error_count,
    storylines: (.result.storylines | length)
  }
}'

# Find output files in Docker container
docker exec nlp-orchestrator ls -lht /app/data/ | head -10

# Copy output file from container to host
docker cp nlp-orchestrator:/app/data/extracted_events_2025-12-14.jsonl ./data/
```

---

## 🔍 All Docker Commands You Need

### Service Management

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# Restart specific service
docker compose restart celery-worker

# View all running services
docker ps --filter "name=nlp-"

# Check service health
curl http://localhost:8000/health | jq
```

### Log Viewing

```bash
# View Celery worker logs (main processing)
docker logs nlp-celery-worker

# Follow logs in real-time
docker logs nlp-celery-worker --follow

# Last 50 lines
docker logs nlp-celery-worker --tail 50

# View orchestrator logs
docker logs nlp-orchestrator --tail 50

# View all service logs together
docker compose logs -f

# View specific service
docker compose logs -f ner-service
docker compose logs -f dp-service
docker compose logs -f event-llm-service
```

### Processing Metrics

```bash
# Count total processed documents
docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" | wc -l

# Show first 20 processed
docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" | head -20

# Show last 20 processed
docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" | tail -20

# Check for errors
docker logs nlp-celery-worker 2>&1 | grep -i "error\|exception\|failed"

# View processing timeline (with timestamps)
docker logs nlp-celery-worker 2>&1 | grep "✓ Document" | awk '{print $1, $2, $NF}'
```

### Data Access

```bash
# List files inside container
docker exec nlp-orchestrator ls -lh /app/data/

# View file contents (first 10 lines of JSONL)
docker exec nlp-orchestrator head -10 /app/data/extracted_events_2025-12-14.jsonl

# Count lines in output file
docker exec nlp-orchestrator wc -l /app/data/extracted_events_2025-12-14.jsonl

# Copy file from container
docker cp nlp-orchestrator:/app/data/extracted_events_2025-12-14.jsonl ./data/

# Copy all JSONL files
for file in $(docker exec nlp-orchestrator find /app/data -name "*.jsonl" -type f); do
  docker cp "nlp-orchestrator:$file" ./data/
done
```

---

## ⏱️ Expected Processing Time

| Documents | Estimated Time |
|-----------|----------------|
| 50 docs   | ~20-25 minutes |
| 100 docs  | ~45-50 minutes |
| 253 docs  | ~2 hours |
| 500 docs  | ~4 hours |

**Processing rate:** ~2-2.5 documents per minute
**Per document:** ~25-30 seconds average

---

## 🎯 Example: Complete Workflow

Here's a complete example from start to finish:

```bash
# 1. Ensure services are running
docker compose up -d
sleep 30  # Wait for services to start

# 2. Verify health
curl http://localhost:8000/health | jq '.status'
# Should return: "ok"

# 3. Prepare your data
cat data/processed_articles_2025-10-20.jsonl | \
  jq -s '{documents: ., batch_id: "batch_test_001"}' > /tmp/batch.json

# 4. Submit batch
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/documents/batch \
  -H "Content-Type: application/json" \
  -d @/tmp/batch.json)

# 5. Extract job ID
JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
echo "Job ID: $JOB_ID"

# 6. Monitor progress (run in separate terminal)
watch -n 30 "docker logs nlp-celery-worker 2>&1 | grep '✓ Document doc_' | wc -l"

# 7. Check status periodically
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq '{status, progress}'

# 8. When complete, get results
curl -s http://localhost:8000/api/v1/jobs/$JOB_ID > results_$JOB_ID.json

# 9. Extract output file
docker cp nlp-orchestrator:/app/data/extracted_events_$(date +%Y-%m-%d).jsonl ./data/

# 10. Analyze results
cat results_$JOB_ID.json | jq
```

---

## 🐛 Troubleshooting Docker Commands

### Check if services are stuck

```bash
# View resource usage
docker stats --no-stream

# Check if container is healthy
docker inspect nlp-orchestrator | jq '.[0].State.Health'
docker inspect nlp-ner-service | jq '.[0].State.Health'
```

### Restart frozen services

```bash
# Restart Celery worker
docker compose restart celery-worker

# Restart all services
docker compose restart

# Full restart (clears state)
docker compose down
docker compose up -d
```

### View detailed errors

```bash
# Get full error log from worker
docker logs nlp-celery-worker > worker_logs.txt
grep -i "error\|exception\|traceback" worker_logs.txt

# Check individual service errors
docker logs nlp-ner-service 2>&1 | grep -i error
docker logs nlp-dp-service 2>&1 | grep -i error
docker logs nlp-event-llm-service 2>&1 | grep -i error
```

---

## 📊 Real-time Dashboard (Terminal-based)

Create a simple monitoring dashboard:

```bash
# Run this in a terminal - updates every 10 seconds
while true; do
  clear
  echo "==================== BATCH PROCESSING DASHBOARD ===================="
  echo ""
  echo "Job ID: $JOB_ID"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""

  # Get status
  STATUS=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"

  # Count processed
  PROCESSED=$(docker logs nlp-celery-worker 2>&1 | grep "✓ Document doc_" | wc -l)
  echo "Processed: $PROCESSED documents"

  # Get latest
  echo ""
  echo "Latest 5 processed:"
  docker logs nlp-celery-worker 2>&1 | grep "✓ Document" | tail -5

  # Check if complete
  if [ "$STATUS" = "SUCCESS" ] || [ "$STATUS" = "FAILURE" ]; then
    echo ""
    echo "===================== JOB COMPLETED: $STATUS ====================="
    break
  fi

  sleep 10
done
```

---

## 💾 Backup and Storage

```bash
# Backup all output files
docker exec nlp-orchestrator tar czf /tmp/outputs.tar.gz /app/data/*.jsonl
docker cp nlp-orchestrator:/tmp/outputs.tar.gz ./backups/batch_$(date +%Y%m%d).tar.gz

# Extract backup
tar xzf ./backups/batch_20251214.tar.gz
```

---

## 🔧 Configuration Changes Made

The following files were updated to support long-running batches:

### `config/settings.yaml`
```yaml
celery:
  task_time_limit: 36000      # 10 hours (was 1 hour)
  task_soft_time_limit: 32400 # 9 hours (was 50 minutes)
```

### `src/utils/config_manager.py`
```python
task_time_limit: int = Field(default=36000)      # 10 hours
task_soft_time_limit: int = Field(default=32400) # 9 hours
```

**To apply changes:** Rebuild and restart services
```bash
docker compose down
docker compose build celery-worker orchestrator
docker compose up -d
```

---

## 📚 Script Locations

- **Submit batch:** `./scripts/submit_batch.sh`
- **Monitor batch:** `./scripts/monitor_batch.sh`
- **Analyze results:** `./scripts/analyze_batch.sh`
- **Documentation:** `./scripts/README.md`

All scripts support `--help` for more options.

---

**For detailed documentation, see:** `scripts/README.md`
