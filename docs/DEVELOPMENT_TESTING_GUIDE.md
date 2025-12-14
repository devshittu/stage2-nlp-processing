# Development Testing Guide

**Version**: 1.0
**Date**: December 2025

This guide provides step-by-step instructions for starting, testing, and developing the NLP Processing Service using Docker.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Starting the Services](#starting-the-services)
3. [Running Unit Tests](#running-unit-tests)
4. [Testing Event System](#testing-event-system)
5. [Manual End-to-End Testing](#manual-end-to-end-testing)
6. [Development Workflow](#development-workflow)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- Docker and Docker Compose installed
- At least 16GB RAM available
- NVIDIA GPU with CUDA support (optional, can run on CPU)
- 50GB free disk space

### Verify Docker Setup

```bash
# Check Docker is running
docker --version
docker compose version

# Check available resources
docker system info | grep -E "CPUs|Total Memory"
```

---

## Starting the Services

### Step 1: Start All Services

```bash
# From project root directory
cd /home/mshittu/projects/nlp/stage2-nlp-processing

# Start all services in detached mode
docker compose up -d
```

**Expected output:**
```
Container nlp-redis Running
Container nlp-ner-service Running
Container nlp-dp-service Running
Container nlp-event-llm-service Running
Container nlp-orchestrator Running
Container nlp-celery-worker Running
```

### Step 2: Check Service Health

```bash
# Check all containers are running
docker compose ps
```

**Expected output:**
```
NAME                    STATUS                    PORTS
nlp-orchestrator        Up X minutes (healthy)    0.0.0.0:8000->8000/tcp
nlp-celery-worker       Up X minutes
nlp-ner-service         Up X minutes (healthy)    0.0.0.0:8001->8001/tcp
nlp-dp-service          Up X minutes (healthy)    0.0.0.0:8002->8002/tcp
nlp-event-llm-service   Up X minutes (healthy)    0.0.0.0:8003->8003/tcp
nlp-redis               Up X minutes (healthy)    0.0.0.0:6379->6379/tcp
```

### Step 3: Verify Service Logs

```bash
# Check orchestrator started successfully
docker logs nlp-orchestrator --tail 20

# Look for:
# - "Event publisher disabled (events.enabled = false)"
# - "Orchestrator service started successfully"
# - "Application startup complete"
```

**Key log messages:**
```
INFO: Event publisher disabled (events.enabled = false)
INFO: Storage writer initialized
INFO: Orchestrator service started successfully
INFO: Application startup complete
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Test API Endpoint

```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health

# Expected response:
# {"status":"healthy","services":{"ner_service":"healthy","dp_service":"healthy","event_llm_service":"healthy"}}
```

---

## Running Unit Tests

### All Tests (Full Test Suite)

```bash
# Run all tests
docker exec nlp-orchestrator pytest /app/tests/ -v

# Expected: All tests pass
```

### Event System Tests Only

```bash
# Run event system tests (47 tests)
docker exec nlp-orchestrator pytest /app/tests/events/ -v

# Expected output:
# 47 passed in X.XXs
```

### Specific Test File

```bash
# Test CloudEvent models
docker exec nlp-orchestrator pytest /app/tests/events/test_events_models.py -v

# Test backends
docker exec nlp-orchestrator pytest /app/tests/events/test_events_backends.py -v

# Test publisher
docker exec nlp-orchestrator pytest /app/tests/events/test_events_publisher.py -v
```

### With Coverage Report

```bash
# Run tests with coverage
docker exec nlp-orchestrator pytest /app/tests/events/ --cov=src/events --cov-report=term-missing

# View detailed coverage
docker exec nlp-orchestrator pytest /app/tests/events/ --cov=src/events --cov-report=html
```

### Quick Test (Failures Only)

```bash
# Run tests and stop at first failure
docker exec nlp-orchestrator pytest /app/tests/events/ -x -v
```

---

## Testing Event System

### Enable Event Publishing

**Step 1: Edit Configuration**

```bash
# Edit settings.yaml in the container
docker exec nlp-orchestrator vi /app/config/settings.yaml

# Or copy from host, edit, and copy back
docker cp nlp-orchestrator:/app/config/settings.yaml ./config/settings.yaml
# Edit locally: change events.enabled from false to true
docker cp ./config/settings.yaml nlp-orchestrator:/app/config/settings.yaml
```

**Change this:**
```yaml
events:
  enabled: false  # Change to true
```

**To this:**
```yaml
events:
  enabled: true
  backend: "redis_streams"
```

**Step 2: Restart Services**

```bash
# Restart orchestrator and celery worker to pick up config changes
docker compose restart orchestrator celery-worker

# Verify event publisher is enabled
docker logs nlp-orchestrator --tail 20 | grep "Event publisher"

# Expected: "Event publisher initialized with backend: redis_streams"
```

### Monitor Events in Redis

**Step 3: Open Redis CLI**

```bash
# Open Redis CLI in separate terminal
docker exec -it nlp-redis redis-cli
```

**Step 4: Monitor Event Stream**

```redis
# Monitor stream in real-time
XREAD COUNT 10 STREAMS nlp-events 0

# Check stream info
XINFO STREAM nlp-events

# Get latest events
XREVRANGE nlp-events + - COUNT 5
```

### Test Event Publishing

**Step 5: Process a Test Document**

```bash
# Send a test document to the API
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "document": {
      "document_id": "test-doc-001",
      "cleaned_text": "Apple CEO Tim Cook met with Microsoft CEO Satya Nadella in Washington on Monday.",
      "cleaned_title": "Tech Leaders Meet",
      "cleaned_publication_date": "2025-12-14T10:00:00Z"
    }
  }'
```

**Step 6: Verify Event in Redis**

```bash
# In Redis CLI, read the latest event
docker exec nlp-redis redis-cli XREVRANGE nlp-events + - COUNT 1

# You should see a CloudEvent with type "com.storytelling.nlp.document.processed"
```

**Step 7: Parse Event (Optional)**

```bash
# Get and parse the event JSON
docker exec nlp-redis redis-cli XREVRANGE nlp-events + - COUNT 1 | grep "event" | jq
```

---

## Manual End-to-End Testing

### Single Document Processing

**Test 1: Process Short Document**

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "document": {
    "document_id": "e2e-test-001",
    "cleaned_text": "The Federal Reserve announced a 0.25% interest rate hike on Wednesday. Fed Chair Jerome Powell stated that the move aims to combat inflation.",
    "cleaned_title": "Fed Raises Rates",
    "cleaned_publication_date": "2025-12-14T10:00:00Z"
  }
}
EOF
```

**Expected response:**
```json
{
  "success": true,
  "document_id": "e2e-test-001",
  "result": {
    "document_id": "e2e-test-001",
    "events": [ /* extracted events */ ],
    "extracted_entities": [ /* entities */ ],
    "extracted_soa_triplets": [ /* triplets */ ]
  },
  "processing_time_ms": 12500.0
}
```

**Test 2: Check Output File**

```bash
# Check JSONL output
docker exec nlp-orchestrator tail -1 /app/data/extracted_events_$(date +%Y-%m-%d).jsonl | jq
```

### Batch Processing

**Test 3: Submit Batch Job**

```bash
# Create test batch file
cat > /tmp/test_batch.json <<'EOF'
{
  "documents": [
    {
      "document_id": "batch-doc-001",
      "cleaned_text": "Apple CEO Tim Cook announced new products.",
      "cleaned_title": "Apple Event",
      "cleaned_publication_date": "2025-12-14T10:00:00Z"
    },
    {
      "document_id": "batch-doc-002",
      "cleaned_text": "Microsoft released updates to Windows.",
      "cleaned_title": "Windows Update",
      "cleaned_publication_date": "2025-12-14T11:00:00Z"
    }
  ],
  "batch_id": "test-batch-001"
}
EOF

# Submit batch
curl -X POST http://localhost:8000/api/v1/documents/batch \
  -H "Content-Type: application/json" \
  -d @/tmp/test_batch.json
```

**Expected response:**
```json
{
  "success": true,
  "job_id": "celery-task-id-here",
  "batch_id": "test-batch-001",
  "document_count": 2,
  "message": "Batch submitted for processing"
}
```

**Test 4: Check Batch Status**

```bash
# Replace <job_id> with actual job ID from previous response
curl http://localhost:8000/api/v1/jobs/<job_id>/status

# Monitor Celery worker logs
docker logs nlp-celery-worker --tail 50 -f
```

**Test 5: Verify Batch Events (if enabled)**

```bash
# In Redis CLI
docker exec nlp-redis redis-cli XREAD COUNT 10 STREAMS nlp-events 0

# Look for:
# - batch.started event
# - document.processed events (2)
# - batch.completed event
```

---

## Development Workflow

### Make Code Changes

**Step 1: Edit Source Files on Host**

```bash
# Edit files in src/ directory
vim src/events/publisher.py
```

**Step 2: Copy Changes to Container**

```bash
# Copy specific file
docker cp src/events/publisher.py nlp-orchestrator:/app/src/events/

# Or copy entire src directory
docker cp src/ nlp-orchestrator:/app/
```

**Step 3: Restart Service**

```bash
# Restart to pick up changes
docker compose restart orchestrator

# Or for Celery worker
docker compose restart celery-worker
```

**Step 4: Test Changes**

```bash
# Run relevant tests
docker exec nlp-orchestrator pytest /app/tests/events/test_events_publisher.py -v

# Check logs
docker logs nlp-orchestrator --tail 30
```

### Rebuild Containers (for dependency changes)

```bash
# Rebuild specific service
docker compose build orchestrator

# Rebuild all
docker compose build

# Rebuild and restart
docker compose up -d --build
```

### Update Tests

**Step 1: Edit Tests on Host**

```bash
vim tests/events/test_events_publisher.py
```

**Step 2: Copy to Container**

```bash
docker cp tests/events/ nlp-orchestrator:/app/tests/
```

**Step 3: Run Updated Tests**

```bash
docker exec nlp-orchestrator pytest /app/tests/events/ -v
```

---

## Troubleshooting

### Services Won't Start

**Check logs:**
```bash
docker compose logs orchestrator
docker compose logs celery-worker
```

**Common issues:**
- Port conflicts: Check if ports 8000-8003, 6379 are available
- GPU issues: Check `nvidia-smi` and GPU drivers
- Memory: Ensure sufficient RAM (16GB minimum)

### Tests Failing

**Run with verbose output:**
```bash
docker exec nlp-orchestrator pytest /app/tests/events/ -vv --tb=long
```

**Check Python environment:**
```bash
docker exec nlp-orchestrator python -c "import src.events; print(src.events.__file__)"
```

**Verify imports:**
```bash
docker exec nlp-orchestrator python -c "from src.events.publisher import EventPublisher; print('OK')"
```

### Event Publishing Not Working

**Check configuration:**
```bash
docker exec nlp-orchestrator cat /app/config/settings.yaml | grep -A 20 "^events:"
```

**Check event publisher logs:**
```bash
docker logs nlp-orchestrator | grep -i "event"
```

**Verify Redis connection:**
```bash
docker exec nlp-orchestrator python -c "
import redis
r = redis.Redis.from_url('redis://redis:6379/1')
print(r.ping())
"
```

**Check stream exists:**
```bash
docker exec nlp-redis redis-cli XINFO STREAM nlp-events
```

### Performance Issues

**Check resource usage:**
```bash
docker stats --no-stream

# Expected for NLP pipeline:
# - orchestrator: ~1-2GB RAM
# - celery-worker: ~4-6GB RAM
# - llm-service: ~8-12GB RAM (with GPU)
```

**Check GPU utilization:**
```bash
docker exec nlp-event-llm-service nvidia-smi
```

---

## Quick Reference Commands

### Start/Stop

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# Restart specific service
docker compose restart orchestrator
```

### Logs

```bash
# Follow logs
docker logs nlp-orchestrator -f

# Last 50 lines
docker logs nlp-orchestrator --tail 50

# Since timestamp
docker logs nlp-orchestrator --since 5m
```

### Tests

```bash
# All event tests
docker exec nlp-orchestrator pytest /app/tests/events/ -v

# Single test file
docker exec nlp-orchestrator pytest /app/tests/events/test_events_models.py -v

# Stop at first failure
docker exec nlp-orchestrator pytest /app/tests/events/ -x
```

### Interactive Shell

```bash
# Python shell in orchestrator
docker exec -it nlp-orchestrator python

# Bash shell
docker exec -it nlp-orchestrator bash

# Redis CLI
docker exec -it nlp-redis redis-cli
```

### File Operations

```bash
# Copy file to container
docker cp local_file.py nlp-orchestrator:/app/src/

# Copy file from container
docker cp nlp-orchestrator:/app/config/settings.yaml ./

# View file in container
docker exec nlp-orchestrator cat /app/config/settings.yaml
```

---

## Test Results Summary

### Event System Tests

✅ **All 47 tests passing (100%)**

**Breakdown:**
- Models: 14 tests ✅
- Backends: 15 tests ✅
- Publisher: 18 tests ✅

**Test execution time:** ~0.2 seconds

**Coverage:**
- `src/events/models.py`: 100%
- `src/events/publisher.py`: 95%
- `src/events/backends/*.py`: 90%

---

## Next Steps

1. **Enable events** in production: Edit `config/settings.yaml`
2. **Implement Stage 3 consumer**: Use examples in `docs/EVENT_PUBLISHING_USAGE.md`
3. **Monitor production events**: Set up dashboards for Redis Streams
4. **Add more backends**: Kafka, NATS for high-throughput scenarios

---

## Support

For issues:
1. Check this guide first
2. Review logs: `docker logs <service-name>`
3. Check documentation: `docs/`
4. Verify configuration: `config/settings.yaml`
