# Quick Test Guide - Inter-Stage Communication

## ✅ Test Status: 100% Pass Rate (47/47 tests)

---

## Quick Start (3 Steps)

### 1. Start Services

```bash
cd /home/mshittu/projects/nlp/stage2-nlp-processing
docker compose up -d
docker compose ps  # Verify all services are "Up" and "healthy"
```

### 2. Run All Event Tests

```bash
# Run all 47 event system tests
docker exec nlp-orchestrator pytest /app/tests/events/ -v

# Expected: ======================== 47 passed in X.XXs ========================
```

### 3. Verify Services

```bash
# Test API health
curl http://localhost:8000/api/v1/health

# Check event publisher status (should show "disabled" by default)
docker logs nlp-orchestrator --tail 20 | grep "Event publisher"
```

---

## Testing Event Publishing (Optional)

### Enable Events

```bash
# Copy config, edit, and copy back
docker cp nlp-orchestrator:/app/config/settings.yaml ./config/settings.yaml

# Edit settings.yaml - change events.enabled from false to true
# Then copy back
docker cp ./config/settings.yaml nlp-orchestrator:/app/config/settings.yaml

# Restart services
docker compose restart orchestrator celery-worker
```

### Monitor Events

```bash
# Terminal 1: Monitor Redis stream
docker exec -it nlp-redis redis-cli
> XREAD COUNT 10 STREAMS nlp-events 0

# Terminal 2: Send test document
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "document": {
      "document_id": "test-001",
      "cleaned_text": "Apple CEO Tim Cook met with Microsoft CEO Satya Nadella.",
      "cleaned_title": "Test",
      "cleaned_publication_date": "2025-12-14T10:00:00Z"
    }
  }'

# Terminal 1: See event appear in Redis
```

---

## Test Categories

### Unit Tests (47 tests total)

```bash
# All event tests (47 tests)
docker exec nlp-orchestrator pytest /app/tests/events/ -v

# Models only (14 tests)
docker exec nlp-orchestrator pytest /app/tests/events/test_events_models.py -v

# Backends only (15 tests)
docker exec nlp-orchestrator pytest /app/tests/events/test_events_backends.py -v

# Publisher only (18 tests)
docker exec nlp-orchestrator pytest /app/tests/events/test_events_publisher.py -v
```

### Integration Tests

```bash
# Single document processing
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d @test_data/short_articles.json

# Batch processing
curl -X POST http://localhost:8000/api/v1/documents/batch \
  -H "Content-Type: application/json" \
  -d @test_data/batch_documents.json
```

---

## Test Results

✅ **Models Tests**: 14/14 passing
- CloudEvent creation and serialization
- Event type enums
- Data model validation

✅ **Backend Tests**: 15/15 passing
- NullBackend (no-op)
- RedisStreamsBackend (initialization, publishing, consumer groups)
- WebhookBackend (HTTP callbacks, retries, headers)

✅ **Publisher Tests**: 18/18 passing
- Event publishing (enabled/disabled)
- Convenience methods (document.processed, batch.completed, etc.)
- Metrics tracking
- Factory functions

---

## Development Workflow

### Make Code Changes

```bash
# 1. Edit file on host
vim src/events/publisher.py

# 2. Copy to container
docker cp src/events/publisher.py nlp-orchestrator:/app/src/events/

# 3. Restart service
docker compose restart orchestrator

# 4. Run tests
docker exec nlp-orchestrator pytest /app/tests/events/ -v
```

### Update Tests

```bash
# 1. Edit test on host
vim tests/events/test_events_publisher.py

# 2. Copy to container
docker cp tests/events/ nlp-orchestrator:/app/tests/

# 3. Run updated tests
docker exec nlp-orchestrator pytest /app/tests/events/ -v
```

---

## Useful Commands

### Logs

```bash
# Orchestrator logs
docker logs nlp-orchestrator --tail 50

# Celery worker logs
docker logs nlp-celery-worker --tail 50

# Follow logs
docker logs nlp-orchestrator -f
```

### Redis Operations

```bash
# Open Redis CLI
docker exec -it nlp-redis redis-cli

# Check stream
> XINFO STREAM nlp-events
> XLEN nlp-events
> XREVRANGE nlp-events + - COUNT 5

# Clear stream (for testing)
> DEL nlp-events
```

### Service Management

```bash
# Restart specific service
docker compose restart orchestrator

# Rebuild and restart
docker compose up -d --build orchestrator

# Stop all
docker compose down

# Start fresh
docker compose down && docker compose up -d
```

---

## Troubleshooting

### Tests Failing?

```bash
# Run with detailed output
docker exec nlp-orchestrator pytest /app/tests/events/ -vv --tb=long

# Run single test
docker exec nlp-orchestrator pytest /app/tests/events/test_events_models.py::TestCloudEvent::test_create_cloud_event -v
```

### Events Not Publishing?

```bash
# Check config
docker exec nlp-orchestrator cat /app/config/settings.yaml | grep -A 5 "^events:"

# Check logs
docker logs nlp-orchestrator | grep -i "event"

# Test Redis connection
docker exec nlp-orchestrator python -c "import redis; r=redis.Redis.from_url('redis://redis:6379/1'); print(r.ping())"
```

### Services Not Starting?

```bash
# Check status
docker compose ps

# Check logs
docker compose logs orchestrator

# Restart
docker compose restart
```

---

## Documentation

- **Full Testing Guide**: `docs/DEVELOPMENT_TESTING_GUIDE.md`
- **Usage Guide**: `docs/EVENT_PUBLISHING_USAGE.md`
- **Architecture**: `docs/INTER_STAGE_COMMUNICATION.md`

---

## Summary

✅ **Status**: All systems operational
✅ **Tests**: 47/47 passing (100%)
✅ **Services**: All healthy
✅ **Event System**: Ready for production (disabled by default)
✅ **Documentation**: Complete

**Next Steps**:
1. Review `docs/DEVELOPMENT_TESTING_GUIDE.md` for detailed instructions
2. Enable events when ready: `events.enabled: true` in settings.yaml
3. Implement Stage 3 consumer using examples in `docs/EVENT_PUBLISHING_USAGE.md`
