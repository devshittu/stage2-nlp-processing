# Event Publishing Usage Guide

**Version**: 1.0
**Last Updated**: December 2025

## Overview

The NLP Processing Service (Stage 2) publishes events to notify downstream stages (Stage 3+) when documents and batches are processed. This enables event-driven, asynchronous communication between pipeline stages.

### Key Features

- **CloudEvents Standard**: Industry-standard event format (CNCF specification)
- **Multiple Backends**: Redis Streams, Kafka, NATS, RabbitMQ, Webhooks
- **Backward Compatible**: Disabled by default, no breaking changes
- **Highly Configurable**: Fine-grained control over what gets published
- **Non-Critical**: Event publishing failures don't affect processing

### Event Types

| Event Type | When Published | Payload |
|------------|---------------|---------|
| `document.processed` | Single document completes successfully | Document ID, metrics, output locations |
| `document.failed` | Single document fails | Document ID, error details |
| `batch.started` | Batch processing begins | Job ID, document count |
| `batch.completed` | Batch processing finishes | Job ID, success/failure counts, aggregate metrics |

---

## Quick Start

### 1. Enable Event Publishing

Edit `config/settings.yaml`:

```yaml
events:
  enabled: true  # Change from false to true
  backend: "redis_streams"  # or "webhook", "kafka", etc.
```

### 2. Configure Backend (Redis Streams - Default)

```yaml
events:
  enabled: true
  backend: "redis_streams"

  redis_streams:
    url: "redis://redis:6379/1"
    stream_name: "nlp-events"
    max_len: 10000
    ttl_seconds: 86400  # 24 hours
```

### 3. Restart Services

```bash
docker compose restart orchestrator celery-worker
```

### 4. Verify Events Are Publishing

```bash
# Monitor Redis Stream
docker exec nlp-redis redis-cli XREAD COUNT 10 STREAMS nlp-events 0

# Check logs
docker logs nlp-orchestrator | grep "Event published"
```

---

## Configuration Guide

### Redis Streams Backend (Recommended)

**Best for**: Most deployments, low latency, simple setup

```yaml
events:
  enabled: true
  backend: "redis_streams"

  redis_streams:
    url: "redis://redis:6379/1"  # Same Redis as caching
    stream_name: "nlp-events"
    max_len: 10000  # Automatically trim old events
    ttl_seconds: 86400  # 24 hours retention
    connection_pool:
      max_connections: 10
      timeout: 5
```

**Consumer Example** (Stage 3+):

```python
import redis
import json

r = redis.Redis.from_url("redis://redis:6379/1")

# Create consumer group (run once)
r.xgroup_create("nlp-events", "stage3-embeddings", id="0", mkstream=True)

# Read events
while True:
    events = r.xreadgroup(
        "stage3-embeddings",
        "consumer-1",
        {"nlp-events": ">"},
        count=10,
        block=5000  # 5 second timeout
    )

    for stream_name, messages in events:
        for message_id, data in messages:
            event = json.loads(data[b"event"])

            if event["type"] == "com.storytelling.nlp.document.processed":
                document_id = event["data"]["document_id"]
                output_location = event["data"]["output_location"]["jsonl"]

                # Process document from Stage 2
                process_document_for_embedding(document_id, output_location)

                # Acknowledge message
                r.xack("nlp-events", "stage3-embeddings", message_id)
```

### Webhook Backend

**Best for**: HTTP-based integrations, external systems

```yaml
events:
  enabled: true
  backend: "webhook"

  webhook:
    urls:
      - "https://stage3-embedding:8000/webhook/nlp-events"
      - "https://monitoring.example.com/webhook/nlp-events"
    headers:
      X-API-Key: "${WEBHOOK_API_KEY}"  # From environment
      Content-Type: "application/cloudevents+json"
    timeout_seconds: 5
    retry_attempts: 3
    retry_backoff: "exponential"
    retry_delay_seconds: 1
    verify_ssl: true
```

**Webhook Endpoint Example** (Stage 3+):

```python
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class CloudEvent(BaseModel):
    specversion: str
    type: str
    source: str
    id: str
    data: dict

@app.post("/webhook/nlp-events")
async def receive_nlp_event(event: CloudEvent):
    """Receive events from Stage 2 NLP Processing."""

    if event.type == "com.storytelling.nlp.document.processed":
        document_id = event.data["document_id"]
        metrics = event.data["metrics"]

        print(f"Document {document_id} processed: {metrics['event_count']} events")

        # Trigger Stage 3 processing
        await trigger_embedding_generation(document_id)

    elif event.type == "com.storytelling.nlp.batch.completed":
        job_id = event.data["job_id"]
        total_docs = event.data["total_documents"]
        successful = event.data["successful"]

        print(f"Batch {job_id} completed: {successful}/{total_docs} successful")

    return {"status": "received"}
```

### Kafka Backend

**Best for**: High-throughput production deployments, millions of events/day

```yaml
events:
  enabled: true
  backend: "kafka"

  kafka:
    bootstrap_servers:
      - "kafka-1:9092"
      - "kafka-2:9092"
      - "kafka-3:9092"
    topic: "nlp-document-events"
    compression_type: "gzip"
    acks: 1  # 0=none, 1=leader, -1=all replicas
    retries: 3
    max_in_flight_requests: 5
    client_id: "stage2-nlp-producer"
```

**Consumer Example**:

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'nlp-document-events',
    bootstrap_servers=['kafka-1:9092'],
    group_id='stage3-embeddings',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    event = message.value

    if event["type"] == "com.storytelling.nlp.document.processed":
        document_id = event["data"]["document_id"]
        process_for_embedding(document_id)
```

### Event Filtering

Control which events to publish:

```yaml
events:
  enabled: true
  backend: "redis_streams"

  publish_events:
    document_processed: true   # Publish successful completions
    document_failed: false     # Don't publish failures
    batch_started: true        # Publish batch starts
    batch_completed: true      # Publish batch completions
```

---

## Event Schema Reference

### CloudEvent Structure

All events follow the [CloudEvents v1.0](https://cloudevents.io/) specification:

```json
{
  "specversion": "1.0",
  "type": "com.storytelling.nlp.document.processed",
  "source": "stage2-nlp-processing",
  "id": "evt_a1b2c3d4e5f6",
  "time": "2025-12-14T12:00:00Z",
  "datacontenttype": "application/json",
  "subject": "document/doc-123",
  "data": { /* event-specific payload */ }
}
```

### document.processed Event

```json
{
  "type": "com.storytelling.nlp.document.processed",
  "subject": "document/doc-123",
  "data": {
    "document_id": "doc-123",
    "job_id": "job-456",
    "status": "success",
    "processing_time_seconds": 12.5,
    "output_location": {
      "jsonl": "file:///app/data/extracted_events_2025-12-14.jsonl",
      "postgresql": "postgresql://db/documents/doc-123",
      "elasticsearch": "http://es:9200/documents/doc-123"
    },
    "metrics": {
      "event_count": 5,
      "entity_count": 23,
      "soa_triplet_count": 12
    },
    "metadata": {
      "pipeline_version": "1.0.0",
      "model_versions": {
        "ner": "Babelscape/wikineural-multilingual-ner",
        "dp": "en_core_web_trf",
        "event_extraction": "mistralai/Mistral-7B-Instruct-v0.3"
      }
    }
  }
}
```

### batch.completed Event

```json
{
  "type": "com.storytelling.nlp.batch.completed",
  "subject": "batch/job-456",
  "data": {
    "job_id": "job-456",
    "total_documents": 100,
    "successful": 98,
    "failed": 2,
    "duration_seconds": 3600.0,
    "started_at": "2025-12-14T12:00:00Z",
    "completed_at": "2025-12-14T13:00:00Z",
    "output_locations": {
      "jsonl": "file:///app/data/extracted_events_2025-12-14.jsonl"
    },
    "aggregate_metrics": {
      "total_events": 490,
      "total_entities": 2300,
      "linkages_count": 125,
      "storylines_count": 8,
      "avg_processing_time_ms": 36000
    }
  }
}
```

---

## Advanced Usage

### Environment Variables

Override configuration with environment variables:

```bash
# .env.prod
EVENTS_ENABLED=true
EVENTS_BACKEND=redis_streams
REDIS_EVENTS_URL=redis://redis-prod:6379/1
WEBHOOK_API_KEY=prod-secret-key-xyz
```

### Multiple Backends

**NEW**: Publish to multiple backends simultaneously for improved reliability and flexibility.

**Use Cases**:
- **Reliability**: If one backend fails, events still reach the other
- **Migration**: Dual-publish during backend transitions
- **Monitoring**: Send events to both processing system and monitoring service
- **Integration**: Support different consumer requirements simultaneously

**Configuration**:

```yaml
events:
  enabled: true
  backends:  # Note: plural - takes precedence over 'backend'
    - "redis_streams"
    - "webhook"

  # Configure both backends
  redis_streams:
    url: "redis://redis:6379/1"
    stream_name: "nlp-events"
    max_len: 10000
    ttl_seconds: 86400

  webhook:
    urls:
      - "https://stage3-embedding:8000/webhook/nlp-events"
      - "https://monitoring.example.com/webhook/nlp-events"
    headers:
      X-API-Key: "${WEBHOOK_API_KEY}"
    timeout_seconds: 5
    retry_attempts: 3
```

**How It Works**:
- Events publish to **all** backends in parallel
- **Error isolation**: Failure in one backend doesn't affect others
- **At-least-one semantics**: Succeeds if any backend succeeds
- **Comprehensive logging**: All backend results logged for observability

**Example Scenarios**:

1. **Primary + Backup**: Redis Streams (fast, local) + Webhooks (backup, remote)
2. **Processing + Monitoring**: Redis (for Stage 3 processing) + Webhook (for metrics dashboard)
3. **Multi-consumer**: Redis Streams (internal) + Kafka (external partners)

### Monitoring Event Publishing

Track event publishing metrics:

```python
from src.events.publisher import create_event_publisher
from src.utils.config_manager import get_settings

config = get_settings()
publisher = create_event_publisher(config)

# After processing some documents...
metrics = publisher.get_metrics()

print(f"Events published: {metrics['events_published']}")
print(f"Events failed: {metrics['events_failed']}")
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Avg latency: {metrics['avg_latency_ms']:.2f}ms")
```

---

## Hands-On Tutorial

This section provides practical, step-by-step exercises to help you master all event publishing features.

### Exercise 1: Basic Event Publishing with Redis Streams

**Goal**: Enable event publishing and verify events are being published.

**Steps**:

1. **Enable events** in `config/settings.yaml`:
   ```yaml
   events:
     enabled: true
     backend: "redis_streams"
   ```

2. **Restart services**:
   ```bash
   docker compose restart orchestrator celery-worker
   ```

3. **Process a test document**:
   ```bash
   # Submit a job
   curl -X POST "http://localhost:8000/api/v1/process-text" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "John Smith met Mary Johnson in New York on Monday.",
       "job_id": "test-job-1"
     }'
   ```

4. **Verify events in Redis**:
   ```bash
   # Read latest events from stream
   docker exec nlp-redis redis-cli XREAD COUNT 5 STREAMS nlp-events 0
   ```

5. **Check logs for confirmation**:
   ```bash
   docker logs nlp-celery-worker | grep "Event published"
   ```

**Expected Output**: You should see CloudEvent messages in the Redis stream containing `document.processed` event type.

---

### Exercise 2: Webhook Integration

**Goal**: Set up webhook backend to receive events via HTTP.

**Steps**:

1. **Create a simple webhook receiver** (`test_webhook.py`):
   ```python
   from flask import Flask, request, jsonify

   app = Flask(__name__)

   @app.route('/webhook/nlp-events', methods=['POST'])
   def receive_event():
       event = request.json
       print(f"\n{'='*60}")
       print(f"Received Event: {event['type']}")
       print(f"Document ID: {event.get('data', {}).get('document_id', 'N/A')}")
       print(f"Event ID: {event['id']}")
       print(f"{'='*60}\n")
       return jsonify({"status": "received"}), 200

   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=9000)
   ```

2. **Run webhook receiver** (in a separate terminal):
   ```bash
   python test_webhook.py
   ```

3. **Configure webhook backend** in `config/settings.yaml`:
   ```yaml
   events:
     enabled: true
     backend: "webhook"

     webhook:
       urls:
         - "http://host.docker.internal:9000/webhook/nlp-events"  # For Docker Desktop
         # OR for Linux: - "http://172.17.0.1:9000/webhook/nlp-events"
       timeout_seconds: 5
       retry_attempts: 2
   ```

4. **Restart and test**:
   ```bash
   docker compose restart orchestrator celery-worker

   # Submit test job
   curl -X POST "http://localhost:8000/api/v1/process-text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Test webhook integration.", "job_id": "webhook-test-1"}'
   ```

5. **Verify webhook receiver logs**: You should see the event details printed.

---

### Exercise 3: Multi-Backend Setup (Redis + Webhooks)

**Goal**: Publish to both Redis Streams and Webhooks simultaneously, demonstrating resilience.

**Steps**:

1. **Configure multi-backend** in `config/settings.yaml`:
   ```yaml
   events:
     enabled: true
     backends:  # Plural!
       - "redis_streams"
       - "webhook"

     redis_streams:
       url: "redis://redis:6379/1"
       stream_name: "nlp-events"
       max_len: 10000

     webhook:
       urls:
         - "http://host.docker.internal:9000/webhook/nlp-events"
       timeout_seconds: 5
       retry_attempts: 2
   ```

2. **Start webhook receiver** (from Exercise 2):
   ```bash
   python test_webhook.py
   ```

3. **Restart services**:
   ```bash
   docker compose restart orchestrator celery-worker
   ```

4. **Process a document**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/process-text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Alice visited Paris yesterday.", "job_id": "multi-backend-test"}'
   ```

5. **Verify both backends received events**:

   **Redis**:
   ```bash
   docker exec nlp-redis redis-cli XREAD COUNT 1 STREAMS nlp-events 0
   ```

   **Webhook**: Check the Python webhook receiver output

6. **Check multi-backend logs**:
   ```bash
   docker logs nlp-celery-worker | grep "Multi-backend publish"
   ```

   You should see: `Multi-backend publish complete: 2/2 succeeded`

---

### Exercise 4: Testing Resilience (One Backend Fails)

**Goal**: Demonstrate that if one backend fails, events still publish to the other.

**Steps**:

1. **Keep multi-backend configuration** from Exercise 3.

2. **Stop the webhook receiver** (Ctrl+C in the terminal running `test_webhook.py`).

3. **Process another document**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/process-text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Resilience test document.", "job_id": "resilience-test"}'
   ```

4. **Check logs**:
   ```bash
   docker logs nlp-celery-worker --tail 50 | grep -A 5 "Multi-backend"
   ```

   You should see:
   - `Failed to publish to WebhookBackend` (or similar error)
   - `Multi-backend publish complete: 1/2 succeeded`
   - Overall publish still **succeeds** (returns True)

5. **Verify Redis still received the event**:
   ```bash
   docker exec nlp-redis redis-cli XREAD COUNT 1 STREAMS nlp-events 0
   ```

**Key Learning**: Document processing continues successfully even when one backend fails. Events are still delivered via the working backend(s).

---

### Exercise 5: Multiple Webhook Endpoints

**Goal**: Send events to multiple different webhook URLs simultaneously.

**Steps**:

1. **Create two webhook receivers** on different ports:

   **Receiver 1** (`webhook_receiver_1.py`):
   ```python
   from flask import Flask, request, jsonify
   app = Flask(__name__)

   @app.route('/webhook/nlp-events', methods=['POST'])
   def receive():
       event = request.json
       print(f"[RECEIVER 1] Got event: {event['type']} - {event['id']}")
       return jsonify({"status": "received"}), 200

   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=9001)
   ```

   **Receiver 2** (`webhook_receiver_2.py`):
   ```python
   from flask import Flask, request, jsonify
   app = Flask(__name__)

   @app.route('/webhook/nlp-events', methods=['POST'])
   def receive():
       event = request.json
       print(f"[RECEIVER 2] Got event: {event['type']} - {event['id']}")
       return jsonify({"status": "received"}), 200

   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=9002)
   ```

2. **Run both receivers** (in separate terminals):
   ```bash
   # Terminal 1
   python webhook_receiver_1.py

   # Terminal 2
   python webhook_receiver_2.py
   ```

3. **Configure multiple webhook URLs** in `config/settings.yaml`:
   ```yaml
   events:
     enabled: true
     backend: "webhook"  # Single backend with multiple URLs

     webhook:
       urls:
         - "http://host.docker.internal:9001/webhook/nlp-events"
         - "http://host.docker.internal:9002/webhook/nlp-events"
       timeout_seconds: 5
       retry_attempts: 2
   ```

4. **Restart and test**:
   ```bash
   docker compose restart orchestrator celery-worker

   curl -X POST "http://localhost:8000/api/v1/process-text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Multiple webhooks test.", "job_id": "multi-webhook-test"}'
   ```

5. **Verify both receivers got the event**: Check both terminal outputs.

---

### Exercise 6: Event Filtering

**Goal**: Control which events are published to reduce noise.

**Steps**:

1. **Configure event filtering** in `config/settings.yaml`:
   ```yaml
   events:
     enabled: true
     backend: "redis_streams"

     publish_events:
       document_processed: true   # Publish
       document_failed: false     # Don't publish
       batch_started: false       # Don't publish
       batch_completed: true      # Publish
   ```

2. **Restart services**:
   ```bash
   docker compose restart orchestrator celery-worker
   ```

3. **Clear Redis stream** (for clean test):
   ```bash
   docker exec nlp-redis redis-cli DEL nlp-events
   ```

4. **Process a batch**:
   ```bash
   # This will create both document.processed and batch.started events
   curl -X POST "http://localhost:8000/api/v1/process-text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Filtering test.", "job_id": "filter-test-1"}'
   ```

5. **Check which events were published**:
   ```bash
   docker exec nlp-redis redis-cli XREAD COUNT 10 STREAMS nlp-events 0
   ```

   You should see:
   - ✅ `document.processed` events
   - ✅ `batch.completed` events
   - ❌ NO `document.failed` events
   - ❌ NO `batch.started` events

---

### Exercise 7: Consuming Events (Building a Simple Consumer)

**Goal**: Build a consumer that reads and processes events from Redis Streams.

**Steps**:

1. **Create consumer script** (`event_consumer.py`):
   ```python
   import redis
   import json
   import time

   # Connect to Redis
   r = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)

   # Create consumer group (run once, may fail if already exists)
   try:
       r.xgroup_create("nlp-events", "test-consumers", id="0", mkstream=True)
       print("Created consumer group: test-consumers")
   except redis.exceptions.ResponseError as e:
       print(f"Consumer group might already exist: {e}")

   print("\nListening for events...")
   print("=" * 60)

   # Read events
   while True:
       events = r.xreadgroup(
           "test-consumers",
           "consumer-1",
           {"nlp-events": ">"},
           count=10,
           block=5000  # 5 second timeout
       )

       for stream_name, messages in events:
           for message_id, data in messages:
               # Parse event
               event = json.loads(data["event"])

               print(f"\n📨 New Event Received:")
               print(f"  Type: {event['type']}")
               print(f"  ID: {event['id']}")
               print(f"  Time: {event['time']}")

               if event['type'] == 'com.storytelling.nlp.document.processed':
                   doc_data = event['data']
                   print(f"  Document: {doc_data['document_id']}")
                   print(f"  Events: {doc_data['metrics']['event_count']}")
                   print(f"  Entities: {doc_data['metrics']['entity_count']}")

               # Acknowledge message
               r.xack("nlp-events", "test-consumers", message_id)
               print(f"  ✅ Acknowledged: {message_id}")

       time.sleep(1)
   ```

2. **Run consumer**:
   ```bash
   python event_consumer.py
   ```

3. **In another terminal, trigger some events**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/process-text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Consumer test: Alice met Bob in Paris.", "job_id": "consumer-test"}'
   ```

4. **Watch consumer output**: You should see events being consumed and acknowledged in real-time.

5. **Verify consumer group state**:
   ```bash
   docker exec nlp-redis redis-cli XINFO GROUPS nlp-events
   docker exec nlp-redis redis-cli XINFO CONSUMERS nlp-events test-consumers
   ```

---

### Exercise 8: Using Event Publishing in Custom Code

**Goal**: Integrate event publishing into your own scripts or applications.

**Steps**:

1. **Create a custom processing script** (`custom_processor.py`):
   ```python
   from src.events.publisher import create_event_publisher
   from src.utils.config_manager import get_settings

   # Load configuration
   config = get_settings()

   # Create publisher
   publisher = create_event_publisher(config)

   # Publish a custom event
   result = publisher.publish_document_processed(
       document_id="custom-doc-123",
       job_id="custom-job-456",
       processing_time_seconds=5.2,
       output_locations={
           "jsonl": "file:///app/data/custom_output.jsonl"
       },
       metrics={
           "event_count": 10,
           "entity_count": 25,
           "soa_triplet_count": 8
       },
       metadata={
           "custom_field": "custom_value",
           "processor": "custom_script"
       }
   )

   if result:
       print("✅ Event published successfully!")
   else:
       print("❌ Event publishing failed")

   # Get metrics
   metrics = publisher.get_metrics()
   print(f"\nPublisher Metrics:")
   print(f"  Published: {metrics['events_published']}")
   print(f"  Failed: {metrics['events_failed']}")
   print(f"  Success Rate: {metrics['success_rate']:.2%}")
   print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f}ms")

   # Close publisher
   publisher.close()
   ```

2. **Run your custom script**:
   ```bash
   docker exec nlp-orchestrator python /app/custom_processor.py
   ```

3. **Verify the event was published**:
   ```bash
   docker exec nlp-redis redis-cli XREAD COUNT 1 STREAMS nlp-events 0
   ```

---

### Exercise 9: Independent Pipeline Usage

**Goal**: Use event publishing for a different pipeline, not just the 8-stage NLP pipeline.

**Steps**:

1. **Create a separate event publisher** for a different use case (`image_processor.py`):
   ```python
   from src.events.backends.redis_streams import RedisStreamsBackend
   from src.events.publisher import EventPublisher
   from src.events.models import CloudEvent

   # Create standalone publisher (independent of NLP config)
   backend = RedisStreamsBackend(
       url="redis://localhost:6379/1",
       stream_name="image-processing-events",  # Different stream!
       max_len=5000,
       ttl_seconds=3600
   )

   publisher = EventPublisher(backend=backend, enabled=True)

   # Publish custom event for image processing pipeline
   event = CloudEvent(
       type="com.example.image.processed",
       subject="image/img-789",
       data={
           "image_id": "img-789",
           "format": "JPEG",
           "dimensions": {"width": 1920, "height": 1080},
           "processing_time_ms": 250,
           "operations": ["resize", "compress", "watermark"]
       }
   )

   result = publisher.publish(event)
   print(f"Published image event: {result}")

   publisher.close()
   ```

2. **Run the independent script**:
   ```bash
   python image_processor.py
   ```

3. **Verify event in separate stream**:
   ```bash
   docker exec nlp-redis redis-cli XREAD COUNT 1 STREAMS image-processing-events 0
   ```

**Key Learning**: Event publishing infrastructure can be reused for any pipeline, not just NLP. Just create your own backend instance with different configuration.

---

### Exercise 10: Performance Monitoring

**Goal**: Monitor event publishing performance and identify bottlenecks.

**Steps**:

1. **Enable detailed logging** in `config/settings.yaml`:
   ```yaml
   events:
     enabled: true
     backend: "redis_streams"

     monitoring:
       track_publish_latency: true
       log_events: true
       log_level: "DEBUG"  # Detailed logging
   ```

2. **Restart services**:
   ```bash
   docker compose restart orchestrator celery-worker
   ```

3. **Process multiple documents**:
   ```bash
   for i in {1..20}; do
     curl -X POST "http://localhost:8000/api/v1/process-text" \
       -H "Content-Type: application/json" \
       -d "{\"text\": \"Performance test document $i\", \"job_id\": \"perf-test-$i\"}"
   done
   ```

4. **Analyze latency**:
   ```bash
   docker logs nlp-celery-worker | grep "latency_ms" | tail -20
   ```

5. **Check publisher metrics** via Python:
   ```python
   from src.events.publisher import create_event_publisher
   from src.utils.config_manager import get_settings

   config = get_settings()
   publisher = create_event_publisher(config)

   # Process some documents...

   metrics = publisher.get_metrics()
   print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")
   print(f"Success rate: {metrics['success_rate']:.2%}")
   ```

---

## Summary of Hands-On Exercises

| Exercise | Feature | Difficulty |
|----------|---------|------------|
| 1 | Basic Redis Streams | ⭐ Beginner |
| 2 | Webhook Integration | ⭐⭐ Intermediate |
| 3 | Multi-Backend Setup | ⭐⭐ Intermediate |
| 4 | Resilience Testing | ⭐⭐⭐ Advanced |
| 5 | Multiple Webhooks | ⭐⭐ Intermediate |
| 6 | Event Filtering | ⭐ Beginner |
| 7 | Building Consumers | ⭐⭐⭐ Advanced |
| 8 | Custom Integration | ⭐⭐ Intermediate |
| 9 | Independent Pipeline | ⭐⭐⭐ Advanced |
| 10 | Performance Monitoring | ⭐⭐ Intermediate |

**Recommended Learning Path**:
1. Start with Exercise 1 (Basic)
2. Try Exercise 6 (Filtering)
3. Move to Exercise 2 (Webhooks)
4. Practice Exercise 3 (Multi-Backend)
5. Test Exercise 4 (Resilience)
6. Build Exercise 7 (Consumer)
7. Explore Exercises 8, 9, 10 as needed

---

## Troubleshooting

### Events Not Being Published

1. **Check if events are enabled**:
   ```bash
   docker exec nlp-orchestrator grep "events:" /app/config/settings.yaml -A 2
   ```
   Ensure `enabled: true`

2. **Check logs for errors**:
   ```bash
   docker logs nlp-orchestrator | grep -i "event"
   docker logs nlp-celery-worker | grep -i "event"
   ```

3. **Verify backend connectivity**:
   ```bash
   # For Redis Streams
   docker exec nlp-redis redis-cli PING

   # For Webhooks
   curl -X POST https://your-webhook-url/webhook/test
   ```

### High Latency

Event publishing adds minimal latency (<5ms for Redis Streams), but if you notice issues:

1. **Check backend performance**:
   - Redis Streams: Monitor `SLOWLOG GET 10`
   - Webhooks: Check webhook endpoint response times

2. **Adjust connection pool**:
   ```yaml
   redis_streams:
     connection_pool:
       max_connections: 20  # Increase from 10
   ```

3. **Disable non-critical events**:
   ```yaml
   publish_events:
     document_failed: false  # Only publish successes
   ```

### Events Not Reaching Consumers

1. **For Redis Streams**:
   ```bash
   # Check stream exists and has messages
   docker exec nlp-redis redis-cli XINFO STREAM nlp-events

   # Check consumer group
   docker exec nlp-redis redis-cli XINFO GROUPS nlp-events
   ```

2. **For Webhooks**:
   - Check webhook endpoint logs
   - Verify SSL certificates if using HTTPS
   - Check firewall rules

---

## Performance Impact

Event publishing is designed to be non-critical and lightweight:

| Backend | Latency | Throughput | Notes |
|---------|---------|------------|-------|
| Redis Streams | <1ms | 50,000/sec | Minimal impact |
| Webhooks | 5-50ms | 1,000/sec | Depends on endpoint |
| Kafka | 1-5ms | 1M/sec | Production-grade |

**Impact on Document Processing**:
- Single document: +1-2ms (0.1-0.2% overhead)
- Batch processing: Negligible (async)

---

## Security Considerations

### Authentication

**Webhooks**: Use API keys in headers
```yaml
webhook:
  headers:
    X-API-Key: "${WEBHOOK_API_KEY}"
```

**Kafka**: Use SASL authentication
```yaml
kafka:
  security_protocol: SASL_SSL
  sasl_mechanism: SCRAM-SHA-256
  sasl_username: "${KAFKA_USERNAME}"
  sasl_password: "${KAFKA_PASSWORD}"
```

### Data Privacy

- Events **do not** include full document text
- Only metadata and references (document IDs, output locations)
- For sensitive deployments, use encrypted connections (TLS/SSL)

---

## Migration Guide

### Enabling for Existing Deployments

1. **Update configuration** (`config/settings.yaml`):
   ```yaml
   events:
     enabled: true
   ```

2. **Restart services** (no code changes needed):
   ```bash
   docker compose restart orchestrator celery-worker
   ```

3. **Verify** events are publishing:
   ```bash
   docker logs nlp-orchestrator | grep "Event published"
   ```

4. **Implement consumers** in downstream stages (Stage 3+)

### Disabling Event Publishing

Simply set `enabled: false` in configuration:

```yaml
events:
  enabled: false
```

No code changes required - backward compatible.

---

## Examples

See full working examples in:
- `docs/INTER_STAGE_COMMUNICATION.md` - Architecture and design
- `tests/events/` - Unit tests with usage patterns
- Consumer examples above for each backend

---

## Support

For issues or questions:
- Check logs: `docker logs nlp-orchestrator`
- Review design doc: `docs/INTER_STAGE_COMMUNICATION.md`
- Test connectivity to your backend
- Verify configuration syntax in `config/settings.yaml`
