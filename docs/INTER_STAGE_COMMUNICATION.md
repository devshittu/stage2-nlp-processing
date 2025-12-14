# Inter-Stage Communication Design

**Version**: 1.0
**Date**: December 2025
**Status**: Implementation in Progress

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Event Schema](#event-schema)
4. [Backend Implementations](#backend-implementations)
5. [Configuration](#configuration)
6. [Integration Points](#integration-points)
7. [Backward Compatibility](#backward-compatibility)
8. [Security Considerations](#security-considerations)
9. [Monitoring & Observability](#monitoring--observability)
10. [Future Enhancements](#future-enhancements)

---

## Overview

### Purpose

Enable asynchronous, event-driven communication between the 8-stage Sequential Storytelling Pipeline:

```
Stage 1: Cleaning → Stage 2: NLP → Stage 3: Embedding → Stage 4: Clustering →
Stage 5: Graph → Stage 6: Timeline → Stage 7: API → Stage 8: Frontend
```

### Goals

1. **Decoupling**: Stages operate independently with loose coupling
2. **Scalability**: Support horizontal scaling and high-throughput processing
3. **Reliability**: Guaranteed delivery with retry mechanisms
4. **Observability**: Full tracing and monitoring of events
5. **Flexibility**: Pluggable backends for different deployment scenarios
6. **Standards Compliance**: Use CloudEvents specification (CNCF)

### Non-Goals

- Synchronous request-response between stages (use APIs for that)
- Complex event processing (CEP) or event sourcing
- Real-time streaming to end users (handled by Stage 8)

---

## Architecture

### Design Pattern: Event-Driven Architecture (EDA)

Stage 2 publishes events when documents are processed. Downstream stages subscribe to these events.

```
┌─────────────────┐
│   Stage 1       │
│   (Cleaning)    │
└────────┬────────┘
         │ HTTP POST (documents)
         ▼
┌─────────────────────────────────────────────┐
│   Stage 2 (NLP Processing)                  │
│                                             │
│   ┌──────────────┐     ┌─────────────────┐ │
│   │ Orchestrator │────▶│ Event Publisher │ │
│   └──────────────┘     └────────┬────────┘ │
└─────────────────────────────────┼──────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │  Redis   │  │  Kafka   │  │ Webhook  │
              │ Streams  │  │          │  │          │
              └────┬─────┘  └────┬─────┘  └────┬─────┘
                   │             │             │
                   └─────────────┴─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Stage 3 (Embedding)   │
                    └─────────────────────────┘
```

### Components

1. **Event Publisher**: Core abstraction for publishing events
2. **Backend Adapters**: Pluggable implementations (Redis, Kafka, Webhooks, etc.)
3. **Event Schema**: CloudEvents-compliant event structure
4. **Configuration Manager**: YAML-based configuration for backends
5. **Integration Hooks**: Hooks in orchestrator for event publishing

---

## Event Schema

### CloudEvents Specification

We use [CloudEvents v1.0](https://cloudevents.io/) for interoperability.

#### Core Fields

```json
{
  "specversion": "1.0",
  "type": "com.storytelling.nlp.document.processed",
  "source": "stage2-nlp-processing",
  "id": "unique-event-id-uuid",
  "time": "2025-12-14T12:00:00Z",
  "datacontenttype": "application/json",
  "subject": "document/doc-123",
  "data": {
    // Event-specific payload
  }
}
```

#### Event Types

| Event Type | Trigger | Payload |
|------------|---------|---------|
| `com.storytelling.nlp.document.processed` | Single document completed | Document ID, output location, metrics |
| `com.storytelling.nlp.batch.started` | Batch processing begins | Job ID, document count, started timestamp |
| `com.storytelling.nlp.batch.completed` | Batch processing completes | Job ID, success/failure counts, duration |
| `com.storytelling.nlp.document.failed` | Document processing failed | Document ID, error details, retry count |

### Document Processed Event

```json
{
  "specversion": "1.0",
  "type": "com.storytelling.nlp.document.processed",
  "source": "stage2-nlp-processing",
  "id": "evt_a1b2c3d4e5f6",
  "time": "2025-12-14T12:00:00Z",
  "datacontenttype": "application/json",
  "subject": "document/doc-123",
  "data": {
    "document_id": "doc-123",
    "job_id": "job-456",
    "status": "success",
    "processing_time_seconds": 12.5,
    "output_location": {
      "jsonl": "file:///app/data/extracted_events_2025-12-14.jsonl",
      "postgres": "postgresql://db/documents/doc-123",
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
        "ner": "dslim/bert-base-NER",
        "dp": "benepar_en3",
        "event_extraction": "mistralai/Mistral-7B-Instruct-v0.2"
      }
    }
  }
}
```

### Batch Completed Event

```json
{
  "specversion": "1.0",
  "type": "com.storytelling.nlp.batch.completed",
  "source": "stage2-nlp-processing",
  "id": "evt_batch_789",
  "time": "2025-12-14T13:00:00Z",
  "datacontenttype": "application/json",
  "subject": "batch/job-456",
  "data": {
    "job_id": "job-456",
    "total_documents": 100,
    "successful": 98,
    "failed": 2,
    "duration_seconds": 3600,
    "started_at": "2025-12-14T12:00:00Z",
    "completed_at": "2025-12-14T13:00:00Z",
    "output_locations": {
      "jsonl": "file:///app/data/extracted_events_2025-12-14.jsonl"
    },
    "aggregate_metrics": {
      "total_events": 490,
      "total_entities": 2300,
      "avg_processing_time": 36.0
    }
  }
}
```

---

## Backend Implementations

### 1. Redis Streams (Primary - Default)

**Pros**:
- Already in infrastructure (used for caching)
- Low latency (<1ms)
- Simple setup
- Consumer groups for load balancing
- Persistence options

**Cons**:
- Not designed for massive scale (vs Kafka)
- Limited retention policies

**Configuration**:
```yaml
events:
  enabled: true
  backend: redis_streams
  redis_streams:
    url: "redis://redis:6379/1"
    stream_name: "nlp-events"
    max_len: 10000
    ttl_seconds: 86400  # 24 hours
```

**Implementation**:
```python
# src/events/backends/redis_streams.py
class RedisStreamsBackend(EventBackend):
    def publish(self, event: CloudEvent) -> bool:
        self.redis.xadd(
            self.stream_name,
            {"event": event.to_json()},
            maxlen=self.max_len
        )
```

### 2. Apache Kafka (High-Throughput)

**Pros**:
- Industry standard for event streaming
- Massive throughput (millions/sec)
- Strong durability guarantees
- Long retention (days/weeks)
- Rich ecosystem (Connect, Streams, etc.)

**Cons**:
- Operational complexity
- Higher resource requirements
- Overkill for small deployments

**Configuration**:
```yaml
events:
  enabled: true
  backend: kafka
  kafka:
    bootstrap_servers: ["kafka:9092"]
    topic: "nlp-document-events"
    compression_type: "gzip"
    acks: 1  # 0=none, 1=leader, -1=all
```

### 3. NATS (Cloud-Native)

**Pros**:
- Lightweight and fast
- Cloud-native design
- Built-in clustering
- JetStream for persistence

**Cons**:
- Newer, less mature ecosystem
- Requires NATS infrastructure

**Configuration**:
```yaml
events:
  enabled: true
  backend: nats
  nats:
    servers: ["nats://nats:4222"]
    subject: "nlp.document.processed"
    jetstream: true
```

### 4. RabbitMQ (Flexible Routing)

**Pros**:
- Flexible routing (exchanges, bindings)
- Strong guarantees (ACK, persistence)
- Well-established

**Cons**:
- Lower throughput vs Kafka
- More complex than Redis

**Configuration**:
```yaml
events:
  enabled: true
  backend: rabbitmq
  rabbitmq:
    url: "amqp://rabbitmq:5672"
    exchange: "nlp-events"
    routing_key: "document.processed"
```

### 5. Webhooks (HTTP Callbacks)

**Pros**:
- Simple integration for external systems
- No message broker required
- Standard HTTP/HTTPS

**Cons**:
- No built-in retry or persistence
- Requires webhook endpoint implementation
- Potential latency/failures

**Configuration**:
```yaml
events:
  enabled: true
  backend: webhook
  webhook:
    urls:
      - "https://stage3-embedding.example.com/webhook/nlp-events"
      - "https://monitoring.example.com/webhook/nlp-events"
    timeout_seconds: 5
    retry_attempts: 3
    retry_backoff: exponential
```

### 6. Multi-Backend Support

Publish to multiple backends simultaneously:

```yaml
events:
  enabled: true
  backends:  # Note: plural
    - type: redis_streams
      config: { ... }
    - type: webhook
      config: { ... }
```

---

## Configuration

### Full Configuration Schema

```yaml
# config/settings.yaml

events:
  # Global enable/disable (backward compatibility)
  enabled: false  # Default: disabled to prevent regressions

  # Single backend mode
  backend: redis_streams

  # OR Multi-backend mode
  # backends:
  #   - type: redis_streams
  #     config: { ... }

  # Event filtering
  publish_events:
    document_processed: true
    document_failed: true
    batch_started: true
    batch_completed: true

  # Redis Streams configuration
  redis_streams:
    url: "redis://redis:6379/1"
    stream_name: "nlp-events"
    max_len: 10000  # Maximum stream length
    ttl_seconds: 86400  # 24 hours
    connection_pool:
      max_connections: 10
      timeout: 5

  # Kafka configuration
  kafka:
    bootstrap_servers:
      - "kafka:9092"
    topic: "nlp-document-events"
    compression_type: "gzip"
    acks: 1
    retries: 3
    max_in_flight_requests: 5
    client_id: "stage2-nlp-producer"

  # NATS configuration
  nats:
    servers:
      - "nats://nats:4222"
    subject: "nlp.document.processed"
    jetstream: true
    stream: "NLP_EVENTS"
    durable_name: "nlp-processor"

  # RabbitMQ configuration
  rabbitmq:
    url: "amqp://guest:guest@rabbitmq:5672/"
    exchange: "nlp-events"
    exchange_type: "topic"
    routing_key: "document.processed"
    durable: true

  # Webhook configuration
  webhook:
    urls:
      - "https://stage3.example.com/webhook/nlp"
    headers:
      X-API-Key: "${WEBHOOK_API_KEY}"  # From environment
    timeout_seconds: 5
    retry_attempts: 3
    retry_backoff: exponential
    retry_delay_seconds: 1

  # Monitoring and observability
  monitoring:
    track_publish_latency: true
    log_events: true
    log_level: INFO  # DEBUG for detailed event payloads
```

### Environment Variables

```bash
# .env.dev
EVENTS_ENABLED=false
EVENTS_BACKEND=redis_streams
REDIS_EVENTS_URL=redis://redis:6379/1
WEBHOOK_API_KEY=secret-key-here
```

---

## Integration Points

### 1. Orchestrator Service

**File**: `src/api/orchestrator_service.py`

**Integration**: Publish events after document processing completion

```python
# After successful document processing
if config.events.enabled:
    event_publisher.publish_document_processed(
        document_id=doc.document_id,
        job_id=job_id,
        processing_time=processing_time,
        metrics=metrics,
        output_locations=output_locations
    )
```

### 2. Celery Tasks

**File**: `src/batch/celery_tasks.py`

**Integration**: Publish batch-level events

```python
# At batch start
if config.events.enabled:
    event_publisher.publish_batch_started(
        job_id=job_id,
        document_count=len(documents)
    )

# At batch completion
if config.events.enabled:
    event_publisher.publish_batch_completed(
        job_id=job_id,
        metrics=aggregate_metrics
    )
```

### 3. Error Handling

**Integration**: Publish failure events for observability

```python
# On document processing failure
if config.events.enabled:
    event_publisher.publish_document_failed(
        document_id=doc_id,
        error=str(error),
        retry_count=retry_count
    )
```

---

## Backward Compatibility

### Feature Flag Pattern

```python
# Default: disabled to prevent regressions
if not config.events.enabled:
    return  # No event publishing

# Only publish if explicitly enabled
```

### Graceful Degradation

```python
try:
    event_publisher.publish(event)
except Exception as e:
    logger.warning(f"Failed to publish event: {e}")
    # Continue processing - event publishing is non-critical
```

### Configuration Migration

Old deployments without `events:` section will work unchanged:

```python
# src/core/config.py
events: Optional[EventsConfig] = Field(
    default=EventsConfig(enabled=False),
    description="Event publishing configuration"
)
```

---

## Security Considerations

### 1. Authentication

**Webhooks**: API key in headers
```yaml
webhook:
  headers:
    X-API-Key: "${WEBHOOK_API_KEY}"
```

**Kafka**: SASL/SCRAM authentication
```yaml
kafka:
  security_protocol: SASL_SSL
  sasl_mechanism: SCRAM-SHA-256
  sasl_username: "${KAFKA_USERNAME}"
  sasl_password: "${KAFKA_PASSWORD}"
```

### 2. Encryption

**TLS for external webhooks**:
```yaml
webhook:
  urls:
    - "https://..."  # HTTPS required
  verify_ssl: true
```

### 3. Data Privacy

**Sensitive data filtering**:
```python
# Don't include full document text in events
data = {
    "document_id": doc_id,
    # "text": doc.text,  # NEVER include
    "output_location": location  # Reference only
}
```

### 4. Rate Limiting

**Prevent event flooding**:
```python
# Circuit breaker pattern
if event_publisher.failure_rate > 0.5:
    event_publisher.disable_temporarily()
```

---

## Monitoring & Observability

### Metrics to Track

1. **Publishing Metrics**:
   - Events published per second
   - Publish latency (p50, p95, p99)
   - Publish failures
   - Backend-specific errors

2. **Event Metrics**:
   - Events by type
   - Event payload size
   - Time from processing to publish

3. **Consumer Metrics** (Stage 3+):
   - Consumer lag
   - Processing time
   - Dead-letter queue size

### Logging

```python
logger.info(
    "Event published",
    extra={
        "event_id": event.id,
        "event_type": event.type,
        "document_id": event.data["document_id"],
        "backend": "redis_streams",
        "latency_ms": latency
    }
)
```

### Tracing

CloudEvents include correlation IDs for distributed tracing:

```json
{
  "traceparent": "00-trace-id-span-id-01",
  "data": {
    "job_id": "job-123",
    "document_id": "doc-456"
  }
}
```

---

## Future Enhancements

### Phase 2: Bi-directional Events

Stage 3 publishes back to Stage 2:

```
Stage 2 ←──(embedding.failed)──── Stage 3
```

Allows automatic retry/remediation.

### Phase 3: Event Replay

Support replaying events for reprocessing:

```python
event_publisher.replay(
    from_time="2025-12-01T00:00:00Z",
    to_time="2025-12-14T00:00:00Z",
    event_types=["document.processed"]
)
```

### Phase 4: Schema Registry

Use Confluent Schema Registry or similar for schema evolution:

```yaml
events:
  schema_registry:
    url: "http://schema-registry:8081"
    compatibility: BACKWARD
```

### Phase 5: Dead Letter Queue

Automatic routing of failed events:

```yaml
events:
  dlq:
    enabled: true
    backend: redis_streams
    stream_name: "nlp-events-dlq"
    max_retries: 3
```

---

## Implementation Checklist

- [ ] Create event publisher abstraction (`src/events/publisher.py`)
- [ ] Implement Redis Streams backend (`src/events/backends/redis_streams.py`)
- [ ] Implement Webhook backend (`src/events/backends/webhook.py`)
- [ ] Add configuration schema (`src/core/config.py`)
- [ ] Integrate into orchestrator service
- [ ] Integrate into Celery tasks
- [ ] Write unit tests (mocked backends)
- [ ] Write integration tests (real Redis)
- [ ] Update documentation
- [ ] Test in Docker environment
- [ ] Performance testing (latency impact)

---

## References

- [CloudEvents Specification v1.0](https://github.com/cloudevents/spec/blob/v1.0/spec.md)
- [Redis Streams](https://redis.io/docs/data-types/streams/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [NATS JetStream](https://docs.nats.io/nats-concepts/jetstream)
- [RabbitMQ Tutorials](https://www.rabbitmq.com/getstarted.html)
- [Event-Driven Architecture Pattern](https://martinfowler.com/articles/201701-event-driven.html)
