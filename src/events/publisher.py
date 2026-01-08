"""
Event publisher for inter-stage communication.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .backends.base import EventBackend, NullBackend
from .models import (
    BatchCompletedData,
    BatchStartedData,
    CloudEvent,
    DocumentFailedData,
    DocumentProcessedData,
    EventType,
)

logger = logging.getLogger(__name__)


class EventPublisher:
    """
    High-level event publisher for NLP pipeline events.

    Provides convenient methods for publishing domain-specific events
    while abstracting backend implementation details.
    """

    def __init__(self, backend: EventBackend, enabled: bool = True):
        """
        Initialize event publisher.

        Args:
            backend: Event backend implementation
            enabled: Whether event publishing is enabled
        """
        self.backend = backend
        self.enabled = enabled

        if not enabled:
            logger.info("Event publishing is disabled")
            self.backend = NullBackend()
        else:
            logger.info(
                f"Event publishing enabled with backend: {backend.__class__.__name__}"
            )

        # Metrics
        self._events_published = 0
        self._events_failed = 0
        self._total_latency = 0.0

    def publish(self, event: CloudEvent) -> bool:
        """
        Publish a CloudEvent.

        Args:
            event: CloudEvent to publish

        Returns:
            True if published successfully, False otherwise
        """
        if not self.enabled:
            return True

        try:
            start_time = time.time()
            result = self.backend.publish(event)
            latency = (time.time() - start_time) * 1000  # ms

            self._events_published += 1
            self._total_latency += latency

            logger.info(
                f"Event published",
                extra={
                    "event_id": event.id,
                    "event_type": event.type,
                    "latency_ms": round(latency, 2)
                }
            )

            return result

        except Exception as e:
            self._events_failed += 1
            logger.error(
                f"Failed to publish event: {e}",
                extra={
                    "event_id": event.id,
                    "event_type": event.type
                },
                exc_info=True
            )
            # Don't propagate - event publishing is non-critical
            return False

    def publish_document_processed(
        self,
        document_id: str,
        job_id: str,
        processing_time_seconds: float,
        output_locations: Dict[str, str],
        metrics: Dict[str, int],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Publish document.processed event.

        Args:
            document_id: Document identifier
            job_id: Batch job identifier
            processing_time_seconds: Time taken to process document
            output_locations: Paths to output files/databases
            metrics: Extracted metrics (event_count, entity_count, etc.)
            metadata: Optional metadata (model versions, etc.)

        Returns:
            True if published successfully
        """
        data = DocumentProcessedData(
            document_id=document_id,
            job_id=job_id,
            status="success",
            processing_time_seconds=processing_time_seconds,
            output_location=output_locations,
            metrics=metrics,
            metadata=metadata
        )

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            subject=f"document/{document_id}",
            data=data.model_dump(mode='json')
        )

        return self.publish(event)

    def publish_document_failed(
        self,
        document_id: str,
        job_id: str,
        error_type: str,
        error_message: str,
        retry_count: int = 0
    ) -> bool:
        """
        Publish document.failed event.

        Args:
            document_id: Document identifier
            job_id: Batch job identifier
            error_type: Type of error (e.g., "ValidationError", "TimeoutError")
            error_message: Error message
            retry_count: Number of retries attempted

        Returns:
            True if published successfully
        """
        data = DocumentFailedData(
            document_id=document_id,
            job_id=job_id,
            error_type=error_type,
            error_message=error_message,
            retry_count=retry_count
        )

        event = CloudEvent(
            type=EventType.DOCUMENT_FAILED,
            subject=f"document/{document_id}",
            data=data.model_dump(mode='json')
        )

        return self.publish(event)

    def publish_batch_started(
        self,
        job_id: str,
        total_documents: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Publish batch.started event.

        Args:
            job_id: Batch job identifier
            total_documents: Number of documents in batch
            metadata: Optional metadata

        Returns:
            True if published successfully
        """
        data = BatchStartedData(
            job_id=job_id,
            total_documents=total_documents,
            metadata=metadata
        )

        event = CloudEvent(
            type=EventType.BATCH_STARTED,
            subject=f"batch/{job_id}",
            data=data.model_dump(mode='json')
        )

        return self.publish(event)

    def publish_batch_completed(
        self,
        job_id: str,
        total_documents: int,
        successful: int,
        failed: int,
        duration_seconds: float,
        started_at: datetime,
        output_locations: Dict[str, str],
        aggregate_metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Publish batch.completed event.

        Args:
            job_id: Batch job identifier
            total_documents: Total number of documents
            successful: Number of successfully processed documents
            failed: Number of failed documents
            duration_seconds: Total processing duration
            started_at: Batch start timestamp
            output_locations: Paths to output files/databases
            aggregate_metrics: Optional aggregate metrics

        Returns:
            True if published successfully
        """
        data = BatchCompletedData(
            job_id=job_id,
            total_documents=total_documents,
            successful=successful,
            failed=failed,
            duration_seconds=duration_seconds,
            started_at=started_at,
            output_locations=output_locations,
            aggregate_metrics=aggregate_metrics
        )

        event = CloudEvent(
            type=EventType.BATCH_COMPLETED,
            subject=f"batch/{job_id}",
            data=data.model_dump(mode='json')
        )

        return self.publish(event)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get publisher metrics.

        Returns:
            Dictionary with metrics
        """
        avg_latency = (
            self._total_latency / self._events_published
            if self._events_published > 0
            else 0
        )

        return {
            "events_published": self._events_published,
            "events_failed": self._events_failed,
            "avg_latency_ms": round(avg_latency, 2),
            "success_rate": (
                self._events_published / (self._events_published + self._events_failed)
                if (self._events_published + self._events_failed) > 0
                else 1.0
            )
        }

    def close(self) -> None:
        """Close backend connection."""
        try:
            self.backend.close()
            logger.info(
                f"Event publisher closed",
                extra=self.get_metrics()
            )
        except Exception as e:
            logger.warning(f"Error closing event publisher: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_event_publisher(config: Any) -> EventPublisher:
    """
    Factory function to create EventPublisher from configuration.

    Supports both single and multiple backends:
    - Single: config.events.backend = "redis_streams"
    - Multiple: config.events.backends = ["redis_streams", "webhook"]

    Available backends:
    - redis_streams: Redis Streams (default, always available)
    - webhook: HTTP webhook callbacks (always available)
    - kafka: Apache Kafka (requires kafka-python)
    - nats: NATS messaging (requires nats-py)
    - rabbitmq: RabbitMQ (requires pika)

    Args:
        config: Configuration object with events section

    Returns:
        EventPublisher instance
    """
    # Check if events are enabled
    if not hasattr(config, "events") or not config.events.enabled:
        logger.info("Events disabled - using NullBackend")
        return EventPublisher(backend=NullBackend(), enabled=False)

    # Import backends here to avoid circular imports
    from .backends.redis_streams import RedisStreamsBackend
    from .backends.webhook import WebhookBackend
    from .backends.multi import MultiBackend
    from .backends import (
        KafkaBackend, KAFKA_AVAILABLE,
        NATSBackend, NATS_AVAILABLE,
        RabbitMQBackend, RABBITMQ_AVAILABLE
    )

    # Determine if using single or multiple backends
    backend_types = []

    if config.events.backends:
        # Multi-backend mode
        backend_types = config.events.backends
        logger.info(f"Using multi-backend mode: {backend_types}")
    elif config.events.backend:
        # Single backend mode (backward compatible)
        backend_types = [config.events.backend]
        logger.info(f"Using single backend mode: {config.events.backend}")
    else:
        logger.warning("No backend configured, using NullBackend")
        return EventPublisher(backend=NullBackend(), enabled=False)

    # Create backend instances
    backends = []

    for backend_type in backend_types:
        try:
            if backend_type == "redis_streams":
                redis_config = config.events.redis_streams
                backend = RedisStreamsBackend(
                    url=redis_config.url,
                    stream_name=redis_config.stream_name,
                    max_len=redis_config.max_len,
                    ttl_seconds=getattr(redis_config, "ttl_seconds", None),
                    max_connections=redis_config.connection_pool.max_connections,
                    timeout=redis_config.connection_pool.timeout
                )
                backends.append(backend)
                logger.info(f"Initialized RedisStreamsBackend: {redis_config.stream_name}")

            elif backend_type == "webhook":
                webhook_config = config.events.webhook
                headers = getattr(webhook_config, "headers", {})
                verify_ssl = getattr(webhook_config, "verify_ssl", True)

                backend = WebhookBackend(
                    urls=webhook_config.urls,
                    headers=headers,
                    timeout_seconds=webhook_config.timeout_seconds,
                    retry_attempts=webhook_config.retry_attempts,
                    retry_backoff=webhook_config.retry_backoff,
                    retry_delay_seconds=webhook_config.retry_delay_seconds,
                    verify_ssl=verify_ssl
                )
                backends.append(backend)
                logger.info(f"Initialized WebhookBackend: {len(webhook_config.urls)} endpoint(s)")

            elif backend_type == "kafka":
                if not KAFKA_AVAILABLE:
                    logger.warning("Kafka backend requested but kafka-python not installed, skipping")
                    continue
                kafka_config = config.events.kafka
                backend = KafkaBackend(
                    bootstrap_servers=kafka_config.brokers if isinstance(kafka_config.brokers, list) else [kafka_config.brokers],
                    topic=kafka_config.topic,
                    client_id=getattr(kafka_config, "client_id", "stage2-nlp-publisher"),
                    acks=getattr(kafka_config, "acks", "all"),
                    retries=getattr(kafka_config, "retries", 3),
                    compression_type=getattr(kafka_config, "compression", "gzip")
                )
                backends.append(backend)
                logger.info(f"Initialized KafkaBackend: {kafka_config.topic}")

            elif backend_type == "nats":
                if not NATS_AVAILABLE:
                    logger.warning("NATS backend requested but nats-py not installed, skipping")
                    continue
                nats_config = config.events.nats
                backend = NATSBackend(
                    servers=nats_config.servers if isinstance(nats_config.servers, list) else [nats_config.servers],
                    subject=getattr(nats_config, "subject", "nlp.events"),
                    client_name=getattr(nats_config, "client_name", "stage2-nlp-publisher"),
                    use_jetstream=getattr(nats_config, "use_jetstream", False)
                )
                backends.append(backend)
                logger.info(f"Initialized NATSBackend: {nats_config.servers}")

            elif backend_type == "rabbitmq":
                if not RABBITMQ_AVAILABLE:
                    logger.warning("RabbitMQ backend requested but pika not installed, skipping")
                    continue
                rabbitmq_config = config.events.rabbitmq
                backend = RabbitMQBackend(
                    url=rabbitmq_config.url,
                    exchange=rabbitmq_config.exchange,
                    exchange_type=getattr(rabbitmq_config, "exchange_type", "topic"),
                    routing_key=getattr(rabbitmq_config, "routing_key", "document.processed"),
                    durable=getattr(rabbitmq_config, "durable", True)
                )
                backends.append(backend)
                logger.info(f"Initialized RabbitMQBackend: {rabbitmq_config.exchange}")

            else:
                logger.warning(f"Unknown backend type: {backend_type}, skipping")

        except Exception as e:
            logger.error(f"Failed to initialize {backend_type} backend: {e}", exc_info=True)
            # Continue with other backends even if one fails

    # Create final backend
    if not backends:
        logger.warning("No backends successfully initialized, using NullBackend")
        final_backend = NullBackend()
    elif len(backends) == 1:
        # Single backend
        final_backend = backends[0]
        logger.info("Using single backend")
    else:
        # Multiple backends - wrap in MultiBackend
        final_backend = MultiBackend(backends)
        logger.info(f"Using MultiBackend with {len(backends)} backends")

    return EventPublisher(backend=final_backend, enabled=True)
