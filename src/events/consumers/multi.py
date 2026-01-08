"""
Multi-backend consumer for Stage 2 NLP Processing Service.

Aggregates events from multiple consumer backends with deduplication.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from .base import ConsumerBackend, NullConsumer
from ..models import CloudEvent

logger = logging.getLogger(__name__)


class MultiConsumer(ConsumerBackend):
    """
    Multi-backend consumer that aggregates events from multiple sources.

    Features:
    - Parallel consumption from multiple backends
    - Event deduplication by job_id
    - Graceful degradation when backends fail
    """

    def __init__(
        self,
        consumers: List[ConsumerBackend],
        max_workers: int = 4,
        dedup_enabled: bool = True,
        dedup_ttl_seconds: int = 3600,
    ):
        """
        Initialize multi-backend consumer.

        Args:
            consumers: List of consumer backend instances
            max_workers: Maximum parallel consumer threads
            dedup_enabled: Whether to deduplicate events by job_id
            dedup_ttl_seconds: Time-to-live for deduplication cache
        """
        self.consumers = consumers
        self.max_workers = max_workers
        self.dedup_enabled = dedup_enabled
        self.dedup_ttl_seconds = dedup_ttl_seconds

        self._seen_events: Dict[str, float] = {}
        self._connected = False

    def connect(self) -> None:
        """Connect all consumer backends."""
        connected_count = 0

        for consumer in self.consumers:
            try:
                consumer.connect()
                connected_count += 1
                logger.info(f"Connected: {consumer.__class__.__name__}")
            except Exception as e:
                logger.warning(
                    f"Failed to connect {consumer.__class__.__name__}: {e}"
                )

        if connected_count == 0:
            raise ConnectionError("Failed to connect any consumer backend")

        self._connected = True
        logger.info(
            f"MultiConsumer connected: {connected_count}/{len(self.consumers)} backends"
        )

    def consume(
        self,
        handler: Callable[[CloudEvent], bool],
        max_events: int = 10,
        timeout_seconds: float = 5.0
    ) -> int:
        """Consume events from all backends in parallel."""
        if not self._connected:
            raise ConnectionError("Consumer not connected. Call connect() first.")

        self._cleanup_dedup_cache()

        total_processed = 0
        events_per_backend = max(1, max_events // len(self.consumers))

        def dedup_handler(event: CloudEvent) -> bool:
            job_id = self._get_job_id(event)

            if self.dedup_enabled and job_id:
                if job_id in self._seen_events:
                    logger.debug(f"Duplicate event skipped: {job_id}")
                    return True

                self._seen_events[job_id] = time.time()

            return handler(event)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            for consumer in self.consumers:
                future = executor.submit(
                    self._safe_consume,
                    consumer,
                    dedup_handler,
                    events_per_backend,
                    timeout_seconds
                )
                futures[future] = consumer.__class__.__name__

            for future in as_completed(futures, timeout=timeout_seconds * 2):
                backend_name = futures[future]
                try:
                    count = future.result()
                    total_processed += count
                    if count > 0:
                        logger.debug(f"{backend_name}: processed {count} events")
                except Exception as e:
                    logger.error(f"{backend_name}: consume failed - {e}")

        return total_processed

    def _safe_consume(
        self,
        consumer: ConsumerBackend,
        handler: Callable[[CloudEvent], bool],
        max_events: int,
        timeout_seconds: float
    ) -> int:
        """Safely consume from a single backend with error handling."""
        try:
            return consumer.consume(handler, max_events, timeout_seconds)
        except Exception as e:
            logger.error(f"Error in {consumer.__class__.__name__}: {e}")
            return 0

    def _get_job_id(self, event: CloudEvent) -> Optional[str]:
        """Extract job_id from event for deduplication."""
        if event.data:
            if "job_id" in event.data:
                return event.data["job_id"]
            if "source_job_id" in event.data:
                return event.data["source_job_id"]

        return event.id

    def _cleanup_dedup_cache(self) -> None:
        """Remove expired entries from deduplication cache."""
        if not self.dedup_enabled:
            return

        now = time.time()
        expired = [
            job_id for job_id, ts in self._seen_events.items()
            if now - ts > self.dedup_ttl_seconds
        ]

        for job_id in expired:
            del self._seen_events[job_id]

        if expired:
            logger.debug(f"Cleaned {len(expired)} expired dedup entries")

    def acknowledge(self, event_id: str) -> bool:
        """Acknowledge event on all backends."""
        success = False

        for consumer in self.consumers:
            try:
                if consumer.acknowledge(event_id):
                    success = True
            except Exception as e:
                logger.debug(f"Acknowledge failed on {consumer.__class__.__name__}: {e}")

        return success

    def close(self) -> None:
        """Close all consumer backends."""
        for consumer in self.consumers:
            try:
                consumer.close()
            except Exception as e:
                logger.warning(f"Error closing {consumer.__class__.__name__}: {e}")

        self._connected = False
        self._seen_events.clear()
        logger.info("MultiConsumer closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the multi-consumer."""
        return {
            "num_backends": len(self.consumers),
            "connected": self._connected,
            "dedup_enabled": self.dedup_enabled,
            "dedup_cache_size": len(self._seen_events),
            "backends": [c.__class__.__name__ for c in self.consumers]
        }


def create_multi_consumer(config: Any) -> MultiConsumer:
    """
    Factory function to create MultiConsumer from configuration.

    Args:
        config: Configuration object with upstream_integration section

    Returns:
        MultiConsumer instance
    """
    from .redis_streams import RedisStreamsConsumer
    from .webhook import WebhookConsumer
    from . import KAFKA_AVAILABLE, NATS_AVAILABLE, RABBITMQ_AVAILABLE

    consumers = []

    # Get config section
    upstream_config = getattr(config, 'upstream_integration', None)
    if not upstream_config:
        upstream_config = getattr(config, 'stage1_integration', None)

    # Redis Streams consumer (default)
    if upstream_config and hasattr(upstream_config, 'redis_stream'):
        redis_config = upstream_config.redis_stream
        if getattr(redis_config, 'enabled', True):
            try:
                consumer = RedisStreamsConsumer(
                    url=getattr(redis_config, 'url', 'redis://redis-cache:6379/1'),
                    stream_name=getattr(redis_config, 'stream_name', 'stage1:cleaning:events'),
                    consumer_group=getattr(redis_config, 'consumer_group', 'stage2-nlp-processor'),
                    consumer_name=getattr(redis_config, 'consumer_name', 'consumer-1'),
                )
                consumers.append(consumer)
                logger.info("Redis Streams consumer configured")
            except Exception as e:
                logger.error(f"Failed to create Redis Streams consumer: {e}")
    else:
        # Default Redis Streams consumer
        try:
            consumers.append(RedisStreamsConsumer())
            logger.info("Default Redis Streams consumer configured")
        except Exception as e:
            logger.error(f"Failed to create default Redis consumer: {e}")

    # Webhook consumer
    if upstream_config and hasattr(upstream_config, 'webhook'):
        webhook_config = upstream_config.webhook
        if getattr(webhook_config, 'enabled', False):
            try:
                consumer = WebhookConsumer(
                    secret_key=getattr(webhook_config, 'auth_token', None),
                )
                consumers.append(consumer)
                logger.info("Webhook consumer configured")
            except Exception as e:
                logger.error(f"Failed to create Webhook consumer: {e}")

    # Kafka consumer (optional)
    if KAFKA_AVAILABLE and upstream_config:
        kafka_config = getattr(upstream_config, 'kafka', None)
        if kafka_config and getattr(kafka_config, 'enabled', False):
            try:
                from .kafka import KafkaConsumer
                consumer = KafkaConsumer(
                    brokers=getattr(kafka_config, 'brokers', ['kafka:9092']),
                    topic=getattr(kafka_config, 'topic', 'cleaning-events'),
                    group_id=getattr(kafka_config, 'group_id', 'stage2-nlp-processor'),
                )
                consumers.append(consumer)
                logger.info("Kafka consumer configured")
            except Exception as e:
                logger.error(f"Failed to create Kafka consumer: {e}")

    # NATS consumer (optional)
    if NATS_AVAILABLE and upstream_config:
        nats_config = getattr(upstream_config, 'nats', None)
        if nats_config and getattr(nats_config, 'enabled', False):
            try:
                from .nats import NATSConsumer
                consumer = NATSConsumer(
                    servers=getattr(nats_config, 'servers', ['nats://nats:4222']),
                    subject=getattr(nats_config, 'subject', 'cleaning.events'),
                )
                consumers.append(consumer)
                logger.info("NATS consumer configured")
            except Exception as e:
                logger.error(f"Failed to create NATS consumer: {e}")

    # RabbitMQ consumer (optional)
    if RABBITMQ_AVAILABLE and upstream_config:
        rabbitmq_config = getattr(upstream_config, 'rabbitmq', None)
        if rabbitmq_config and getattr(rabbitmq_config, 'enabled', False):
            try:
                from .rabbitmq import RabbitMQConsumer
                consumer = RabbitMQConsumer(
                    url=getattr(rabbitmq_config, 'url', 'amqp://guest:guest@rabbitmq:5672/'),
                    queue=getattr(rabbitmq_config, 'queue', 'cleaning-events'),
                )
                consumers.append(consumer)
                logger.info("RabbitMQ consumer configured")
            except Exception as e:
                logger.error(f"Failed to create RabbitMQ consumer: {e}")

    if not consumers:
        logger.warning("No consumers configured, using NullConsumer")
        consumers.append(NullConsumer())

    return MultiConsumer(
        consumers=consumers,
        max_workers=len(consumers),
        dedup_enabled=True,
        dedup_ttl_seconds=3600
    )
