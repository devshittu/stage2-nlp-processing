"""
Kafka backend for event publishing.

Uses Apache Kafka for high-throughput, distributed event publishing
suitable for large-scale deployments.
"""
import json
import logging
from typing import Optional, List

from ..models import CloudEvent
from .base import EventBackend

logger = logging.getLogger(__name__)

# Optional Kafka import - gracefully handle if not installed
try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaProducer = None
    KafkaError = Exception


class KafkaBackend(EventBackend):
    """
    Kafka backend for event publishing.

    Uses Apache Kafka for high-throughput event publishing with:
    - Partitioning for parallel processing
    - Replication for fault tolerance
    - Consumer groups for downstream stages
    """

    def __init__(
        self,
        bootstrap_servers: List[str] = None,
        topic: str = "nlp-events",
        client_id: str = "stage2-nlp-publisher",
        acks: str = "all",
        retries: int = 3,
        batch_size: int = 16384,
        linger_ms: int = 10,
        compression_type: str = "gzip",
        max_request_size: int = 1048576,
        timeout_ms: int = 30000
    ):
        """
        Initialize Kafka backend.

        Args:
            bootstrap_servers: List of Kafka broker addresses
            topic: Kafka topic name for events
            client_id: Client identifier
            acks: Acknowledgment level ('0', '1', 'all')
            retries: Number of retries for failed sends
            batch_size: Batch size in bytes
            linger_ms: Time to wait for batch to fill
            compression_type: Compression type (gzip, snappy, lz4, zstd)
            max_request_size: Maximum request size in bytes
            timeout_ms: Request timeout in milliseconds
        """
        if not KAFKA_AVAILABLE:
            raise ImportError(
                "kafka-python not installed. Install with: pip install kafka-python"
            )

        self.bootstrap_servers = bootstrap_servers or ["kafka:9092"]
        self.topic = topic
        self.client_id = client_id
        self._producer: Optional[KafkaProducer] = None

        # Producer configuration
        self._config = {
            "bootstrap_servers": self.bootstrap_servers,
            "client_id": client_id,
            "acks": acks,
            "retries": retries,
            "batch_size": batch_size,
            "linger_ms": linger_ms,
            "compression_type": compression_type,
            "max_request_size": max_request_size,
            "request_timeout_ms": timeout_ms,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
            "key_serializer": lambda k: k.encode("utf-8") if k else None
        }

        # Initialize producer
        self._connect()

    def _connect(self) -> None:
        """Create Kafka producer connection."""
        try:
            self._producer = KafkaProducer(**self._config)
            logger.info(
                f"Connected to Kafka: {self.bootstrap_servers}, topic={self.topic}"
            )
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    def publish(self, event: CloudEvent) -> bool:
        """
        Publish event to Kafka topic.

        Args:
            event: CloudEvent to publish

        Returns:
            True if published successfully

        Raises:
            KafkaError: If publishing fails
        """
        if not self._producer:
            logger.error("Kafka producer not initialized")
            return False

        try:
            # Use event subject or document_id as partition key
            key = event.subject or event.data.get("document_id", event.id)

            # Convert CloudEvent to dict
            event_dict = {
                "specversion": event.specversion,
                "type": event.type,
                "source": event.source,
                "subject": event.subject,
                "id": event.id,
                "time": event.time,
                "datacontenttype": event.datacontenttype,
                "data": event.data
            }

            # Send to Kafka
            future = self._producer.send(
                self.topic,
                key=key,
                value=event_dict
            )

            # Wait for acknowledgment
            record_metadata = future.get(timeout=10)

            logger.debug(
                f"Published event to Kafka",
                extra={
                    "topic": record_metadata.topic,
                    "partition": record_metadata.partition,
                    "offset": record_metadata.offset,
                    "event_id": event.id,
                    "event_type": event.type
                }
            )

            return True

        except KafkaError as e:
            logger.error(
                f"Failed to publish event to Kafka: {e}",
                extra={
                    "event_id": event.id,
                    "event_type": event.type,
                    "topic": self.topic
                }
            )
            raise

    def flush(self, timeout: float = 10.0) -> None:
        """
        Flush all pending messages.

        Args:
            timeout: Maximum time to wait in seconds
        """
        if self._producer:
            self._producer.flush(timeout=timeout)

    def close(self) -> None:
        """Close Kafka producer."""
        if self._producer:
            try:
                self._producer.flush(timeout=5)
                self._producer.close(timeout=5)
                logger.info("Kafka backend closed")
            except Exception as e:
                logger.warning(f"Error closing Kafka producer: {e}")
            finally:
                self._producer = None

    def get_topic_metadata(self) -> dict:
        """
        Get topic metadata.

        Returns:
            Dictionary with topic info
        """
        if not self._producer:
            return {}

        try:
            partitions = self._producer.partitions_for(self.topic)
            return {
                "topic": self.topic,
                "partitions": list(partitions) if partitions else [],
                "bootstrap_servers": self.bootstrap_servers
            }
        except Exception as e:
            logger.error(f"Failed to get topic metadata: {e}")
            return {}
