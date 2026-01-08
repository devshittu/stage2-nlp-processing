"""
Kafka consumer for Stage 2 NLP Processing Service.

Consumes events from Stage 1 (Cleaning Service) via Apache Kafka.
Optional backend - requires kafka-python package.
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .base import ConsumerBackend
from ..models import CloudEvent

logger = logging.getLogger(__name__)

try:
    from kafka import KafkaConsumer as KafkaClient
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaClient = None
    KafkaError = Exception


class KafkaConsumer(ConsumerBackend):
    """
    Apache Kafka consumer with consumer group support.

    Requires kafka-python package: pip install kafka-python
    """

    def __init__(
        self,
        brokers: List[str] = None,
        topic: str = "cleaning-events",
        group_id: str = "stage2-nlp-processor",
        client_id: str = "stage2-nlp-consumer",
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = False,
        max_poll_records: int = 10,
    ):
        """
        Initialize Kafka consumer.

        Args:
            brokers: List of Kafka broker addresses
            topic: Topic to consume from
            group_id: Consumer group ID
            client_id: Client identifier
            auto_offset_reset: Where to start reading (earliest/latest)
            enable_auto_commit: Whether to auto-commit offsets
            max_poll_records: Maximum records per poll
        """
        if not KAFKA_AVAILABLE:
            raise ImportError(
                "kafka-python package not installed. "
                "Install with: pip install kafka-python"
            )

        self.brokers = brokers or ["kafka:9092"]
        self.topic = topic
        self.group_id = group_id
        self.client_id = client_id
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.max_poll_records = max_poll_records

        self._consumer: Optional[KafkaClient] = None
        self._connected = False
        self._pending_offsets: Dict[str, tuple] = {}  # event_id -> (topic, partition, offset)

    def connect(self) -> None:
        """Establish connection to Kafka cluster."""
        try:
            self._consumer = KafkaClient(
                self.topic,
                bootstrap_servers=self.brokers,
                group_id=self.group_id,
                client_id=self.client_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=self.enable_auto_commit,
                max_poll_records=self.max_poll_records,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
            )

            self._connected = True
            logger.info(
                f"Kafka consumer connected: {self.topic} "
                f"(group: {self.group_id}, brokers: {self.brokers})"
            )

        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise ConnectionError(f"Kafka connection failed: {e}")

    def consume(
        self,
        handler: Callable[[CloudEvent], bool],
        max_events: int = 10,
        timeout_seconds: float = 5.0
    ) -> int:
        """
        Consume events from Kafka topic.

        Args:
            handler: Handler function for events
            max_events: Maximum events to consume
            timeout_seconds: Poll timeout in seconds

        Returns:
            Number of events processed
        """
        if not self._connected or not self._consumer:
            raise ConnectionError("Consumer not connected. Call connect() first.")

        processed = 0
        timeout_ms = int(timeout_seconds * 1000)

        try:
            # Poll for messages
            records = self._consumer.poll(timeout_ms=timeout_ms, max_records=max_events)

            for topic_partition, messages in records.items():
                for message in messages:
                    if processed >= max_events:
                        break

                    event = self._parse_event(message.value)
                    if event:
                        # Track offset for manual commit
                        self._pending_offsets[event.id] = (
                            topic_partition.topic,
                            topic_partition.partition,
                            message.offset
                        )

                        try:
                            if handler(event):
                                if not self.enable_auto_commit:
                                    self.acknowledge(event.id)
                                processed += 1
                        except Exception as e:
                            logger.error(f"Handler error for event {event.id}: {e}")

        except Exception as e:
            logger.error(f"Error consuming from Kafka: {e}")

        return processed

    def _parse_event(self, data: Dict[str, Any]) -> Optional[CloudEvent]:
        """Parse Kafka message data into CloudEvent."""
        try:
            return CloudEvent.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to parse Kafka message: {e}")
            return None

    def acknowledge(self, event_id: str) -> bool:
        """
        Acknowledge event by committing its offset.

        Args:
            event_id: Event ID to acknowledge

        Returns:
            True if commit was successful
        """
        if not self._consumer or event_id not in self._pending_offsets:
            return False

        try:
            topic, partition, offset = self._pending_offsets[event_id]
            from kafka import TopicPartition, OffsetAndMetadata

            tp = TopicPartition(topic, partition)
            offsets = {tp: OffsetAndMetadata(offset + 1, None)}
            self._consumer.commit(offsets)

            del self._pending_offsets[event_id]
            return True

        except Exception as e:
            logger.error(f"Failed to commit offset for event {event_id}: {e}")
            return False

    def close(self) -> None:
        """Close the Kafka consumer."""
        if self._consumer:
            try:
                self._consumer.close()
                logger.info("Kafka consumer closed")
            except Exception as e:
                logger.warning(f"Error closing Kafka consumer: {e}")
            finally:
                self._consumer = None
                self._connected = False
                self._pending_offsets.clear()

    def get_pending_count(self) -> int:
        """Get count of pending acknowledgments."""
        return len(self._pending_offsets)
