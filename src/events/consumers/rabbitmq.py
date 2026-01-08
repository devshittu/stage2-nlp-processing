"""
RabbitMQ consumer for Stage 2 NLP Processing Service.

Consumes events from Stage 1 (Cleaning Service) via RabbitMQ.
Optional backend - requires pika package.
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .base import ConsumerBackend
from ..models import CloudEvent

logger = logging.getLogger(__name__)

try:
    import pika
    from pika.exceptions import AMQPError
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    pika = None
    AMQPError = Exception


class RabbitMQConsumer(ConsumerBackend):
    """
    RabbitMQ consumer with queue support.

    Requires pika package: pip install pika
    """

    def __init__(
        self,
        url: str = "amqp://guest:guest@rabbitmq:5672/",
        queue: str = "cleaning-events",
        exchange: str = "cleaning-events",
        exchange_type: str = "topic",
        routing_key: str = "document.processed",
        durable: bool = True,
        prefetch_count: int = 10,
        auto_ack: bool = False,
    ):
        """
        Initialize RabbitMQ consumer.

        Args:
            url: RabbitMQ connection URL
            queue: Queue name to consume from
            exchange: Exchange name
            exchange_type: Exchange type (topic, direct, fanout)
            routing_key: Routing key for binding
            durable: Whether queue/exchange are durable
            prefetch_count: Number of messages to prefetch
            auto_ack: Whether to auto-acknowledge messages
        """
        if not RABBITMQ_AVAILABLE:
            raise ImportError(
                "pika package not installed. "
                "Install with: pip install pika"
            )

        self.url = url
        self.queue = queue
        self.exchange = exchange
        self.exchange_type = exchange_type
        self.routing_key = routing_key
        self.durable = durable
        self.prefetch_count = prefetch_count
        self.auto_ack = auto_ack

        self._connection = None
        self._channel = None
        self._connected = False
        self._pending_tags: Dict[str, int] = {}

    def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            params = pika.URLParameters(self.url)
            self._connection = pika.BlockingConnection(params)
            self._channel = self._connection.channel()

            self._channel.basic_qos(prefetch_count=self.prefetch_count)

            self._channel.exchange_declare(
                exchange=self.exchange,
                exchange_type=self.exchange_type,
                durable=self.durable
            )

            self._channel.queue_declare(
                queue=self.queue,
                durable=self.durable
            )

            self._channel.queue_bind(
                queue=self.queue,
                exchange=self.exchange,
                routing_key=self.routing_key
            )

            self._connected = True
            logger.info(
                f"RabbitMQ consumer connected: {self.queue} "
                f"(exchange: {self.exchange}, key: {self.routing_key})"
            )

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise ConnectionError(f"RabbitMQ connection failed: {e}")

    def consume(
        self,
        handler: Callable[[CloudEvent], bool],
        max_events: int = 10,
        timeout_seconds: float = 5.0
    ) -> int:
        """Consume events from RabbitMQ queue."""
        if not self._connected or not self._channel:
            raise ConnectionError("Consumer not connected. Call connect() first.")

        processed = 0

        try:
            for method_frame, properties, body in self._channel.consume(
                queue=self.queue,
                auto_ack=self.auto_ack,
                inactivity_timeout=timeout_seconds
            ):
                if method_frame is None:
                    break

                if processed >= max_events:
                    if not self.auto_ack:
                        self._channel.basic_nack(
                            delivery_tag=method_frame.delivery_tag,
                            requeue=True
                        )
                    break

                event = self._parse_event(body)
                if event:
                    self._pending_tags[event.id] = method_frame.delivery_tag

                    try:
                        if handler(event):
                            if not self.auto_ack:
                                self.acknowledge(event.id)
                            processed += 1
                        else:
                            if not self.auto_ack:
                                self._channel.basic_nack(
                                    delivery_tag=method_frame.delivery_tag,
                                    requeue=True
                                )
                    except Exception as e:
                        logger.error(f"Handler error for event {event.id}: {e}")
                        if not self.auto_ack:
                            self._channel.basic_nack(
                                delivery_tag=method_frame.delivery_tag,
                                requeue=True
                            )

            self._channel.cancel()

        except Exception as e:
            logger.error(f"Error consuming from RabbitMQ: {e}")

        return processed

    def _parse_event(self, body: bytes) -> Optional[CloudEvent]:
        """Parse RabbitMQ message body into CloudEvent."""
        try:
            payload = json.loads(body.decode('utf-8'))
            return CloudEvent.from_dict(payload)
        except Exception as e:
            logger.warning(f"Failed to parse RabbitMQ message: {e}")
            return None

    def acknowledge(self, event_id: str) -> bool:
        """Acknowledge event by its ID."""
        if not self._channel or event_id not in self._pending_tags:
            return False

        try:
            delivery_tag = self._pending_tags[event_id]
            self._channel.basic_ack(delivery_tag=delivery_tag)
            del self._pending_tags[event_id]
            return True
        except Exception as e:
            logger.error(f"Failed to acknowledge event {event_id}: {e}")
            return False

    def close(self) -> None:
        """Close the RabbitMQ connection."""
        try:
            if self._channel:
                self._channel.close()
            if self._connection:
                self._connection.close()
            logger.info("RabbitMQ consumer closed")
        except Exception as e:
            logger.warning(f"Error closing RabbitMQ connection: {e}")
        finally:
            self._channel = None
            self._connection = None
            self._connected = False
            self._pending_tags.clear()

    def get_pending_count(self) -> int:
        """Get count of pending acknowledgments."""
        return len(self._pending_tags)

    def get_queue_size(self) -> int:
        """Get approximate message count in queue."""
        if not self._channel:
            return 0

        try:
            result = self._channel.queue_declare(
                queue=self.queue,
                durable=self.durable,
                passive=True
            )
            return result.method.message_count
        except Exception:
            return 0
