"""
RabbitMQ backend for event publishing.

Uses RabbitMQ for reliable message delivery with flexible routing.
"""
import json
import logging
from typing import Optional

from ..models import CloudEvent
from .base import EventBackend

logger = logging.getLogger(__name__)

# Optional pika import - gracefully handle if not installed
try:
    import pika
    from pika.exceptions import AMQPError
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    pika = None
    AMQPError = Exception


class RabbitMQBackend(EventBackend):
    """
    RabbitMQ backend for event publishing.

    Uses RabbitMQ for reliable messaging with:
    - Exchange-based routing (topic, direct, fanout)
    - Message persistence
    - Delivery acknowledgments
    - Dead letter queues
    """

    def __init__(
        self,
        url: str = "amqp://guest:guest@rabbitmq:5672/",
        exchange: str = "nlp-events",
        exchange_type: str = "topic",
        routing_key: str = "document.processed",
        durable: bool = True,
        delivery_mode: int = 2,  # Persistent
        heartbeat: int = 600,
        blocked_connection_timeout: int = 300
    ):
        """
        Initialize RabbitMQ backend.

        Args:
            url: RabbitMQ connection URL (AMQP format)
            exchange: Exchange name
            exchange_type: Exchange type (topic, direct, fanout, headers)
            routing_key: Default routing key for messages
            durable: Make exchange durable (survive broker restart)
            delivery_mode: 1=non-persistent, 2=persistent
            heartbeat: Heartbeat interval in seconds
            blocked_connection_timeout: Blocked connection timeout
        """
        if not RABBITMQ_AVAILABLE:
            raise ImportError(
                "pika not installed. Install with: pip install pika"
            )

        self.url = url
        self.exchange = exchange
        self.exchange_type = exchange_type
        self.routing_key = routing_key
        self.durable = durable
        self.delivery_mode = delivery_mode

        self._connection: Optional[pika.BlockingConnection] = None
        self._channel = None

        # Connection parameters
        self._params = pika.URLParameters(url)
        self._params.heartbeat = heartbeat
        self._params.blocked_connection_timeout = blocked_connection_timeout

        # Initialize connection
        self._connect()

    def _connect(self) -> None:
        """Create RabbitMQ connection and declare exchange."""
        try:
            self._connection = pika.BlockingConnection(self._params)
            self._channel = self._connection.channel()

            # Declare exchange
            self._channel.exchange_declare(
                exchange=self.exchange,
                exchange_type=self.exchange_type,
                durable=self.durable
            )

            logger.info(
                f"Connected to RabbitMQ: {self.exchange} ({self.exchange_type})"
            )

        except AMQPError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def _ensure_connection(self) -> None:
        """Ensure connection is open, reconnect if needed."""
        if not self._connection or self._connection.is_closed:
            logger.info("RabbitMQ connection closed, reconnecting...")
            self._connect()
        elif not self._channel or self._channel.is_closed:
            logger.info("RabbitMQ channel closed, reopening...")
            self._channel = self._connection.channel()

    def publish(self, event: CloudEvent) -> bool:
        """
        Publish event to RabbitMQ exchange.

        Args:
            event: CloudEvent to publish

        Returns:
            True if published successfully

        Raises:
            AMQPError: If publishing fails
        """
        try:
            self._ensure_connection()

            # Convert CloudEvent to JSON
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
            body = json.dumps(event_dict).encode("utf-8")

            # Determine routing key based on event type
            # e.g., "com.storytelling.nlp.document.processed" -> "document.processed"
            event_routing_key = ".".join(event.type.split(".")[-2:])

            # Message properties
            properties = pika.BasicProperties(
                delivery_mode=self.delivery_mode,
                content_type="application/cloudevents+json",
                content_encoding="utf-8",
                message_id=event.id,
                timestamp=int(event.time.timestamp()) if hasattr(event.time, 'timestamp') else None,
                headers={
                    "ce-specversion": event.specversion,
                    "ce-type": event.type,
                    "ce-source": event.source,
                    "ce-id": event.id
                }
            )

            # Publish message
            self._channel.basic_publish(
                exchange=self.exchange,
                routing_key=event_routing_key,
                body=body,
                properties=properties
            )

            logger.debug(
                f"Published event to RabbitMQ",
                extra={
                    "exchange": self.exchange,
                    "routing_key": event_routing_key,
                    "event_id": event.id,
                    "event_type": event.type
                }
            )

            return True

        except AMQPError as e:
            logger.error(
                f"Failed to publish event to RabbitMQ: {e}",
                extra={
                    "event_id": event.id,
                    "event_type": event.type,
                    "exchange": self.exchange
                }
            )
            raise

    def close(self) -> None:
        """Close RabbitMQ connection."""
        if self._connection:
            try:
                if not self._connection.is_closed:
                    self._connection.close()
                logger.info("RabbitMQ backend closed")
            except Exception as e:
                logger.warning(f"Error closing RabbitMQ connection: {e}")
            finally:
                self._connection = None
                self._channel = None

    def declare_queue(
        self,
        queue_name: str,
        routing_key: str = "#",
        durable: bool = True,
        auto_delete: bool = False
    ) -> str:
        """
        Declare a queue and bind it to the exchange.

        This is typically called by downstream stages (Stage 3+).

        Args:
            queue_name: Name of the queue
            routing_key: Routing pattern for binding (# = all)
            durable: Make queue durable
            auto_delete: Auto-delete queue when unused

        Returns:
            Queue name
        """
        self._ensure_connection()

        try:
            # Declare queue
            result = self._channel.queue_declare(
                queue=queue_name,
                durable=durable,
                auto_delete=auto_delete
            )

            # Bind to exchange
            self._channel.queue_bind(
                exchange=self.exchange,
                queue=queue_name,
                routing_key=routing_key
            )

            logger.info(
                f"Declared queue '{queue_name}' bound to '{self.exchange}' "
                f"with routing key '{routing_key}'"
            )

            return result.method.queue

        except AMQPError as e:
            logger.error(f"Failed to declare queue: {e}")
            raise

    def get_exchange_info(self) -> dict:
        """
        Get exchange information.

        Returns:
            Dictionary with exchange info
        """
        return {
            "exchange": self.exchange,
            "exchange_type": self.exchange_type,
            "durable": self.durable,
            "url": self.url.split("@")[-1] if "@" in self.url else self.url,
            "connected": self._connection.is_open if self._connection else False
        }
