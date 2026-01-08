"""
Redis Streams consumer for Stage 2 NLP Processing Service.

Consumes events from Stage 1 (Cleaning Service) via Redis Streams.
"""
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import redis

from .base import ConsumerBackend
from ..models import CloudEvent

logger = logging.getLogger(__name__)


class RedisStreamsConsumer(ConsumerBackend):
    """
    Redis Streams consumer with consumer group support.

    Uses consumer groups for reliable, distributed event consumption
    with automatic acknowledgment and pending message recovery.
    """

    def __init__(
        self,
        url: str = "redis://redis-cache:6379/1",
        stream_name: str = "stage1:cleaning:events",
        consumer_group: str = "stage2-nlp-processor",
        consumer_name: str = "consumer-1",
        start_from_beginning: bool = False,
        auto_acknowledge: bool = True,
        claim_pending_timeout_ms: int = 60000,
    ):
        """
        Initialize Redis Streams consumer.

        Args:
            url: Redis connection URL (Stage 2 uses DB 3 for cache)
            stream_name: Name of the Redis stream to consume from
            consumer_group: Consumer group name for distributed consumption
            consumer_name: Unique name for this consumer instance
            start_from_beginning: If True, consume all events from beginning
            auto_acknowledge: If True, automatically ack processed events
            claim_pending_timeout_ms: Timeout before claiming pending messages
        """
        self.url = url
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.start_from_beginning = start_from_beginning
        self.auto_acknowledge = auto_acknowledge
        self.claim_pending_timeout_ms = claim_pending_timeout_ms

        self._client: Optional[redis.Redis] = None
        self._connected = False

    def connect(self) -> None:
        """Establish connection and create consumer group if needed."""
        try:
            self._client = redis.from_url(
                self.url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )

            # Test connection
            self._client.ping()

            # Create consumer group if it doesn't exist
            try:
                start_id = "0" if self.start_from_beginning else "$"
                self._client.xgroup_create(
                    self.stream_name,
                    self.consumer_group,
                    id=start_id,
                    mkstream=True
                )
                logger.info(
                    f"Created consumer group '{self.consumer_group}' "
                    f"on stream '{self.stream_name}'"
                )
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    logger.debug(f"Consumer group '{self.consumer_group}' already exists")
                else:
                    raise

            self._connected = True
            logger.info(
                f"Redis Streams consumer connected: {self.stream_name} "
                f"(group: {self.consumer_group}, consumer: {self.consumer_name})"
            )

        except Exception as e:
            logger.error(f"Failed to connect to Redis Streams: {e}")
            raise ConnectionError(f"Redis Streams connection failed: {e}")

    def consume(
        self,
        handler: Callable[[CloudEvent], bool],
        max_events: int = 10,
        timeout_seconds: float = 5.0
    ) -> int:
        """
        Consume events from the stream.

        Args:
            handler: Callback to process each CloudEvent
            max_events: Maximum events to consume
            timeout_seconds: Block timeout in seconds

        Returns:
            Number of events successfully processed
        """
        if not self._connected or not self._client:
            raise ConnectionError("Consumer not connected. Call connect() first.")

        processed = 0
        block_ms = int(timeout_seconds * 1000)

        try:
            # First, claim any pending messages that have timed out
            pending = self._claim_pending_messages(max_events)
            for message_id, data in pending:
                event = self._parse_event(data)
                if event:
                    try:
                        if handler(event):
                            if self.auto_acknowledge:
                                self.acknowledge(message_id)
                            processed += 1
                    except Exception as e:
                        logger.error(f"Handler error for event {event.id}: {e}")

            # Then read new messages
            remaining = max_events - processed
            if remaining > 0:
                messages = self._client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_name: ">"},
                    count=remaining,
                    block=block_ms
                )

                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, data in stream_messages:
                            event = self._parse_event(data)
                            if event:
                                try:
                                    if handler(event):
                                        if self.auto_acknowledge:
                                            self.acknowledge(message_id)
                                        processed += 1
                                except Exception as e:
                                    logger.error(f"Handler error for event {event.id}: {e}")

        except Exception as e:
            logger.error(f"Error consuming from Redis Streams: {e}")

        return processed

    def _claim_pending_messages(self, max_count: int) -> List[tuple]:
        """
        Claim pending messages that have exceeded timeout.

        Args:
            max_count: Maximum messages to claim

        Returns:
            List of (message_id, data) tuples
        """
        claimed = []

        try:
            # Get pending messages
            pending_info = self._client.xpending_range(
                self.stream_name,
                self.consumer_group,
                min="-",
                max="+",
                count=max_count
            )

            for info in pending_info:
                message_id = info["message_id"]
                idle_time = info.get("times_delivered", 0)

                # Claim if idle too long
                if idle_time >= self.claim_pending_timeout_ms:
                    try:
                        result = self._client.xclaim(
                            self.stream_name,
                            self.consumer_group,
                            self.consumer_name,
                            min_idle_time=self.claim_pending_timeout_ms,
                            message_ids=[message_id]
                        )
                        if result:
                            for msg_id, data in result:
                                claimed.append((msg_id, data))
                    except Exception as e:
                        logger.warning(f"Failed to claim message {message_id}: {e}")

        except Exception as e:
            logger.debug(f"Error checking pending messages: {e}")

        return claimed

    def _parse_event(self, data: Dict[str, str]) -> Optional[CloudEvent]:
        """
        Parse Redis message data into CloudEvent.

        Args:
            data: Raw message data from Redis

        Returns:
            CloudEvent instance or None if parsing fails
        """
        try:
            # Check for 'event' field (JSON-encoded CloudEvent)
            if "event" in data:
                event_data = json.loads(data["event"])
                return CloudEvent.from_dict(event_data)

            # Check for 'data' field
            if "data" in data:
                event_data = json.loads(data["data"])
                return CloudEvent.from_dict(event_data)

            # Try to construct from flat fields
            return CloudEvent(
                type=data.get("type", "com.storytelling.cleaning.batch.completed"),
                source=data.get("source", "stage1-cleaning"),
                id=data.get("id", ""),
                subject=data.get("subject"),
                data=json.loads(data.get("payload", "{}"))
            )

        except Exception as e:
            logger.warning(f"Failed to parse event data: {e}")
            return None

    def acknowledge(self, event_id: str) -> bool:
        """
        Acknowledge successful processing of an event.

        Args:
            event_id: Message ID to acknowledge

        Returns:
            True if acknowledgment was successful
        """
        if not self._client:
            return False

        try:
            result = self._client.xack(
                self.stream_name,
                self.consumer_group,
                event_id
            )
            return result > 0
        except Exception as e:
            logger.error(f"Failed to acknowledge event {event_id}: {e}")
            return False

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            try:
                self._client.close()
                logger.info("Redis Streams consumer closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self._client = None
                self._connected = False

    def get_pending_count(self) -> int:
        """Get count of pending messages in the consumer group."""
        if not self._client:
            return 0

        try:
            info = self._client.xpending(self.stream_name, self.consumer_group)
            return info.get("pending", 0) if info else 0
        except Exception:
            return 0

    def get_stream_length(self) -> int:
        """Get total length of the stream."""
        if not self._client:
            return 0

        try:
            return self._client.xlen(self.stream_name)
        except Exception:
            return 0
