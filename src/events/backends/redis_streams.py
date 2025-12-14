"""
Redis Streams backend for event publishing.
"""
import logging
from typing import Optional

import redis
from redis.connection import ConnectionPool

from ..models import CloudEvent
from .base import EventBackend

logger = logging.getLogger(__name__)


class RedisStreamsBackend(EventBackend):
    """
    Redis Streams backend for event publishing.

    Uses Redis Streams for high-throughput, low-latency event publishing
    with consumer group support for downstream stages.
    """

    def __init__(
        self,
        url: str = "redis://redis:6379/1",
        stream_name: str = "nlp-events",
        max_len: int = 10000,
        ttl_seconds: Optional[int] = None,
        max_connections: int = 10,
        timeout: int = 5
    ):
        """
        Initialize Redis Streams backend.

        Args:
            url: Redis connection URL
            stream_name: Name of the Redis stream
            max_len: Maximum stream length (older entries trimmed)
            ttl_seconds: Optional TTL for stream entries (requires Redis 7.0+)
            max_connections: Maximum connections in pool
            timeout: Connection timeout in seconds
        """
        self.url = url
        self.stream_name = stream_name
        self.max_len = max_len
        self.ttl_seconds = ttl_seconds

        # Create connection pool
        self.pool = ConnectionPool.from_url(
            url,
            max_connections=max_connections,
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
            decode_responses=True
        )

        # Create Redis client
        self.redis = redis.Redis(connection_pool=self.pool)

        # Test connection
        try:
            self.redis.ping()
            logger.info(f"Connected to Redis Streams: {stream_name}")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def publish(self, event: CloudEvent) -> bool:
        """
        Publish event to Redis Stream.

        Args:
            event: CloudEvent to publish

        Returns:
            True if published successfully

        Raises:
            redis.RedisError: If publishing fails
        """
        try:
            # Serialize event to JSON
            event_json = event.to_json()

            # Publish to stream with trimming
            message_id = self.redis.xadd(
                self.stream_name,
                {"event": event_json},
                maxlen=self.max_len,
                approximate=True  # More efficient trimming
            )

            logger.debug(
                f"Published event to Redis Stream",
                extra={
                    "stream": self.stream_name,
                    "event_id": event.id,
                    "event_type": event.type,
                    "message_id": message_id
                }
            )

            return True

        except redis.RedisError as e:
            logger.error(
                f"Failed to publish event to Redis Stream: {e}",
                extra={
                    "event_id": event.id,
                    "event_type": event.type,
                    "stream": self.stream_name
                }
            )
            raise

    def close(self) -> None:
        """Close Redis connection."""
        try:
            self.pool.disconnect()
            logger.info("Redis Streams backend closed")
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")

    def create_consumer_group(
        self,
        group_name: str,
        consumer_name: str = "consumer-1",
        start_id: str = "0"
    ) -> bool:
        """
        Create consumer group for reading events.

        This is typically called by downstream stages (Stage 3+).

        Args:
            group_name: Name of consumer group
            consumer_name: Name of consumer in group
            start_id: Starting message ID (0 = from beginning, $ = from end)

        Returns:
            True if created successfully
        """
        try:
            self.redis.xgroup_create(
                self.stream_name,
                group_name,
                id=start_id,
                mkstream=True
            )
            logger.info(
                f"Created consumer group '{group_name}' on stream '{self.stream_name}'"
            )
            return True

        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group '{group_name}' already exists")
                return True
            else:
                logger.error(f"Failed to create consumer group: {e}")
                raise

    def get_stream_info(self) -> dict:
        """
        Get stream metadata and statistics.

        Returns:
            Dictionary with stream info
        """
        try:
            info = self.redis.xinfo_stream(self.stream_name)
            return {
                "length": info["length"],
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": info.get("groups", 0)
            }
        except redis.RedisError as e:
            logger.error(f"Failed to get stream info: {e}")
            return {}
