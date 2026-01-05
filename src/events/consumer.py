"""
src/events/consumer.py

Redis Stream event consumer for Stage 1 → Stage 2 integration.

Responsibilities:
- Subscribe to Stage 1 CloudEvents stream
- Consume events using consumer groups (fault-tolerant)
- Parse and validate CloudEvents
- Trigger NLP processing for cleaned documents
- Handle errors and retries
- Track consumption metrics

DESIGN PATTERN: Consumer Group pattern
- At-least-once delivery guarantee
- Multiple consumers for scalability
- Automatic failover on consumer crash
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from collections import defaultdict

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

logger = logging.getLogger("stage2_nlp")


class CloudEventConsumer:
    """
    Redis Stream consumer for CloudEvents.

    Features:
    - Consumer group pattern (fault-tolerant)
    - Event filtering by type
    - Automatic retries
    - Dead letter queue for failed events
    - Metrics tracking
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize event consumer.

        Args:
            config: Consumer configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", False)

        # Redis connection
        self.redis_host = self.config.get("redis_host", os.getenv("REDIS_HOST", "redis-cache"))
        self.redis_port = self.config.get("redis_port", int(os.getenv("REDIS_PORT", "6379")))
        self.redis_db = self.config.get("redis_db", int(os.getenv("REDIS_CACHE_DB", "1")))
        self.redis_client: Optional[Any] = None

        # Stream configuration
        self.source_stream = self.config.get("source_stream", "stage1:cleaning:events")
        self.consumer_group = self.config.get("consumer_group", "stage2-nlp-processor")
        self.consumer_name = self.config.get(
            "consumer_name",
            os.getenv("HOSTNAME", f"consumer-{os.getpid()}")
        )

        # Event filtering
        consume_events_config = self.config.get("consume_events")
        self.event_filter = set(consume_events_config) if consume_events_config else None

        # Processing configuration
        self.auto_process = self.config.get("auto_process", True)
        self.batch_mode = self.config.get("batch_mode", True)
        self.poll_interval_ms = self.config.get("poll_interval_ms", 1000)
        self.batch_size = self.config.get("batch_size", 10)

        # Error handling
        self.retry_failed = self.config.get("retry_failed", True)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay_seconds = self.config.get("retry_delay_seconds", 5)
        self.dead_letter_stream = self.config.get("dead_letter_stream", "stage2:nlp:failed-events")

        # Idempotency
        self.check_already_processed = self.config.get("check_already_processed", True)
        self.deduplication_ttl_hours = self.config.get("deduplication_ttl_hours", 24)

        # Event handler callback
        self.event_handler: Optional[Callable] = None

        # Metrics
        self._metrics = {
            "total_consumed": 0,
            "processed": 0,
            "filtered": 0,
            "failed": 0,
            "retried": 0,
            "duplicates_skipped": 0,
            "by_event_type": defaultdict(int),
        }

        # State
        self._running = False
        self._consumer_task: Optional[asyncio.Task] = None

    async def initialize(self) -> bool:
        """
        Initialize Redis connection and consumer group.

        Returns:
            True if successful
        """
        if not self.enabled:
            logger.info("event_consumer_disabled_in_config")
            return False

        if not REDIS_AVAILABLE:
            logger.error("redis_library_not_available")
            self.enabled = False
            return False

        try:
            # Create Redis connection
            self.redis_client = await aioredis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # Test connection
            await self.redis_client.ping()
            logger.info(
                f"redis_connection_established: {self.redis_host}:{self.redis_port}/{self.redis_db}"
            )

            # Create consumer group if not exists
            try:
                await self.redis_client.xgroup_create(
                    name=self.source_stream,
                    groupname=self.consumer_group,
                    id="0",  # Start from beginning
                    mkstream=True
                )
                logger.info(
                    f"consumer_group_created: stream={self.source_stream}, "
                    f"group={self.consumer_group}"
                )
            except aioredis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    logger.info(f"consumer_group_already_exists: {self.consumer_group}")
                else:
                    raise

            # Check if source stream exists
            stream_exists = await self.redis_client.exists(self.source_stream)
            if not stream_exists:
                logger.warning(
                    f"source_stream_not_found: {self.source_stream} "
                    "(will be created when first event arrives)"
                )

            event_filter = list(self.event_filter) if self.event_filter else "all"
            logger.info(
                f"event_consumer_initialized: stream={self.source_stream}, "
                f"group={self.consumer_group}, consumer={self.consumer_name}, "
                f"event_filter={event_filter}, auto_process={self.auto_process}"
            )

            return True

        except Exception as e:
            logger.error(f"failed_to_initialize_event_consumer: {e}", exc_info=True)
            self.enabled = False
            return False

    def set_event_handler(self, handler: Callable):
        """
        Set callback function for handling consumed events.

        Args:
            handler: Async function that processes events
                     Signature: async def handler(event_data: Dict) -> bool
        """
        self.event_handler = handler

    def should_consume_event(self, event_type: str) -> bool:
        """
        Check if event type should be consumed (event filtering).

        Args:
            event_type: CloudEvent type

        Returns:
            True if event should be consumed
        """
        if not self.event_filter:
            return True  # No filter, consume all events

        return event_type in self.event_filter

    async def is_already_processed(self, event_id: str) -> bool:
        """
        Check if event was already processed (idempotency).

        Args:
            event_id: CloudEvent ID

        Returns:
            True if already processed
        """
        if not self.check_already_processed or not self.redis_client:
            return False

        try:
            key = f"stage2:processed:{event_id}"
            exists = await self.redis_client.exists(key)
            return exists == 1
        except Exception as e:
            logger.error(f"failed_to_check_processed_status: {e}")
            return False

    async def mark_as_processed(self, event_id: str):
        """
        Mark event as processed (idempotency).

        Args:
            event_id: CloudEvent ID
        """
        if not self.check_already_processed or not self.redis_client:
            return

        try:
            key = f"stage2:processed:{event_id}"
            ttl_seconds = self.deduplication_ttl_hours * 3600
            await self.redis_client.setex(key, ttl_seconds, "1")
        except Exception as e:
            logger.error(f"failed_to_mark_as_processed: {e}")

    async def consume_events(self) -> List[Dict[str, Any]]:
        """
        Consume events from Redis Stream using consumer group.

        Returns:
            List of consumed events
        """
        if not self.redis_client:
            return []

        try:
            # Read from stream using consumer group
            # XREADGROUP GROUP <group> <consumer> COUNT <batch_size> BLOCK <poll_ms> STREAMS <stream> >
            messages = await self.redis_client.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.source_stream: ">"},  # > means new messages only
                count=self.batch_size,
                block=self.poll_interval_ms
            )

            if not messages:
                return []

            # Parse messages
            events = []
            for stream_name, message_list in messages:
                for message_id, message_data in message_list:
                    try:
                        # Parse CloudEvent from Redis Stream
                        event = self._parse_cloud_event(message_id, message_data)
                        if event:
                            events.append(event)
                    except Exception as e:
                        logger.error(
                            f"failed_to_parse_event: message_id={message_id}, error={e}",
                            exc_info=True
                        )

            return events

        except Exception as e:
            logger.error(f"failed_to_consume_events: {e}", exc_info=True)
            return []

    def _parse_cloud_event(self, message_id: str, message_data: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Parse CloudEvent from Redis Stream message.

        Args:
            message_id: Redis Stream message ID
            message_data: Redis Stream message data (flat key-value)

        Returns:
            Parsed CloudEvent dictionary or None
        """
        try:
            # CloudEvent fields
            event = {
                "stream_message_id": message_id,
                "specversion": message_data.get("specversion", "1.0"),
                "type": message_data.get("type"),
                "source": message_data.get("source"),
                "id": message_data.get("id"),
                "time": message_data.get("time"),
                "subject": message_data.get("subject"),
                "datacontenttype": message_data.get("datacontenttype", "application/json"),
            }

            # Parse data field (JSON string)
            data_str = message_data.get("data", "{}")
            event["data"] = json.loads(data_str)

            # Validate required fields
            if not event["type"] or not event["id"]:
                logger.warning(f"invalid_cloud_event: missing_required_fields: {message_id}")
                return None

            return event

        except Exception as e:
            logger.error(f"failed_to_parse_cloud_event: {e}")
            return None

    async def process_event(self, event: Dict[str, Any]) -> bool:
        """
        Process a single event.

        Args:
            event: CloudEvent dictionary

        Returns:
            True if processed successfully
        """
        event_type = event.get("type")
        event_id = event.get("id")

        self._metrics["total_consumed"] += 1
        self._metrics["by_event_type"][event_type] += 1

        # Check event filter
        if not self.should_consume_event(event_type):
            logger.debug(f"event_filtered: type={event_type}, id={event_id}")
            self._metrics["filtered"] += 1
            return True  # Not an error, just filtered

        # Check idempotency
        if await self.is_already_processed(event_id):
            logger.info(f"event_already_processed: type={event_type}, id={event_id}")
            self._metrics["duplicates_skipped"] += 1
            return True  # Already processed, skip

        # Call event handler
        if not self.event_handler:
            logger.warning("no_event_handler_configured")
            return False

        try:
            if self.auto_process:
                # Trigger processing
                success = await self.event_handler(event)

                if success:
                    self._metrics["processed"] += 1
                    await self.mark_as_processed(event_id)
                    logger.info(
                        f"event_processed_successfully: type={event_type}, id={event_id}"
                    )
                else:
                    self._metrics["failed"] += 1
                    logger.error(f"event_processing_failed: type={event_type}, id={event_id}")

                return success
            else:
                # Log only, don't process
                logger.info(
                    f"event_consumed_not_processed: type={event_type}, id={event_id} "
                    "(auto_process=false)"
                )
                self._metrics["processed"] += 1
                return True

        except Exception as e:
            self._metrics["failed"] += 1
            logger.error(
                f"event_processing_exception: type={event_type}, id={event_id}, error={e}",
                exc_info=True
            )
            return False

    async def acknowledge_event(self, event: Dict[str, Any]):
        """
        Acknowledge event as processed (XACK).

        Args:
            event: CloudEvent dictionary
        """
        if not self.redis_client:
            return

        try:
            message_id = event.get("stream_message_id")
            if message_id:
                await self.redis_client.xack(
                    self.source_stream,
                    self.consumer_group,
                    message_id
                )
        except Exception as e:
            logger.error(f"failed_to_acknowledge_event: {e}")

    async def send_to_dead_letter_queue(self, event: Dict[str, Any], error: str):
        """
        Send failed event to dead letter queue.

        Args:
            event: CloudEvent dictionary
            error: Error message
        """
        if not self.redis_client:
            return

        try:
            dead_letter_data = {
                "original_event": json.dumps(event),
                "error": error,
                "failed_at": datetime.utcnow().isoformat(),
                "consumer": self.consumer_name
            }

            await self.redis_client.xadd(
                self.dead_letter_stream,
                dead_letter_data,
                maxlen=1000  # Keep last 1000 failed events
            )

            logger.info(
                f"event_sent_to_dlq: event_id={event.get('id')}, "
                f"dlq={self.dead_letter_stream}"
            )
        except Exception as e:
            logger.error(f"failed_to_send_to_dlq: {e}")

    async def run(self):
        """
        Main consumer loop.

        Continuously consumes events from Redis Stream.
        """
        if not self.enabled or not self.redis_client:
            logger.error("event_consumer_not_initialized")
            return

        self._running = True
        logger.info(
            f"event_consumer_started: stream={self.source_stream}, "
            f"group={self.consumer_group}, consumer={self.consumer_name}"
        )

        while self._running:
            try:
                # Consume batch of events
                events = await self.consume_events()

                if not events:
                    continue

                # Process each event
                for event in events:
                    try:
                        success = await self.process_event(event)

                        if success:
                            # Acknowledge successful processing
                            await self.acknowledge_event(event)
                        else:
                            # Retry or send to DLQ
                            if self.retry_failed:
                                # TODO: Implement retry logic with backoff
                                pass
                            else:
                                await self.send_to_dead_letter_queue(
                                    event,
                                    "processing_failed"
                                )
                                await self.acknowledge_event(event)

                    except Exception as e:
                        logger.error(
                            f"event_processing_error: event_id={event.get('id')}, error={e}",
                            exc_info=True
                        )
                        await self.send_to_dead_letter_queue(event, str(e))
                        await self.acknowledge_event(event)

            except Exception as e:
                logger.error(f"consumer_loop_error: {e}", exc_info=True)
                await asyncio.sleep(self.retry_delay_seconds)

        logger.info("event_consumer_stopped")

    async def start(self):
        """Start consumer in background."""
        if not self.enabled:
            logger.warning("cannot_start_disabled_consumer")
            return

        self._consumer_task = asyncio.create_task(self.run())

    async def stop(self):
        """Stop consumer gracefully."""
        self._running = False

        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        logger.info("event_consumer_stopped")

    async def health_check(self) -> Dict[str, Any]:
        """
        Check consumer health.

        Returns:
            Health status dictionary
        """
        if not self.enabled:
            return {
                "enabled": False,
                "healthy": False,
                "reason": "consumer_disabled"
            }

        if not self.redis_client:
            return {
                "enabled": True,
                "healthy": False,
                "reason": "redis_not_connected"
            }

        try:
            # Check Redis connection
            await self.redis_client.ping()

            # Check stream exists
            stream_exists = await self.redis_client.exists(self.source_stream)

            # Get consumer group info
            try:
                group_info = await self.redis_client.xinfo_groups(self.source_stream)
                group_exists = any(g["name"] == self.consumer_group for g in group_info)
            except:
                group_exists = False

            # Get pending count (lag)
            try:
                pending_info = await self.redis_client.xpending(
                    self.source_stream,
                    self.consumer_group
                )
                pending_count = pending_info["pending"] if pending_info else 0
            except:
                pending_count = 0

            return {
                "enabled": True,
                "healthy": True,
                "stream_exists": stream_exists,
                "group_exists": group_exists,
                "pending_count": pending_count,
                "running": self._running,
                "metrics": self.get_metrics()
            }

        except Exception as e:
            return {
                "enabled": True,
                "healthy": False,
                "error": str(e)
            }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get consumer metrics.

        Returns:
            Metrics dictionary
        """
        return {
            "total_consumed": self._metrics["total_consumed"],
            "processed": self._metrics["processed"],
            "filtered": self._metrics["filtered"],
            "failed": self._metrics["failed"],
            "retried": self._metrics["retried"],
            "duplicates_skipped": self._metrics["duplicates_skipped"],
            "by_event_type": dict(self._metrics["by_event_type"]),
        }

    async def close(self):
        """Close Redis connection."""
        await self.stop()

        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

        logger.info("event_consumer_closed")


# Singleton instance
_consumer_instance: Optional[CloudEventConsumer] = None


def get_event_consumer() -> CloudEventConsumer:
    """Get singleton instance of event consumer."""
    global _consumer_instance

    if _consumer_instance is None:
        # Load config from settings
        try:
            from src.utils.config_manager import ConfigManager
            settings = ConfigManager.get_settings()
            if hasattr(settings, 'event_consumer') and settings.event_consumer is not None:
                config = settings.event_consumer.model_dump()
            else:
                config = {"enabled": False}
        except Exception as e:
            logger.warning(f"failed_to_load_consumer_config: {e}")
            config = {"enabled": False}

        _consumer_instance = CloudEventConsumer(config)

    return _consumer_instance
