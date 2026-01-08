"""
NATS consumer for Stage 2 NLP Processing Service.

Consumes events from Stage 1 (Cleaning Service) via NATS messaging.
Optional backend - requires nats-py package.
"""
import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .base import ConsumerBackend
from ..models import CloudEvent

logger = logging.getLogger(__name__)

try:
    import nats
    from nats.js.api import ConsumerConfig, DeliverPolicy
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False
    nats = None


class NATSConsumer(ConsumerBackend):
    """
    NATS consumer with JetStream support.

    Requires nats-py package: pip install nats-py
    """

    def __init__(
        self,
        servers: List[str] = None,
        subject: str = "cleaning.events",
        stream_name: str = "CLEANING_EVENTS",
        durable_name: str = "stage2-nlp-processor",
        deliver_policy: str = "all",
        use_jetstream: bool = True,
        queue_group: Optional[str] = "stage2-nlp-workers",
    ):
        """
        Initialize NATS consumer.

        Args:
            servers: List of NATS server URLs
            subject: Subject to subscribe to
            stream_name: JetStream stream name
            durable_name: Durable consumer name
            deliver_policy: Delivery policy (all, last, new)
            use_jetstream: Whether to use JetStream for persistence
            queue_group: Queue group for load balancing
        """
        if not NATS_AVAILABLE:
            raise ImportError(
                "nats-py package not installed. "
                "Install with: pip install nats-py"
            )

        self.servers = servers or ["nats://nats:4222"]
        self.subject = subject
        self.stream_name = stream_name
        self.durable_name = durable_name
        self.deliver_policy = deliver_policy
        self.use_jetstream = use_jetstream
        self.queue_group = queue_group

        self._nc = None
        self._js = None
        self._subscription = None
        self._connected = False
        self._pending_msgs: Dict[str, Any] = {}
        self._loop = None

    def connect(self) -> None:
        """Establish connection to NATS server."""
        try:
            self._loop = asyncio.new_event_loop()
            self._loop.run_until_complete(self._async_connect())
            self._connected = True

        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            raise ConnectionError(f"NATS connection failed: {e}")

    async def _async_connect(self) -> None:
        """Async connection setup."""
        self._nc = await nats.connect(servers=self.servers)

        if self.use_jetstream:
            self._js = self._nc.jetstream()

            try:
                deliver_policy_map = {
                    "all": DeliverPolicy.ALL,
                    "last": DeliverPolicy.LAST,
                    "new": DeliverPolicy.NEW,
                }

                self._subscription = await self._js.pull_subscribe(
                    self.subject,
                    durable=self.durable_name,
                    stream=self.stream_name,
                )

            except Exception as e:
                logger.warning(f"JetStream setup warning: {e}")
                self._subscription = await self._nc.subscribe(
                    self.subject,
                    queue=self.queue_group
                )
        else:
            self._subscription = await self._nc.subscribe(
                self.subject,
                queue=self.queue_group
            )

        logger.info(
            f"NATS consumer connected: {self.subject} "
            f"(jetstream: {self.use_jetstream}, servers: {self.servers})"
        )

    def consume(
        self,
        handler: Callable[[CloudEvent], bool],
        max_events: int = 10,
        timeout_seconds: float = 5.0
    ) -> int:
        """Consume events from NATS."""
        if not self._connected or not self._loop:
            raise ConnectionError("Consumer not connected. Call connect() first.")

        return self._loop.run_until_complete(
            self._async_consume(handler, max_events, timeout_seconds)
        )

    async def _async_consume(
        self,
        handler: Callable[[CloudEvent], bool],
        max_events: int,
        timeout_seconds: float
    ) -> int:
        """Async consume implementation."""
        processed = 0

        try:
            if self.use_jetstream and hasattr(self._subscription, 'fetch'):
                try:
                    messages = await self._subscription.fetch(
                        batch=max_events,
                        timeout=timeout_seconds
                    )

                    for msg in messages:
                        event = self._parse_event(msg.data)
                        if event:
                            self._pending_msgs[event.id] = msg
                            try:
                                if handler(event):
                                    await msg.ack()
                                    del self._pending_msgs[event.id]
                                    processed += 1
                            except Exception as e:
                                logger.error(f"Handler error: {e}")
                                await msg.nak()

                except asyncio.TimeoutError:
                    pass

            else:
                end_time = asyncio.get_event_loop().time() + timeout_seconds

                while processed < max_events:
                    remaining = end_time - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break

                    try:
                        msg = await asyncio.wait_for(
                            self._subscription.next_msg(),
                            timeout=remaining
                        )

                        event = self._parse_event(msg.data)
                        if event:
                            try:
                                if handler(event):
                                    processed += 1
                            except Exception as e:
                                logger.error(f"Handler error: {e}")

                    except asyncio.TimeoutError:
                        break

        except Exception as e:
            logger.error(f"Error consuming from NATS: {e}")

        return processed

    def _parse_event(self, data: bytes) -> Optional[CloudEvent]:
        """Parse NATS message data into CloudEvent."""
        try:
            payload = json.loads(data.decode('utf-8'))
            return CloudEvent.from_dict(payload)
        except Exception as e:
            logger.warning(f"Failed to parse NATS message: {e}")
            return None

    def acknowledge(self, event_id: str) -> bool:
        """Acknowledge event."""
        if event_id in self._pending_msgs:
            del self._pending_msgs[event_id]
            return True
        return False

    def close(self) -> None:
        """Close the NATS connection."""
        if self._loop:
            try:
                self._loop.run_until_complete(self._async_close())
            except Exception as e:
                logger.warning(f"Error closing NATS connection: {e}")
            finally:
                self._loop.close()
                self._loop = None
                self._nc = None
                self._js = None
                self._subscription = None
                self._connected = False

    async def _async_close(self) -> None:
        """Async close implementation."""
        if self._subscription:
            await self._subscription.unsubscribe()
        if self._nc:
            await self._nc.close()
        logger.info("NATS consumer closed")

    def get_pending_count(self) -> int:
        """Get count of pending acknowledgments."""
        return len(self._pending_msgs)
