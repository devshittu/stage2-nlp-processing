"""
NATS backend for event publishing.

Uses NATS (Neural Autonomic Transport System) for cloud-native,
lightweight, high-performance messaging.
"""
import asyncio
import json
import logging
from typing import Optional, List

from ..models import CloudEvent
from .base import EventBackend

logger = logging.getLogger(__name__)

# Optional NATS import - gracefully handle if not installed
try:
    import nats
    from nats.aio.client import Client as NATSClient
    from nats.errors import Error as NATSError
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False
    nats = None
    NATSClient = None
    NATSError = Exception


class NATSBackend(EventBackend):
    """
    NATS backend for event publishing.

    Uses NATS for cloud-native messaging with:
    - Pub/sub messaging
    - Request/reply patterns
    - Queue groups for load balancing
    - JetStream for persistence (optional)
    """

    def __init__(
        self,
        servers: List[str] = None,
        subject: str = "nlp.events",
        client_name: str = "stage2-nlp-publisher",
        connect_timeout: float = 5.0,
        reconnect_time_wait: float = 2.0,
        max_reconnect_attempts: int = 60,
        use_jetstream: bool = False,
        stream_name: str = "NLP_EVENTS"
    ):
        """
        Initialize NATS backend.

        Args:
            servers: List of NATS server URLs
            subject: NATS subject for events
            client_name: Client name for identification
            connect_timeout: Connection timeout in seconds
            reconnect_time_wait: Time to wait between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts
            use_jetstream: Enable JetStream for persistence
            stream_name: JetStream stream name (if use_jetstream=True)
        """
        if not NATS_AVAILABLE:
            raise ImportError(
                "nats-py not installed. Install with: pip install nats-py"
            )

        self.servers = servers or ["nats://nats:4222"]
        self.subject = subject
        self.client_name = client_name
        self.use_jetstream = use_jetstream
        self.stream_name = stream_name

        self._connect_options = {
            "servers": self.servers,
            "name": client_name,
            "connect_timeout": connect_timeout,
            "reconnect_time_wait": reconnect_time_wait,
            "max_reconnect_attempts": max_reconnect_attempts,
            "allow_reconnect": True,
            "verbose": False
        }

        self._nc: Optional[NATSClient] = None
        self._js = None  # JetStream context
        self._loop = None

        # Initialize connection
        self._connect_sync()

    def _connect_sync(self) -> None:
        """Create NATS connection (sync wrapper for async connect)."""
        try:
            # Create or get event loop
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

            # Run async connect
            self._loop.run_until_complete(self._connect_async())

        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            raise

    async def _connect_async(self) -> None:
        """Async NATS connection."""
        self._nc = await nats.connect(**self._connect_options)

        if self.use_jetstream:
            self._js = self._nc.jetstream()
            # Create stream if it doesn't exist
            try:
                await self._js.add_stream(
                    name=self.stream_name,
                    subjects=[f"{self.subject}.*"]
                )
                logger.info(f"Created JetStream stream: {self.stream_name}")
            except Exception as e:
                if "stream name already in use" in str(e).lower():
                    logger.debug(f"JetStream stream already exists: {self.stream_name}")
                else:
                    logger.warning(f"JetStream stream creation failed: {e}")

        logger.info(
            f"Connected to NATS: {self.servers}, subject={self.subject}"
        )

    def publish(self, event: CloudEvent) -> bool:
        """
        Publish event to NATS subject.

        Args:
            event: CloudEvent to publish

        Returns:
            True if published successfully

        Raises:
            NATSError: If publishing fails
        """
        if not self._nc:
            logger.error("NATS client not initialized")
            return False

        try:
            # Convert CloudEvent to JSON bytes
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
            payload = json.dumps(event_dict).encode("utf-8")

            # Determine full subject (include event type for filtering)
            full_subject = f"{self.subject}.{event.type.split('.')[-1]}"

            # Run async publish
            self._loop.run_until_complete(
                self._publish_async(full_subject, payload)
            )

            logger.debug(
                f"Published event to NATS",
                extra={
                    "subject": full_subject,
                    "event_id": event.id,
                    "event_type": event.type
                }
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to publish event to NATS: {e}",
                extra={
                    "event_id": event.id,
                    "event_type": event.type,
                    "subject": self.subject
                }
            )
            raise

    async def _publish_async(self, subject: str, payload: bytes) -> None:
        """Async publish to NATS."""
        if self.use_jetstream and self._js:
            await self._js.publish(subject, payload)
        else:
            await self._nc.publish(subject, payload)

    def close(self) -> None:
        """Close NATS connection."""
        if self._nc:
            try:
                self._loop.run_until_complete(self._nc.drain())
                self._loop.run_until_complete(self._nc.close())
                logger.info("NATS backend closed")
            except Exception as e:
                logger.warning(f"Error closing NATS connection: {e}")
            finally:
                self._nc = None

    def get_connection_status(self) -> dict:
        """
        Get connection status.

        Returns:
            Dictionary with connection info
        """
        if not self._nc:
            return {"connected": False}

        return {
            "connected": self._nc.is_connected,
            "servers": self.servers,
            "subject": self.subject,
            "jetstream_enabled": self.use_jetstream,
            "client_name": self.client_name
        }
