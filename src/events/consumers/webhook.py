"""
Webhook consumer for Stage 2 NLP Processing Service.

Receives events via HTTP POST from Stage 1 (Cleaning Service).
"""
import hashlib
import hmac
import json
import logging
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from .base import ConsumerBackend
from ..models import CloudEvent

logger = logging.getLogger(__name__)


class WebhookConsumer(ConsumerBackend):
    """
    Webhook-based event consumer.

    Receives events via HTTP POST requests. This consumer is passive -
    it provides methods to validate and process incoming webhook payloads.
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        signature_header: str = "X-Signature-SHA256",
        timestamp_header: str = "X-Timestamp",
        max_timestamp_drift_seconds: int = 300,
        queue_max_size: int = 1000,
    ):
        """
        Initialize webhook consumer.

        Args:
            secret_key: Secret key for HMAC signature verification
            signature_header: Header name containing the signature
            timestamp_header: Header name containing the timestamp
            max_timestamp_drift_seconds: Maximum allowed timestamp drift
            queue_max_size: Maximum size of internal event queue
        """
        self.secret_key = secret_key
        self.signature_header = signature_header
        self.timestamp_header = timestamp_header
        self.max_timestamp_drift_seconds = max_timestamp_drift_seconds
        self.queue_max_size = queue_max_size

        self._event_queue: deque = deque(maxlen=queue_max_size)
        self._pending_acks: Dict[str, CloudEvent] = {}
        self._connected = False

    def connect(self) -> None:
        """Initialize the webhook consumer (no actual connection needed)."""
        self._connected = True
        logger.info("Webhook consumer initialized")

    def validate_signature(
        self,
        payload: bytes,
        signature: str,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Validate webhook signature.

        Args:
            payload: Raw request body bytes
            signature: Signature from header
            timestamp: Timestamp from header

        Returns:
            True if signature is valid
        """
        if not self.secret_key:
            return True  # No secret configured, skip validation

        try:
            # Check timestamp drift
            if timestamp:
                ts = int(timestamp)
                drift = abs(time.time() - ts)
                if drift > self.max_timestamp_drift_seconds:
                    logger.warning(f"Timestamp drift too large: {drift}s")
                    return False

            # Compute expected signature
            if timestamp:
                message = f"{timestamp}.{payload.decode('utf-8')}"
            else:
                message = payload.decode('utf-8')

            expected = hmac.new(
                self.secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Compare signatures
            return hmac.compare_digest(expected, signature)

        except Exception as e:
            logger.error(f"Signature validation error: {e}")
            return False

    def receive_event(
        self,
        event_data: Dict[str, Any],
        signature: Optional[str] = None,
        timestamp: Optional[str] = None,
        raw_payload: Optional[bytes] = None
    ) -> Optional[CloudEvent]:
        """
        Receive and validate an event from a webhook request.

        Args:
            event_data: Parsed event data (JSON body)
            signature: Optional signature for verification
            timestamp: Optional timestamp for verification
            raw_payload: Raw request body for signature verification

        Returns:
            CloudEvent if valid, None otherwise
        """
        # Validate signature if provided
        if signature and raw_payload:
            if not self.validate_signature(raw_payload, signature, timestamp):
                logger.warning("Webhook signature validation failed")
                return None

        try:
            event = CloudEvent.from_dict(event_data)

            # Add to queue for batch processing
            self._event_queue.append(event)
            self._pending_acks[event.id] = event

            logger.info(f"Webhook event received: {event.type} ({event.id})")
            return event

        except Exception as e:
            logger.error(f"Failed to parse webhook event: {e}")
            return None

    def validate_and_process(
        self,
        event_data: Dict[str, Any],
        handler: Callable[[CloudEvent], bool],
        signature: Optional[str] = None,
        timestamp: Optional[str] = None,
        raw_payload: Optional[bytes] = None
    ) -> tuple:
        """
        Validate and immediately process a webhook event.

        Args:
            event_data: Parsed event data
            handler: Handler function for the event
            signature: Optional signature
            timestamp: Optional timestamp
            raw_payload: Raw payload for signature verification

        Returns:
            Tuple of (success: bool, event_id: str or None, error: str or None)
        """
        event = self.receive_event(event_data, signature, timestamp, raw_payload)

        if not event:
            return (False, None, "Event validation failed")

        try:
            if handler(event):
                self.acknowledge(event.id)
                return (True, event.id, None)
            else:
                return (False, event.id, "Handler returned False")
        except Exception as e:
            logger.error(f"Handler error: {e}")
            return (False, event.id, str(e))

    def consume(
        self,
        handler: Callable[[CloudEvent], bool],
        max_events: int = 10,
        timeout_seconds: float = 5.0
    ) -> int:
        """
        Process queued events (for batch processing mode).

        Args:
            handler: Handler function for events
            max_events: Maximum events to process
            timeout_seconds: Not used for webhook consumer

        Returns:
            Number of events processed
        """
        processed = 0

        while self._event_queue and processed < max_events:
            event = self._event_queue.popleft()
            try:
                if handler(event):
                    self.acknowledge(event.id)
                    processed += 1
            except Exception as e:
                logger.error(f"Handler error for event {event.id}: {e}")
                # Re-queue for retry
                self._event_queue.append(event)

        return processed

    def acknowledge(self, event_id: str) -> bool:
        """
        Acknowledge successful processing of an event.

        Args:
            event_id: Event ID to acknowledge

        Returns:
            True if event was pending
        """
        if event_id in self._pending_acks:
            del self._pending_acks[event_id]
            return True
        return False

    def close(self) -> None:
        """Close the consumer."""
        self._connected = False
        self._event_queue.clear()
        self._pending_acks.clear()
        logger.info("Webhook consumer closed")

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self._event_queue)

    def get_pending_count(self) -> int:
        """Get count of pending acknowledgments."""
        return len(self._pending_acks)


def create_webhook_handler(
    consumer: WebhookConsumer,
    event_handler: Callable[[CloudEvent], bool]
):
    """
    Create a FastAPI-compatible webhook handler.

    Usage:
        from fastapi import FastAPI, Request
        from src.events.consumers import WebhookConsumer, create_webhook_handler

        app = FastAPI()
        consumer = WebhookConsumer(secret_key="your-secret")
        handler = create_webhook_handler(consumer, process_cleaning_event)

        @app.post("/webhooks/stage1/events")
        async def webhook_endpoint(request: Request):
            return await handler(request)
    """
    async def handler(request):
        """FastAPI webhook handler."""
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse

        try:
            # Get raw body for signature verification
            raw_body = await request.body()
            body = json.loads(raw_body)

            # Get headers
            signature = request.headers.get(consumer.signature_header)
            timestamp = request.headers.get(consumer.timestamp_header)

            # Process event
            success, event_id, error = consumer.validate_and_process(
                event_data=body,
                handler=event_handler,
                signature=signature,
                timestamp=timestamp,
                raw_payload=raw_body
            )

            if success:
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "accepted",
                        "event_id": event_id
                    }
                )
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "rejected",
                        "event_id": event_id,
                        "error": error
                    }
                )

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")
        except Exception as e:
            logger.error(f"Webhook handler error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return handler
