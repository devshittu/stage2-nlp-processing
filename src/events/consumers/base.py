"""
Abstract base class for event consumers.

All consumer backends must implement this interface.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from ..models import CloudEvent

logger = logging.getLogger(__name__)


class ConsumerBackend(ABC):
    """
    Abstract base class for event consumer backends.

    Consumers receive events from upstream stages (Stage 1 → Stage 2).
    """

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the message broker.

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    def consume(
        self,
        handler: Callable[[CloudEvent], bool],
        max_events: int = 10,
        timeout_seconds: float = 5.0
    ) -> int:
        """
        Consume events from the backend.

        Args:
            handler: Callback function to process each event.
                    Returns True if event was processed successfully.
            max_events: Maximum events to consume in one call
            timeout_seconds: Timeout for blocking read

        Returns:
            Number of events successfully processed
        """
        pass

    @abstractmethod
    def acknowledge(self, event_id: str) -> bool:
        """
        Acknowledge successful processing of an event.

        Args:
            event_id: ID of the event to acknowledge

        Returns:
            True if acknowledgment was successful
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the connection gracefully."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class NullConsumer(ConsumerBackend):
    """
    No-op consumer for testing or when event consumption is disabled.
    """

    def connect(self) -> None:
        """No-op connection."""
        logger.debug("NullConsumer: connect() called (no-op)")

    def consume(
        self,
        handler: Callable[[CloudEvent], bool],
        max_events: int = 10,
        timeout_seconds: float = 5.0
    ) -> int:
        """No-op consume - returns 0 events."""
        logger.debug("NullConsumer: consume() called (no-op)")
        return 0

    def acknowledge(self, event_id: str) -> bool:
        """No-op acknowledge."""
        logger.debug(f"NullConsumer: acknowledge({event_id}) called (no-op)")
        return True

    def close(self) -> None:
        """No-op close."""
        logger.debug("NullConsumer: close() called (no-op)")
