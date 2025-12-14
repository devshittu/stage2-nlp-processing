"""
Base interface for event backends.
"""
from abc import ABC, abstractmethod
from typing import Optional

from ..models import CloudEvent


class EventBackend(ABC):
    """
    Abstract base class for event publishing backends.

    All backend implementations must inherit from this class
    and implement the publish method.
    """

    @abstractmethod
    def publish(self, event: CloudEvent) -> bool:
        """
        Publish an event to the backend.

        Args:
            event: CloudEvent to publish

        Returns:
            True if published successfully, False otherwise

        Raises:
            Exception: If publishing fails critically
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close backend connection and cleanup resources.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class NullBackend(EventBackend):
    """
    Null backend for testing or when events are disabled.

    Does nothing - discards all events.
    """

    def publish(self, event: CloudEvent) -> bool:
        """Discard event."""
        return True

    def close(self) -> None:
        """No-op."""
        pass
