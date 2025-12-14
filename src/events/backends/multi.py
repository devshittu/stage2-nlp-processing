"""
Multi-backend wrapper for publishing events to multiple backends simultaneously.

Provides resilience - failure in one backend does not affect others.
"""
import logging
from typing import List, Dict, Any

from ..models import CloudEvent
from .base import EventBackend

logger = logging.getLogger(__name__)


class MultiBackend(EventBackend):
    """
    Multi-backend wrapper that publishes to multiple backends simultaneously.

    Features:
    - Publishes to all backends in parallel
    - Resilient: Failure in one backend doesn't affect others
    - Returns True if at least one backend succeeds
    - Logs all failures for observability
    """

    def __init__(self, backends: List[EventBackend]):
        """
        Initialize multi-backend publisher.

        Args:
            backends: List of backend instances to publish to
        """
        if not backends:
            raise ValueError("MultiBackend requires at least one backend")

        self.backends = backends
        self._backend_names = [backend.__class__.__name__ for backend in backends]

        logger.info(
            f"Initialized MultiBackend with {len(backends)} backends: {', '.join(self._backend_names)}"
        )

    def publish(self, event: CloudEvent) -> bool:
        """
        Publish event to all backends.

        Args:
            event: CloudEvent to publish

        Returns:
            True if at least one backend succeeded, False if all failed
        """
        results: Dict[str, Any] = {}
        success_count = 0
        failure_count = 0

        for idx, backend in enumerate(self.backends):
            backend_name = self._backend_names[idx]

            try:
                result = backend.publish(event)

                if result:
                    success_count += 1
                    results[backend_name] = "success"
                    logger.debug(
                        f"Published to {backend_name}",
                        extra={"event_id": event.id, "backend": backend_name}
                    )
                else:
                    failure_count += 1
                    results[backend_name] = "failed"
                    logger.warning(
                        f"Failed to publish to {backend_name} (returned False)",
                        extra={"event_id": event.id, "backend": backend_name}
                    )

            except Exception as e:
                failure_count += 1
                results[backend_name] = f"error: {str(e)}"
                logger.error(
                    f"Exception publishing to {backend_name}: {e}",
                    extra={
                        "event_id": event.id,
                        "backend": backend_name,
                        "error": str(e)
                    },
                    exc_info=True
                )

        # Log summary
        total_backends = len(self.backends)
        logger.info(
            f"Multi-backend publish complete: {success_count}/{total_backends} succeeded",
            extra={
                "event_id": event.id,
                "event_type": event.type,
                "success_count": success_count,
                "failure_count": failure_count,
                "results": results
            }
        )

        # Succeed if at least one backend succeeded
        return success_count > 0

    def close(self) -> None:
        """Close all backends."""
        for idx, backend in enumerate(self.backends):
            backend_name = self._backend_names[idx]
            try:
                backend.close()
                logger.debug(f"Closed {backend_name}")
            except Exception as e:
                logger.warning(
                    f"Error closing {backend_name}: {e}",
                    extra={"backend": backend_name, "error": str(e)}
                )

        logger.info(f"All backends closed ({len(self.backends)} total)")

    def get_backend_status(self) -> Dict[str, str]:
        """
        Get status of all backends.

        Returns:
            Dictionary mapping backend names to status
        """
        status = {}
        for idx, backend in enumerate(self.backends):
            backend_name = self._backend_names[idx]
            # Simple check - backend exists and is not None
            status[backend_name] = "active" if backend else "inactive"

        return status
