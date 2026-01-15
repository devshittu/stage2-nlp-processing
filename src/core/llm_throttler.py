"""
llm_throttler.py

Event LLM Request Throttler with Backpressure Control.

Solves the resource contention problem where multiple workers overwhelm
the single-threaded vLLM inference server.

Features:
- Semaphore-based concurrency limiting (default: 2 concurrent requests)
- Optional batch processing support
- Request queue with backpressure
- Timeout handling with circuit breaker pattern
- Metrics for monitoring queue depth and wait times

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   EventLLMThrottler                          │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  Semaphore(max_concurrent=2)                        │    │
    │  │  ┌─────────┐ ┌─────────┐                            │    │
    │  │  │ Slot 1  │ │ Slot 2  │  ← Only 2 concurrent reqs  │    │
    │  │  └─────────┘ └─────────┘                            │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                         ↓                                    │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │           Event LLM Service (vLLM)                  │    │
    │  │           Single-threaded inference                 │    │
    │  └─────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘
"""

import threading
import time
import queue
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

import httpx

from src.utils.config_manager import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__, service="llm_throttler")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ThrottlerConfig:
    """Configuration for the Event LLM throttler."""
    # Concurrency control
    max_concurrent_requests: int = 2  # Max simultaneous Event LLM requests

    # Timeouts
    http_timeout: float = 300.0  # 5 minutes per request
    acquire_timeout: float = 600.0  # 10 minutes max wait for semaphore

    # Retry settings
    max_retries: int = 3
    retry_base_delay: float = 2.0  # Exponential backoff base

    # Batch settings
    batch_enabled: bool = True
    batch_size: int = 4  # Documents per batch request
    batch_timeout: float = 600.0  # Timeout for batch requests

    # Circuit breaker
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5  # Consecutive failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before attempting recovery

    # Endpoints
    event_llm_url: str = "http://event-llm-service:8003/api/v1/extract"
    event_llm_batch_url: str = "http://event-llm-service:8003/api/v1/extract/batch"


@dataclass
class ThrottlerMetrics:
    """Metrics for monitoring throttler performance."""
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    requests_timeout: int = 0
    total_wait_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    current_queue_depth: int = 0
    peak_queue_depth: int = 0
    circuit_breaker_trips: int = 0

    def avg_wait_time_ms(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.total_wait_time_ms / self.requests_total

    def avg_processing_time_ms(self) -> float:
        if self.requests_success == 0:
            return 0.0
        return self.total_processing_time_ms / self.requests_success

    def success_rate(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.requests_success / self.requests_total


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests fail fast
    - HALF_OPEN: Testing if service recovered
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    def record_success(self):
        """Record a successful request."""
        with self._lock:
            self.failure_count = 0
            self.state = self.CLOSED

    def record_failure(self):
        """Record a failed request."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
                logger.warning(
                    f"Circuit breaker OPEN after {self.failure_count} failures",
                    extra={"failure_count": self.failure_count}
                )

    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        with self._lock:
            if self.state == self.CLOSED:
                return True

            if self.state == self.OPEN:
                # Check if recovery timeout has passed
                if self.last_failure_time and \
                   (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = self.HALF_OPEN
                    logger.info("Circuit breaker HALF_OPEN, attempting recovery")
                    return True
                return False

            # HALF_OPEN: allow one request to test
            return True

    def reset(self):
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self.state = self.CLOSED
            self.failure_count = 0
            self.last_failure_time = None


# =============================================================================
# Event LLM Throttler
# =============================================================================

class EventLLMThrottler:
    """
    Throttler for Event LLM requests with backpressure control.

    Ensures that the single-threaded vLLM server is not overwhelmed
    by limiting concurrent requests and implementing proper queuing.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[ThrottlerConfig] = None):
        """Singleton pattern for global throttler instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[ThrottlerConfig] = None):
        if self._initialized:
            return

        self.config = config or ThrottlerConfig()
        self._semaphore = threading.Semaphore(self.config.max_concurrent_requests)
        self._queue_depth = 0
        self._queue_lock = threading.Lock()
        self._metrics = ThrottlerMetrics()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.failure_threshold,
            recovery_timeout=self.config.recovery_timeout
        ) if self.config.circuit_breaker_enabled else None

        self._initialized = True

        logger.info(
            "EventLLMThrottler initialized",
            extra={
                "max_concurrent": self.config.max_concurrent_requests,
                "batch_enabled": self.config.batch_enabled,
                "batch_size": self.config.batch_size,
                "circuit_breaker": self.config.circuit_breaker_enabled
            }
        )

    @contextmanager
    def _acquire_slot(self, document_id: str):
        """Acquire a slot for making a request with timeout."""
        wait_start = time.time()

        # Track queue depth
        with self._queue_lock:
            self._queue_depth += 1
            self._metrics.current_queue_depth = self._queue_depth
            if self._queue_depth > self._metrics.peak_queue_depth:
                self._metrics.peak_queue_depth = self._queue_depth

        logger.debug(
            f"[{document_id}] Waiting for Event LLM slot",
            extra={
                "queue_depth": self._queue_depth,
                "document_id": document_id
            }
        )

        try:
            # Acquire with timeout
            acquired = self._semaphore.acquire(timeout=self.config.acquire_timeout)

            if not acquired:
                raise TimeoutError(
                    f"Timeout waiting for Event LLM slot after {self.config.acquire_timeout}s"
                )

            wait_time_ms = (time.time() - wait_start) * 1000
            self._metrics.total_wait_time_ms += wait_time_ms

            logger.debug(
                f"[{document_id}] Acquired Event LLM slot after {wait_time_ms:.0f}ms",
                extra={
                    "wait_time_ms": wait_time_ms,
                    "document_id": document_id
                }
            )

            yield

        finally:
            self._semaphore.release()
            with self._queue_lock:
                self._queue_depth -= 1
                self._metrics.current_queue_depth = self._queue_depth

    def extract_events_single(
        self,
        text: str,
        document_id: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract events from a single document with throttling.

        Args:
            text: Document text
            document_id: Unique document identifier
            entities: Pre-extracted entities from NER
            context: Document context (title, date, etc.)

        Returns:
            Dict with events and metadata

        Raises:
            TimeoutError: If waiting for slot times out
            httpx.HTTPError: If HTTP request fails
        """
        self._metrics.requests_total += 1

        # Check circuit breaker
        if self._circuit_breaker and not self._circuit_breaker.can_execute():
            self._metrics.requests_failed += 1
            raise RuntimeError("Circuit breaker is OPEN - Event LLM service unavailable")

        with self._acquire_slot(document_id):
            return self._make_request(text, document_id, entities, context)

    def _make_request(
        self,
        text: str,
        document_id: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to Event LLM service with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()

                with httpx.Client(timeout=self.config.http_timeout) as client:
                    response = client.post(
                        self.config.event_llm_url,
                        json={
                            "text": text,
                            "document_id": document_id,
                            "entities": entities or [],
                            "context": context or {}
                        }
                    )
                    response.raise_for_status()

                    processing_time_ms = (time.time() - start_time) * 1000
                    self._metrics.total_processing_time_ms += processing_time_ms
                    self._metrics.requests_success += 1

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    result = response.json()

                    logger.debug(
                        f"[{document_id}] Event LLM request succeeded in {processing_time_ms:.0f}ms",
                        extra={
                            "document_id": document_id,
                            "processing_time_ms": processing_time_ms,
                            "events_count": len(result.get("events", []))
                        }
                    )

                    return result

            except httpx.TimeoutException as e:
                self._metrics.requests_timeout += 1
                last_error = e
                logger.warning(
                    f"[{document_id}] Event LLM request timeout (attempt {attempt + 1}/{self.config.max_retries})",
                    extra={"document_id": document_id, "attempt": attempt + 1}
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code

                # Handle 429 Too Many Requests (backpressure from LLM service)
                if status_code == 429:
                    self._metrics.requests_throttled = getattr(self._metrics, 'requests_throttled', 0) + 1
                    retry_after = int(e.response.headers.get("Retry-After", "10"))
                    logger.warning(
                        f"[{document_id}] LLM service busy (429), backing off for {retry_after}s",
                        extra={"document_id": document_id, "retry_after": retry_after}
                    )
                    time.sleep(retry_after)
                    continue  # Retry immediately after backoff

                # Handle 503 Service Unavailable
                elif status_code == 503:
                    logger.warning(
                        f"[{document_id}] LLM service unavailable (503), attempt {attempt + 1}",
                        extra={"document_id": document_id, "attempt": attempt + 1}
                    )
                else:
                    logger.warning(
                        f"[{document_id}] Event LLM request failed ({status_code}): {e}",
                        extra={"document_id": document_id, "status_code": status_code}
                    )

            except httpx.HTTPError as e:
                last_error = e
                logger.warning(
                    f"[{document_id}] Event LLM request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}",
                    extra={"document_id": document_id, "attempt": attempt + 1, "error": str(e)}
                )

            # Adaptive retry with jitter (2026 best practice: prevents thundering herd)
            if attempt < self.config.max_retries - 1:
                import random
                base_wait = self.config.retry_base_delay * (2 ** attempt)
                jitter = random.uniform(0, base_wait * 0.5)  # Up to 50% jitter
                wait_time = min(base_wait + jitter, 60.0)  # Cap at 60 seconds
                logger.debug(
                    f"[{document_id}] Retry backoff: {wait_time:.2f}s (base={base_wait:.2f}, jitter={jitter:.2f})",
                    extra={"document_id": document_id, "wait_time": wait_time}
                )
                time.sleep(wait_time)

        # All retries failed
        self._metrics.requests_failed += 1

        if self._circuit_breaker:
            self._circuit_breaker.record_failure()

        raise last_error or RuntimeError(f"Event LLM request failed for {document_id}")

    def extract_events_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract events from multiple documents in sequence with throttling.

        This method processes documents one at a time through the throttler,
        ensuring proper backpressure and no queue saturation.

        Args:
            documents: List of document dicts with text, document_id, entities, context

        Returns:
            List of extraction results
        """
        results = []

        for doc in documents:
            try:
                result = self.extract_events_single(
                    text=doc.get("text", ""),
                    document_id=doc.get("document_id", "unknown"),
                    entities=doc.get("entities"),
                    context=doc.get("context")
                )
                results.append({
                    "success": True,
                    "document_id": doc.get("document_id"),
                    "data": result
                })
            except Exception as e:
                logger.error(
                    f"Batch extraction failed for {doc.get('document_id')}: {e}",
                    extra={"document_id": doc.get("document_id"), "error": str(e)}
                )
                results.append({
                    "success": False,
                    "document_id": doc.get("document_id"),
                    "error": str(e)
                })

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get current throttler metrics."""
        return {
            "requests_total": self._metrics.requests_total,
            "requests_success": self._metrics.requests_success,
            "requests_failed": self._metrics.requests_failed,
            "requests_timeout": self._metrics.requests_timeout,
            "success_rate": self._metrics.success_rate(),
            "avg_wait_time_ms": self._metrics.avg_wait_time_ms(),
            "avg_processing_time_ms": self._metrics.avg_processing_time_ms(),
            "current_queue_depth": self._metrics.current_queue_depth,
            "peak_queue_depth": self._metrics.peak_queue_depth,
            "circuit_breaker_state": self._circuit_breaker.state if self._circuit_breaker else "disabled"
        }

    def reset_metrics(self):
        """Reset all metrics."""
        self._metrics = ThrottlerMetrics()
        if self._circuit_breaker:
            self._circuit_breaker.reset()

    @classmethod
    def get_instance(cls, config: Optional[ThrottlerConfig] = None) -> 'EventLLMThrottler':
        """Get the singleton throttler instance."""
        return cls(config)


# =============================================================================
# Module-level convenience functions
# =============================================================================

_throttler: Optional[EventLLMThrottler] = None


def get_throttler(config: Optional[ThrottlerConfig] = None) -> EventLLMThrottler:
    """Get the global throttler instance."""
    global _throttler
    if _throttler is None:
        _throttler = EventLLMThrottler(config)
    return _throttler


def extract_events_throttled(
    text: str,
    document_id: str,
    entities: Optional[List[Dict[str, Any]]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to extract events with throttling.

    Uses the global throttler instance.
    """
    throttler = get_throttler()
    return throttler.extract_events_single(text, document_id, entities, context)
