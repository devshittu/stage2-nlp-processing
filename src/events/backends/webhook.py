"""
Webhook backend for event publishing via HTTP callbacks.
"""
import logging
import time
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..models import CloudEvent
from .base import EventBackend

logger = logging.getLogger(__name__)


class WebhookBackend(EventBackend):
    """
    Webhook backend for event publishing via HTTP POST.

    Publishes events to one or more HTTP endpoints with
    retry logic and timeout handling.
    """

    def __init__(
        self,
        urls: List[str],
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 5,
        retry_attempts: int = 3,
        retry_backoff: str = "exponential",
        retry_delay_seconds: float = 1.0,
        verify_ssl: bool = True
    ):
        """
        Initialize webhook backend.

        Args:
            urls: List of webhook URLs to POST events to
            headers: Optional HTTP headers to include
            timeout_seconds: Request timeout
            retry_attempts: Number of retry attempts
            retry_backoff: Retry strategy ("exponential" or "linear")
            retry_delay_seconds: Initial retry delay
            verify_ssl: Verify SSL certificates
        """
        self.urls = urls
        self.headers = headers or {}
        self.timeout = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self.retry_delay = retry_delay_seconds
        self.verify_ssl = verify_ssl

        # Create session with retry logic
        self.session = requests.Session()

        # Configure retries for transient failures
        retry_strategy = Retry(
            total=retry_attempts,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            backoff_factor=retry_delay_seconds if retry_backoff == "exponential" else 0
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/cloudevents+json",
            "User-Agent": "stage2-nlp-processing/1.0"
        })

        # Add custom headers
        if self.headers:
            self.session.headers.update(self.headers)

        logger.info(f"Initialized webhook backend with {len(urls)} endpoints")

    def publish(self, event: CloudEvent) -> bool:
        """
        Publish event to all webhook URLs.

        Args:
            event: CloudEvent to publish

        Returns:
            True if at least one webhook succeeded

        Raises:
            requests.RequestException: If all webhooks fail
        """
        event_json = event.to_json()
        success_count = 0
        errors = []

        for url in self.urls:
            try:
                start_time = time.time()

                response = self.session.post(
                    url,
                    data=event_json,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )

                latency = (time.time() - start_time) * 1000  # ms

                response.raise_for_status()

                logger.debug(
                    f"Published event to webhook",
                    extra={
                        "url": url,
                        "event_id": event.id,
                        "event_type": event.type,
                        "status_code": response.status_code,
                        "latency_ms": round(latency, 2)
                    }
                )

                success_count += 1

            except requests.Timeout as e:
                error_msg = f"Webhook timeout: {url}"
                logger.warning(error_msg, extra={"event_id": event.id})
                errors.append(error_msg)

            except requests.HTTPError as e:
                error_msg = f"Webhook HTTP error {e.response.status_code}: {url}"
                logger.warning(
                    error_msg,
                    extra={
                        "event_id": event.id,
                        "status_code": e.response.status_code,
                        "response": e.response.text[:200]
                    }
                )
                errors.append(error_msg)

            except requests.RequestException as e:
                error_msg = f"Webhook request failed: {url} - {str(e)}"
                logger.error(error_msg, extra={"event_id": event.id})
                errors.append(error_msg)

        # Consider success if at least one webhook succeeded
        if success_count > 0:
            if success_count < len(self.urls):
                logger.warning(
                    f"Partial webhook success: {success_count}/{len(self.urls)}",
                    extra={"event_id": event.id, "errors": errors}
                )
            return True
        else:
            # All webhooks failed
            error_summary = "; ".join(errors)
            logger.error(
                f"All webhooks failed for event {event.id}: {error_summary}"
            )
            raise requests.RequestException(f"All webhooks failed: {error_summary}")

    def close(self) -> None:
        """Close HTTP session."""
        try:
            self.session.close()
            logger.info("Webhook backend closed")
        except Exception as e:
            logger.warning(f"Error closing webhook session: {e}")


class AsyncWebhookBackend(WebhookBackend):
    """
    Async webhook backend for non-blocking event publishing.

    Uses thread pool for concurrent webhook calls.
    """

    def __init__(self, *args, max_workers: int = 5, **kwargs):
        """
        Initialize async webhook backend.

        Args:
            max_workers: Maximum concurrent webhook requests
            *args, **kwargs: Passed to WebhookBackend
        """
        super().__init__(*args, **kwargs)
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"Initialized async webhook backend with {max_workers} workers")

    def publish_async(self, event: CloudEvent) -> None:
        """
        Publish event asynchronously (fire and forget).

        Args:
            event: CloudEvent to publish
        """
        self.executor.submit(self._publish_with_logging, event)

    def _publish_with_logging(self, event: CloudEvent) -> None:
        """Wrapper to log async publish results."""
        try:
            self.publish(event)
        except Exception as e:
            logger.error(
                f"Async webhook publish failed: {e}",
                extra={"event_id": event.id}
            )

    def close(self) -> None:
        """Close session and executor."""
        self.executor.shutdown(wait=True)
        super().close()
