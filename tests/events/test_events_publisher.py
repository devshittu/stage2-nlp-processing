"""
Unit tests for EventPublisher.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.events.publisher import EventPublisher, create_event_publisher
from src.events.models import CloudEvent, EventType
from src.events.backends.base import NullBackend


class TestEventPublisher:
    """Test EventPublisher class."""

    def test_publisher_enabled(self):
        """Test publisher when enabled."""
        backend = Mock()
        backend.publish.return_value = True

        publisher = EventPublisher(backend=backend, enabled=True)

        assert publisher.enabled is True
        assert publisher.backend == backend

    def test_publisher_disabled(self):
        """Test publisher when disabled."""
        backend = Mock()
        publisher = EventPublisher(backend=backend, enabled=False)

        assert publisher.enabled is False
        assert isinstance(publisher.backend, NullBackend)

    def test_publish_event(self):
        """Test publishing a CloudEvent."""
        backend = Mock()
        backend.publish.return_value = True

        publisher = EventPublisher(backend=backend, enabled=True)

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"document_id": "doc-123"}
        )

        result = publisher.publish(event)

        assert result is True
        backend.publish.assert_called_once_with(event)

    def test_publish_when_disabled(self):
        """Test publishing when publisher is disabled."""
        backend = Mock()
        publisher = EventPublisher(backend=backend, enabled=False)

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"test": "data"}
        )

        result = publisher.publish(event)

        assert result is True
        backend.publish.assert_not_called()

    def test_publish_document_processed(self):
        """Test publishing document.processed event."""
        backend = Mock()
        backend.publish.return_value = True

        publisher = EventPublisher(backend=backend, enabled=True)

        result = publisher.publish_document_processed(
            document_id="doc-123",
            job_id="job-456",
            processing_time_seconds=12.5,
            output_locations={"jsonl": "file:///data/events.jsonl"},
            metrics={"event_count": 5, "entity_count": 23},
            metadata={"pipeline_version": "1.0.0"}
        )

        assert result is True
        backend.publish.assert_called_once()

        # Verify the event structure
        call_args = backend.publish.call_args[0][0]
        assert call_args.type == EventType.DOCUMENT_PROCESSED
        assert call_args.subject == "document/doc-123"
        assert call_args.data["document_id"] == "doc-123"
        assert call_args.data["job_id"] == "job-456"
        assert call_args.data["metrics"]["event_count"] == 5

    def test_publish_document_failed(self):
        """Test publishing document.failed event."""
        backend = Mock()
        backend.publish.return_value = True

        publisher = EventPublisher(backend=backend, enabled=True)

        result = publisher.publish_document_failed(
            document_id="doc-123",
            job_id="job-456",
            error_type="ValidationError",
            error_message="Invalid document format",
            retry_count=2
        )

        assert result is True

        call_args = backend.publish.call_args[0][0]
        assert call_args.type == EventType.DOCUMENT_FAILED
        assert call_args.subject == "document/doc-123"
        assert call_args.data["error_type"] == "ValidationError"
        assert call_args.data["retry_count"] == 2

    def test_publish_batch_started(self):
        """Test publishing batch.started event."""
        backend = Mock()
        backend.publish.return_value = True

        publisher = EventPublisher(backend=backend, enabled=True)

        result = publisher.publish_batch_started(
            job_id="job-123",
            total_documents=100,
            metadata={"batch_id": "batch-xyz"}
        )

        assert result is True

        call_args = backend.publish.call_args[0][0]
        assert call_args.type == EventType.BATCH_STARTED
        assert call_args.subject == "batch/job-123"
        assert call_args.data["job_id"] == "job-123"
        assert call_args.data["total_documents"] == 100

    def test_publish_batch_completed(self):
        """Test publishing batch.completed event."""
        backend = Mock()
        backend.publish.return_value = True

        publisher = EventPublisher(backend=backend, enabled=True)

        started_at = datetime.utcnow()
        result = publisher.publish_batch_completed(
            job_id="job-123",
            total_documents=100,
            successful=98,
            failed=2,
            duration_seconds=3600.0,
            started_at=started_at,
            output_locations={"jsonl": "file:///data/events.jsonl"},
            aggregate_metrics={"total_events": 490}
        )

        assert result is True

        call_args = backend.publish.call_args[0][0]
        assert call_args.type == EventType.BATCH_COMPLETED
        assert call_args.subject == "batch/job-123"
        assert call_args.data["successful"] == 98
        assert call_args.data["failed"] == 2
        assert call_args.data["aggregate_metrics"]["total_events"] == 490

    def test_publish_handles_exception(self):
        """Test that publish handles exceptions gracefully."""
        backend = Mock()
        backend.publish.side_effect = Exception("Backend error")

        publisher = EventPublisher(backend=backend, enabled=True)

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"test": "data"}
        )

        # Should return False but not raise
        result = publisher.publish(event)
        assert result is False

    def test_get_metrics(self):
        """Test getting publisher metrics."""
        backend = Mock()
        backend.publish.return_value = True

        publisher = EventPublisher(backend=backend, enabled=True)

        # Publish some events
        for i in range(5):
            event = CloudEvent(
                type=EventType.DOCUMENT_PROCESSED,
                data={"document_id": f"doc-{i}"}
            )
            publisher.publish(event)

        metrics = publisher.get_metrics()

        assert metrics["events_published"] == 5
        assert metrics["events_failed"] == 0
        assert metrics["success_rate"] == 1.0
        assert "avg_latency_ms" in metrics

    def test_get_metrics_with_failures(self):
        """Test metrics when some events fail."""
        backend = Mock()
        # First 3 succeed, last 2 fail
        backend.publish.side_effect = [True, True, True, Exception(), Exception()]

        publisher = EventPublisher(backend=backend, enabled=True)

        for i in range(5):
            event = CloudEvent(
                type=EventType.DOCUMENT_PROCESSED,
                data={"document_id": f"doc-{i}"}
            )
            publisher.publish(event)

        metrics = publisher.get_metrics()

        assert metrics["events_published"] == 3
        assert metrics["events_failed"] == 2
        assert metrics["success_rate"] == 0.6

    def test_close_backend(self):
        """Test closing publisher closes backend."""
        backend = Mock()
        backend.close = Mock()

        publisher = EventPublisher(backend=backend, enabled=True)
        publisher.close()

        backend.close.assert_called_once()

    def test_context_manager(self):
        """Test EventPublisher as context manager."""
        backend = Mock()
        backend.publish.return_value = True
        backend.close = Mock()

        with EventPublisher(backend=backend, enabled=True) as publisher:
            event = CloudEvent(
                type=EventType.DOCUMENT_PROCESSED,
                data={"test": "data"}
            )
            publisher.publish(event)

        backend.close.assert_called_once()


class TestCreateEventPublisher:
    """Test create_event_publisher factory function."""

    def test_create_disabled_publisher(self):
        """Test creating publisher when events are disabled."""
        config = Mock()
        config.events.enabled = False

        publisher = create_event_publisher(config)

        assert publisher.enabled is False
        assert isinstance(publisher.backend, NullBackend)

    def test_create_no_events_config(self):
        """Test creating publisher when events config missing."""
        config = Mock()
        # Use hasattr side effect to simulate missing events attribute
        del config.events

        publisher = create_event_publisher(config)

        assert publisher.enabled is False

    @patch('src.events.backends.redis_streams.RedisStreamsBackend')
    def test_create_redis_streams_publisher(self, mock_backend_class):
        """Test creating publisher with Redis Streams backend."""
        config = Mock()
        config.events = Mock()
        config.events.enabled = True
        config.events.backend = "redis_streams"
        config.events.backends = None  # Not using multi-backend mode
        config.events.redis_streams = Mock()
        config.events.redis_streams.url = "redis://localhost:6379/1"
        config.events.redis_streams.stream_name = "test-stream"
        config.events.redis_streams.max_len = 10000
        config.events.redis_streams.connection_pool = Mock()
        config.events.redis_streams.connection_pool.max_connections = 10
        config.events.redis_streams.connection_pool.timeout = 5
        config.events.redis_streams.get.return_value = None

        mock_backend = Mock()
        mock_backend_class.return_value = mock_backend

        publisher = create_event_publisher(config)

        assert publisher.enabled is True
        assert publisher.backend == mock_backend
        mock_backend_class.assert_called_once()

    @patch('src.events.backends.webhook.WebhookBackend')
    def test_create_webhook_publisher(self, mock_backend_class):
        """Test creating publisher with Webhook backend."""
        config = Mock()
        config.events = Mock()
        config.events.enabled = True
        config.events.backend = "webhook"
        config.events.backends = None  # Not using multi-backend mode
        config.events.webhook = Mock()
        config.events.webhook.urls = ["http://example.com/webhook"]
        config.events.webhook.timeout_seconds = 5
        config.events.webhook.retry_attempts = 3
        config.events.webhook.retry_backoff = "exponential"
        config.events.webhook.retry_delay_seconds = 1.0
        config.events.webhook.get.return_value = {}

        mock_backend = Mock()
        mock_backend_class.return_value = mock_backend

        publisher = create_event_publisher(config)

        assert publisher.enabled is True
        assert publisher.backend == mock_backend
        mock_backend_class.assert_called_once()

    def test_create_unknown_backend(self):
        """Test creating publisher with unknown backend type."""
        config = Mock()
        config.events = Mock()
        config.events.enabled = True
        config.events.backend = "unknown_backend"
        config.events.backends = None  # Not using multi-backend mode

        publisher = create_event_publisher(config)

        # Should fallback to NullBackend
        assert isinstance(publisher.backend, NullBackend)

    @patch('src.events.backends.webhook.WebhookBackend')
    @patch('src.events.backends.redis_streams.RedisStreamsBackend')
    def test_create_multi_backend_publisher(self, mock_redis_class, mock_webhook_class):
        """Test creating publisher with multiple backends."""
        from src.events.backends.multi import MultiBackend

        config = Mock()
        config.events = Mock()
        config.events.enabled = True
        config.events.backends = ["redis_streams", "webhook"]  # Multi-backend mode
        config.events.backend = None  # Should be ignored when backends is set

        # Redis config
        config.events.redis_streams = Mock()
        config.events.redis_streams.url = "redis://localhost:6379/1"
        config.events.redis_streams.stream_name = "test-stream"
        config.events.redis_streams.max_len = 10000
        config.events.redis_streams.connection_pool = Mock()
        config.events.redis_streams.connection_pool.max_connections = 10
        config.events.redis_streams.connection_pool.timeout = 5
        config.events.redis_streams.get.return_value = None

        # Webhook config
        config.events.webhook = Mock()
        config.events.webhook.urls = ["http://example.com/webhook"]
        config.events.webhook.timeout_seconds = 5
        config.events.webhook.retry_attempts = 3
        config.events.webhook.retry_backoff = "exponential"
        config.events.webhook.retry_delay_seconds = 1.0
        config.events.webhook.get.return_value = {}

        mock_redis_backend = Mock()
        mock_webhook_backend = Mock()
        mock_redis_class.return_value = mock_redis_backend
        mock_webhook_class.return_value = mock_webhook_backend

        publisher = create_event_publisher(config)

        assert publisher.enabled is True
        assert isinstance(publisher.backend, MultiBackend)
        assert len(publisher.backend.backends) == 2
        mock_redis_class.assert_called_once()
        mock_webhook_class.assert_called_once()
