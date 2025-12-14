"""
Unit tests for event backends.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.events.models import CloudEvent, EventType
from src.events.backends.base import EventBackend, NullBackend
from src.events.backends.redis_streams import RedisStreamsBackend
from src.events.backends.webhook import WebhookBackend
from src.events.backends.multi import MultiBackend


class TestNullBackend:
    """Test NullBackend."""

    def test_publish_always_succeeds(self):
        """Test that NullBackend always returns True."""
        backend = NullBackend()
        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"document_id": "doc-123"}
        )

        result = backend.publish(event)
        assert result is True

    def test_close_does_nothing(self):
        """Test that close() doesn't raise errors."""
        backend = NullBackend()
        backend.close()  # Should not raise

    def test_context_manager(self):
        """Test NullBackend as context manager."""
        with NullBackend() as backend:
            event = CloudEvent(
                type=EventType.DOCUMENT_PROCESSED,
                data={"test": "data"}
            )
            result = backend.publish(event)
            assert result is True


class TestRedisStreamsBackend:
    """Test RedisStreamsBackend."""

    @patch('src.events.backends.redis_streams.redis.Redis')
    @patch('src.events.backends.redis_streams.ConnectionPool')
    def test_initialize_backend(self, mock_pool_class, mock_redis_class):
        """Test initializing Redis Streams backend."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        backend = RedisStreamsBackend(
            url="redis://localhost:6379/1",
            stream_name="test-stream",
            max_len=1000
        )

        assert backend.stream_name == "test-stream"
        assert backend.max_len == 1000
        mock_redis.ping.assert_called_once()

    @patch('src.events.backends.redis_streams.redis.Redis')
    @patch('src.events.backends.redis_streams.ConnectionPool')
    def test_publish_event(self, mock_pool_class, mock_redis_class):
        """Test publishing event to Redis Stream."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.xadd.return_value = b"1234567890-0"
        mock_redis_class.return_value = mock_redis

        backend = RedisStreamsBackend()

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"document_id": "doc-123"}
        )

        result = backend.publish(event)

        assert result is True
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "nlp-events"  # stream name
        assert "event" in call_args[0][1]  # message dict

    @patch('src.events.backends.redis_streams.redis.Redis')
    @patch('src.events.backends.redis_streams.ConnectionPool')
    def test_publish_respects_max_len(self, mock_pool_class, mock_redis_class):
        """Test that publish respects max_len for stream trimming."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.xadd.return_value = b"1234567890-0"
        mock_redis_class.return_value = mock_redis

        backend = RedisStreamsBackend(max_len=5000)

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"test": "data"}
        )

        backend.publish(event)

        call_args = mock_redis.xadd.call_args
        assert call_args[1]["maxlen"] == 5000
        assert call_args[1]["approximate"] is True

    @patch('src.events.backends.redis_streams.redis.Redis')
    @patch('src.events.backends.redis_streams.ConnectionPool')
    def test_close_backend(self, mock_pool_class, mock_redis_class):
        """Test closing backend disconnects pool."""
        mock_pool = MagicMock()
        mock_pool_class.from_url.return_value = mock_pool

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        backend = RedisStreamsBackend()
        backend.close()

        mock_pool.disconnect.assert_called_once()

    @patch('src.events.backends.redis_streams.redis.Redis')
    @patch('src.events.backends.redis_streams.ConnectionPool')
    def test_create_consumer_group(self, mock_pool_class, mock_redis_class):
        """Test creating consumer group."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.xgroup_create.return_value = True
        mock_redis_class.return_value = mock_redis

        backend = RedisStreamsBackend()
        result = backend.create_consumer_group("test-group")

        assert result is True
        mock_redis.xgroup_create.assert_called_once_with(
            "nlp-events",
            "test-group",
            id="0",
            mkstream=True
        )


class TestWebhookBackend:
    """Test WebhookBackend."""

    @patch('src.events.backends.webhook.requests.Session')
    @patch('src.events.backends.webhook.HTTPAdapter')
    @patch('src.events.backends.webhook.Retry')
    def test_initialize_webhook_backend(self, mock_retry, mock_adapter, mock_session_class):
        """Test initializing webhook backend."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        urls = ["http://example.com/webhook1", "http://example.com/webhook2"]
        backend = WebhookBackend(
            urls=urls,
            timeout_seconds=10,
            retry_attempts=5
        )

        assert backend.urls == urls
        assert backend.timeout == 10
        assert backend.retry_attempts == 5

    @patch('src.events.backends.webhook.requests.Session')
    @patch('src.events.backends.webhook.HTTPAdapter')
    @patch('src.events.backends.webhook.Retry')
    def test_publish_to_webhook(self, mock_retry, mock_adapter, mock_session_class):
        """Test publishing event to webhook."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        backend = WebhookBackend(urls=["http://example.com/webhook"])

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"document_id": "doc-123"}
        )

        result = backend.publish(event)

        assert result is True
        mock_session.post.assert_called_once()

    @patch('src.events.backends.webhook.requests.Session')
    @patch('src.events.backends.webhook.HTTPAdapter')
    @patch('src.events.backends.webhook.Retry')
    def test_publish_to_multiple_webhooks(self, mock_retry, mock_adapter, mock_session_class):
        """Test publishing to multiple webhook URLs."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session

        urls = ["http://example.com/webhook1", "http://example.com/webhook2"]
        backend = WebhookBackend(urls=urls)

        event = CloudEvent(
            type=EventType.BATCH_STARTED,
            data={"job_id": "job-123"}
        )

        result = backend.publish(event)

        assert result is True
        assert mock_session.post.call_count == 2

    @patch('src.events.backends.webhook.requests.Session')
    @patch('src.events.backends.webhook.HTTPAdapter')
    @patch('src.events.backends.webhook.Retry')
    def test_publish_partial_success(self, mock_retry, mock_adapter, mock_session_class):
        """Test publishing with some webhooks failing."""
        import requests

        # First call succeeds, second fails with HTTPError
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status = Mock()

        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.text = "Internal Server Error"
        http_error = requests.HTTPError("Server error")
        http_error.response = mock_response_fail
        mock_response_fail.raise_for_status = Mock(side_effect=http_error)

        mock_session = MagicMock()
        mock_session.post.side_effect = [mock_response_success, mock_response_fail]
        mock_session_class.return_value = mock_session

        urls = ["http://example.com/webhook1", "http://example.com/webhook2"]
        backend = WebhookBackend(urls=urls)

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"test": "data"}
        )

        # Should still succeed if at least one webhook succeeds
        result = backend.publish(event)
        assert result is True

    @patch('src.events.backends.webhook.requests.Session')
    @patch('src.events.backends.webhook.HTTPAdapter')
    @patch('src.events.backends.webhook.Retry')
    def test_publish_all_fail(self, mock_retry, mock_adapter, mock_session_class):
        """Test publishing when all webhooks fail."""
        import requests

        mock_session = MagicMock()
        mock_session.post.side_effect = requests.RequestException("Connection error")
        mock_session_class.return_value = mock_session

        backend = WebhookBackend(urls=["http://example.com/webhook"])

        event = CloudEvent(
            type=EventType.DOCUMENT_FAILED,
            data={"error": "test"}
        )

        with pytest.raises(requests.RequestException):
            backend.publish(event)

    @patch('src.events.backends.webhook.requests.Session')
    @patch('src.events.backends.webhook.HTTPAdapter')
    @patch('src.events.backends.webhook.Retry')
    def test_custom_headers(self, mock_retry, mock_adapter, mock_session_class):
        """Test webhook with custom headers."""
        mock_session = MagicMock()
        mock_session.headers = MagicMock()
        mock_session_class.return_value = mock_session

        headers = {"X-API-Key": "secret-key"}
        backend = WebhookBackend(
            urls=["http://example.com/webhook"],
            headers=headers
        )

        # Verify headers update was called with custom headers
        mock_session.headers.update.assert_called()

    @patch('src.events.backends.webhook.requests.Session')
    @patch('src.events.backends.webhook.HTTPAdapter')
    @patch('src.events.backends.webhook.Retry')
    def test_close_session(self, mock_retry, mock_adapter, mock_session_class):
        """Test closing webhook session."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        backend = WebhookBackend(urls=["http://example.com/webhook"])
        backend.close()

        mock_session.close.assert_called_once()


class TestMultiBackend:
    """Test MultiBackend wrapper for multiple backends."""

    def test_initialize_with_multiple_backends(self):
        """Test initializing MultiBackend with multiple backends."""
        # Create mock backends with proper class names using type()
        MockRedisBackend = type('RedisStreamsBackend', (EventBackend,), {
            'publish': Mock(return_value=True),
            'close': Mock()
        })
        MockWebhookBackend = type('WebhookBackend', (EventBackend,), {
            'publish': Mock(return_value=True),
            'close': Mock()
        })

        backend1 = MockRedisBackend()
        backend2 = MockWebhookBackend()

        multi = MultiBackend(backends=[backend1, backend2])

        assert len(multi.backends) == 2
        assert multi._backend_names == ["RedisStreamsBackend", "WebhookBackend"]

    def test_initialize_empty_backends_raises_error(self):
        """Test that initializing with empty backends list raises ValueError."""
        with pytest.raises(ValueError, match="MultiBackend requires at least one backend"):
            MultiBackend(backends=[])

    def test_publish_to_all_backends_success(self):
        """Test publishing successfully to all backends."""
        backend1 = Mock(spec=EventBackend)
        backend2 = Mock(spec=EventBackend)
        backend1.__class__.__name__ = "Backend1"
        backend2.__class__.__name__ = "Backend2"
        backend1.publish.return_value = True
        backend2.publish.return_value = True

        multi = MultiBackend(backends=[backend1, backend2])

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"document_id": "doc-123"}
        )

        result = multi.publish(event)

        assert result is True
        backend1.publish.assert_called_once_with(event)
        backend2.publish.assert_called_once_with(event)

    def test_publish_partial_failure(self):
        """Test that publish succeeds if at least one backend succeeds."""
        backend1 = Mock(spec=EventBackend)
        backend2 = Mock(spec=EventBackend)
        backend1.__class__.__name__ = "SuccessBackend"
        backend2.__class__.__name__ = "FailBackend"
        backend1.publish.return_value = True  # Success
        backend2.publish.return_value = False  # Failure

        multi = MultiBackend(backends=[backend1, backend2])

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"test": "data"}
        )

        result = multi.publish(event)

        assert result is True  # Should succeed because backend1 succeeded
        backend1.publish.assert_called_once_with(event)
        backend2.publish.assert_called_once_with(event)

    def test_publish_all_backends_fail(self):
        """Test that publish fails only if all backends fail."""
        backend1 = Mock(spec=EventBackend)
        backend2 = Mock(spec=EventBackend)
        backend1.__class__.__name__ = "Backend1"
        backend2.__class__.__name__ = "Backend2"
        backend1.publish.return_value = False
        backend2.publish.return_value = False

        multi = MultiBackend(backends=[backend1, backend2])

        event = CloudEvent(
            type=EventType.BATCH_STARTED,
            data={"job_id": "job-123"}
        )

        result = multi.publish(event)

        assert result is False  # Should fail because all backends failed

    def test_publish_exception_isolation(self):
        """Test that exception in one backend doesn't affect others."""
        backend1 = Mock(spec=EventBackend)
        backend2 = Mock(spec=EventBackend)
        backend3 = Mock(spec=EventBackend)
        backend1.__class__.__name__ = "ErrorBackend"
        backend2.__class__.__name__ = "SuccessBackend"
        backend3.__class__.__name__ = "AnotherSuccessBackend"

        # Backend1 raises exception, backend2 and backend3 succeed
        backend1.publish.side_effect = Exception("Connection error")
        backend2.publish.return_value = True
        backend3.publish.return_value = True

        multi = MultiBackend(backends=[backend1, backend2, backend3])

        event = CloudEvent(
            type=EventType.DOCUMENT_FAILED,
            data={"error": "test"}
        )

        # Should not raise exception, should continue with other backends
        result = multi.publish(event)

        assert result is True  # Should succeed because backend2 and backend3 succeeded
        backend1.publish.assert_called_once()
        backend2.publish.assert_called_once()
        backend3.publish.assert_called_once()

    def test_publish_one_succeeds_one_raises(self):
        """Test resilience when one backend raises exception and one succeeds."""
        backend1 = Mock(spec=EventBackend)
        backend2 = Mock(spec=EventBackend)
        backend1.__class__.__name__ = "FailingBackend"
        backend2.__class__.__name__ = "SuccessBackend"

        backend1.publish.side_effect = RuntimeError("Backend failure")
        backend2.publish.return_value = True

        multi = MultiBackend(backends=[backend1, backend2])

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"document_id": "doc-456"}
        )

        result = multi.publish(event)

        # Should succeed because backend2 succeeded, even though backend1 raised
        assert result is True

    def test_close_all_backends(self):
        """Test that close() is called on all backends."""
        backend1 = Mock(spec=EventBackend)
        backend2 = Mock(spec=EventBackend)
        backend1.__class__.__name__ = "Backend1"
        backend2.__class__.__name__ = "Backend2"

        multi = MultiBackend(backends=[backend1, backend2])
        multi.close()

        backend1.close.assert_called_once()
        backend2.close.assert_called_once()

    def test_close_with_exception_in_one_backend(self):
        """Test that close continues even if one backend raises exception."""
        backend1 = Mock(spec=EventBackend)
        backend2 = Mock(spec=EventBackend)
        backend1.__class__.__name__ = "ErrorBackend"
        backend2.__class__.__name__ = "SuccessBackend"

        backend1.close.side_effect = Exception("Close error")
        backend2.close.return_value = None

        multi = MultiBackend(backends=[backend1, backend2])

        # Should not raise exception
        multi.close()

        backend1.close.assert_called_once()
        backend2.close.assert_called_once()

    def test_get_backend_status(self):
        """Test getting status of all backends."""
        # Create mock backends with proper class names
        MockRedisBackend = type('RedisBackend', (EventBackend,), {
            'publish': Mock(return_value=True),
            'close': Mock()
        })
        MockWebhookBackend = type('WebhookBackend', (EventBackend,), {
            'publish': Mock(return_value=True),
            'close': Mock()
        })

        backend1 = MockRedisBackend()
        backend2 = MockWebhookBackend()

        multi = MultiBackend(backends=[backend1, backend2])
        status = multi.get_backend_status()

        assert status == {
            "RedisBackend": "active",
            "WebhookBackend": "active"
        }

    def test_context_manager(self):
        """Test MultiBackend as context manager."""
        backend1 = Mock(spec=EventBackend)
        backend2 = Mock(spec=EventBackend)
        backend1.__class__.__name__ = "Backend1"
        backend2.__class__.__name__ = "Backend2"
        backend1.publish.return_value = True
        backend2.publish.return_value = True

        with MultiBackend(backends=[backend1, backend2]) as multi:
            event = CloudEvent(
                type=EventType.BATCH_COMPLETED,
                data={"job_id": "job-789"}
            )
            result = multi.publish(event)
            assert result is True

        # Should call close() on exit
        backend1.close.assert_called_once()
        backend2.close.assert_called_once()

    def test_single_backend_still_wrapped(self):
        """Test that MultiBackend works correctly with a single backend."""
        backend = Mock(spec=EventBackend)
        backend.__class__.__name__ = "SingleBackend"
        backend.publish.return_value = True

        multi = MultiBackend(backends=[backend])

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"test": "data"}
        )

        result = multi.publish(event)

        assert result is True
        backend.publish.assert_called_once_with(event)

    def test_publish_returns_false_on_all_exceptions(self):
        """Test that publish returns False when all backends raise exceptions."""
        backend1 = Mock(spec=EventBackend)
        backend2 = Mock(spec=EventBackend)
        backend1.__class__.__name__ = "Backend1"
        backend2.__class__.__name__ = "Backend2"

        backend1.publish.side_effect = Exception("Error 1")
        backend2.publish.side_effect = Exception("Error 2")

        multi = MultiBackend(backends=[backend1, backend2])

        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            data={"test": "data"}
        )

        result = multi.publish(event)

        # Should return False when all backends fail with exceptions
        assert result is False
