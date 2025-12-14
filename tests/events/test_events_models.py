"""
Unit tests for CloudEvents models.
"""
import pytest
from datetime import datetime

from src.events.models import (
    CloudEvent,
    EventType,
    DocumentProcessedData,
    DocumentFailedData,
    BatchStartedData,
    BatchCompletedData
)


class TestCloudEvent:
    """Test CloudEvent model."""

    def test_create_cloud_event(self):
        """Test creating a CloudEvent."""
        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            subject="document/doc-123",
            data={"document_id": "doc-123", "status": "success"}
        )

        assert event.specversion == "1.0"
        assert event.type == EventType.DOCUMENT_PROCESSED
        assert event.source == "stage2-nlp-processing"
        assert event.subject == "document/doc-123"
        assert event.data["document_id"] == "doc-123"
        assert event.id.startswith("evt_")
        assert isinstance(event.time, datetime)

    def test_cloud_event_to_json(self):
        """Test converting CloudEvent to JSON."""
        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            subject="document/doc-123",
            data={"document_id": "doc-123"}
        )

        json_str = event.to_json()
        assert isinstance(json_str, str)
        assert "document/doc-123" in json_str
        assert "stage2-nlp-processing" in json_str

    def test_cloud_event_to_dict(self):
        """Test converting CloudEvent to dictionary."""
        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            subject="document/doc-123",
            data={"document_id": "doc-123"}
        )

        event_dict = event.to_dict()
        assert isinstance(event_dict, dict)
        assert event_dict["type"] == EventType.DOCUMENT_PROCESSED
        assert event_dict["subject"] == "document/doc-123"
        assert event_dict["data"]["document_id"] == "doc-123"

    def test_cloud_event_custom_id(self):
        """Test CloudEvent with custom ID."""
        event = CloudEvent(
            type=EventType.DOCUMENT_PROCESSED,
            id="custom-event-id",
            data={"test": "data"}
        )

        assert event.id == "custom-event-id"

    def test_cloud_event_with_traceparent(self):
        """Test CloudEvent with traceparent for distributed tracing."""
        traceparent = "00-trace-id-span-id-01"
        event = CloudEvent(
            type=EventType.BATCH_STARTED,
            traceparent=traceparent,
            data={"job_id": "job-123"}
        )

        assert event.traceparent == traceparent
        event_dict = event.to_dict()
        assert event_dict["traceparent"] == traceparent


class TestDocumentProcessedData:
    """Test DocumentProcessedData model."""

    def test_create_document_processed_data(self):
        """Test creating DocumentProcessedData."""
        data = DocumentProcessedData(
            document_id="doc-123",
            job_id="job-456",
            status="success",
            processing_time_seconds=12.5,
            output_location={"jsonl": "file:///app/data/events.jsonl"},
            metrics={"event_count": 5, "entity_count": 23}
        )

        assert data.document_id == "doc-123"
        assert data.job_id == "job-456"
        assert data.status == "success"
        assert data.processing_time_seconds == 12.5
        assert data.metrics["event_count"] == 5

    def test_document_processed_data_with_metadata(self):
        """Test DocumentProcessedData with metadata."""
        metadata = {"pipeline_version": "1.0.0"}
        data = DocumentProcessedData(
            document_id="doc-123",
            job_id="job-456",
            processing_time_seconds=12.5,
            output_location={},
            metrics={},
            metadata=metadata
        )

        assert data.metadata == metadata


class TestDocumentFailedData:
    """Test DocumentFailedData model."""

    def test_create_document_failed_data(self):
        """Test creating DocumentFailedData."""
        data = DocumentFailedData(
            document_id="doc-123",
            job_id="job-456",
            error_type="ValidationError",
            error_message="Invalid document format",
            retry_count=2
        )

        assert data.document_id == "doc-123"
        assert data.job_id == "job-456"
        assert data.error_type == "ValidationError"
        assert data.error_message == "Invalid document format"
        assert data.retry_count == 2
        assert isinstance(data.timestamp, datetime)


class TestBatchStartedData:
    """Test BatchStartedData model."""

    def test_create_batch_started_data(self):
        """Test creating BatchStartedData."""
        data = BatchStartedData(
            job_id="job-123",
            total_documents=100
        )

        assert data.job_id == "job-123"
        assert data.total_documents == 100
        assert isinstance(data.started_at, datetime)

    def test_batch_started_data_with_metadata(self):
        """Test BatchStartedData with metadata."""
        metadata = {"batch_id": "batch-xyz"}
        data = BatchStartedData(
            job_id="job-123",
            total_documents=100,
            metadata=metadata
        )

        assert data.metadata == metadata


class TestBatchCompletedData:
    """Test BatchCompletedData model."""

    def test_create_batch_completed_data(self):
        """Test creating BatchCompletedData."""
        started_at = datetime.utcnow()
        data = BatchCompletedData(
            job_id="job-123",
            total_documents=100,
            successful=98,
            failed=2,
            duration_seconds=3600.0,
            started_at=started_at,
            output_locations={"jsonl": "file:///app/data/events.jsonl"}
        )

        assert data.job_id == "job-123"
        assert data.total_documents == 100
        assert data.successful == 98
        assert data.failed == 2
        assert data.duration_seconds == 3600.0
        assert data.started_at == started_at
        assert isinstance(data.completed_at, datetime)

    def test_batch_completed_data_with_metrics(self):
        """Test BatchCompletedData with aggregate metrics."""
        metrics = {"total_events": 490, "total_entities": 2300}
        data = BatchCompletedData(
            job_id="job-123",
            total_documents=100,
            successful=100,
            failed=0,
            duration_seconds=3600.0,
            started_at=datetime.utcnow(),
            output_locations={},
            aggregate_metrics=metrics
        )

        assert data.aggregate_metrics == metrics


class TestEventType:
    """Test EventType enum."""

    def test_event_types(self):
        """Test all event types are defined."""
        assert EventType.DOCUMENT_PROCESSED == "com.storytelling.nlp.document.processed"
        assert EventType.DOCUMENT_FAILED == "com.storytelling.nlp.document.failed"
        assert EventType.BATCH_STARTED == "com.storytelling.nlp.batch.started"
        assert EventType.BATCH_COMPLETED == "com.storytelling.nlp.batch.completed"

    def test_event_type_in_cloud_event(self):
        """Test using EventType in CloudEvent."""
        event = CloudEvent(
            type=EventType.BATCH_COMPLETED,
            data={"job_id": "job-123"}
        )

        assert event.type == "com.storytelling.nlp.batch.completed"
