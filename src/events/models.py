"""
CloudEvents data models following CloudEvents v1.0 specification.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event types for NLP pipeline."""

    DOCUMENT_PROCESSED = "com.storytelling.nlp.document.processed"
    DOCUMENT_FAILED = "com.storytelling.nlp.document.failed"
    BATCH_STARTED = "com.storytelling.nlp.batch.started"
    BATCH_COMPLETED = "com.storytelling.nlp.batch.completed"


class CloudEvent(BaseModel):
    """
    CloudEvents v1.0 compliant event structure.

    See: https://github.com/cloudevents/spec/blob/v1.0/spec.md
    """

    # Required fields
    specversion: str = Field(
        default="1.0",
        description="CloudEvents specification version"
    )
    type: str = Field(
        ...,
        description="Event type (reverse DNS notation)"
    )
    source: str = Field(
        default="stage2-nlp-processing",
        description="Event source identifier"
    )
    id: str = Field(
        default_factory=lambda: f"evt_{uuid4().hex[:12]}",
        description="Unique event identifier"
    )

    # Optional fields
    time: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    datacontenttype: str = Field(
        default="application/json",
        description="Content type of data field"
    )
    subject: Optional[str] = Field(
        default=None,
        description="Subject of the event in context"
    )

    # Event payload
    data: Dict[str, Any] = Field(
        ...,
        description="Event-specific data payload"
    )

    # Extension attributes
    traceparent: Optional[str] = Field(
        default=None,
        description="W3C Trace Context traceparent header"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(exclude_none=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True, mode='json')


class DocumentProcessedData(BaseModel):
    """Data payload for document.processed event."""

    document_id: str
    job_id: str
    status: str = "success"
    processing_time_seconds: float
    output_location: Dict[str, str]
    metrics: Dict[str, int]
    metadata: Optional[Dict[str, Any]] = None


class DocumentFailedData(BaseModel):
    """Data payload for document.failed event."""

    document_id: str
    job_id: str
    error_type: str
    error_message: str
    retry_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchStartedData(BaseModel):
    """Data payload for batch.started event."""

    job_id: str
    total_documents: int
    started_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class BatchCompletedData(BaseModel):
    """Data payload for batch.completed event."""

    job_id: str
    total_documents: int
    successful: int
    failed: int
    duration_seconds: float
    started_at: datetime
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    output_locations: Dict[str, str]
    aggregate_metrics: Optional[Dict[str, Any]] = None
