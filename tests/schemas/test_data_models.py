"""
Unit tests for Pydantic data models.
"""
import pytest
from datetime import datetime
from src.schemas.data_models import (
    Entity, SOATriplet, EntitySpan, Event, EventTrigger,
    EventArgument, EventMetadata, ProcessedDocument,
    create_event_id
)


class TestEntity:
    """Test Entity model."""

    def test_entity_creation(self):
        """Test creating an entity with all fields."""
        entity = Entity(
            text="Apple Inc.",
            type="ORG",
            start_char=0,
            end_char=10,
            confidence=0.95,
            context="[Apple Inc.] announced new products",
            normalized_form="Apple Inc.",
            entity_id="test_entity_0"
        )

        assert entity.text == "Apple Inc."
        assert entity.type == "ORG"
        assert entity.confidence == 0.95
        assert entity.normalized_form == "Apple Inc."

    def test_entity_minimal_fields(self):
        """Test entity with only required fields."""
        entity = Entity(
            text="Biden",
            type="PER",
            start_char=0,
            end_char=5,
            confidence=0.90
        )

        assert entity.text == "Biden"
        assert entity.normalized_form is None
        assert entity.entity_id is None

    def test_entity_position_validation(self):
        """Test entity position validation."""
        # Valid entity
        entity = Entity(
            text="Test",
            type="PER",
            start_char=0,
            end_char=4,
            confidence=0.5
        )
        assert entity.start_char < entity.end_char


class TestSOATriplet:
    """Test SOA Triplet model."""

    def test_soa_triplet_creation(self):
        """Test creating SOA triplet."""
        triplet = SOATriplet(
            subject=EntitySpan(text="Biden", start_char=0, end_char=5),
            action=EntitySpan(text="met", start_char=6, end_char=9),
            object=EntitySpan(text="Netanyahu", start_char=10, end_char=19),
            confidence=0.95,
            sentence="Biden met Netanyahu in Washington."
        )

        assert triplet.subject.text == "Biden"
        assert triplet.action.text == "met"
        assert triplet.object.text == "Netanyahu"
        assert triplet.confidence == 0.95

    def test_entity_span(self):
        """Test EntitySpan component."""
        component = EntitySpan(
            text="announced",
            start_char=10,
            end_char=19
        )

        assert component.text == "announced"
        assert component.start_char == 10


class TestEvent:
    """Test Event model."""

    def test_event_creation(self):
        """Test creating an event."""
        event = Event(
            event_id="event_0",
            event_type="contact_meet",
            trigger=EventTrigger(text="met", start_char=10, end_char=13),
            arguments=[
                EventArgument(
                    argument_role="agent",
                    entity=Entity(
                        text="Biden",
                        type="PER",
                        start_char=0,
                        end_char=5,
                        confidence=0.95
                    ),
                    confidence=0.9
                )
            ],
            metadata=EventMetadata(
                sentiment="neutral",
                confidence=0.85
            ),
            domain="diplomatic_relations"
        )

        assert event.event_type == "contact_meet"
        assert event.trigger.text == "met"
        assert len(event.arguments) == 1
        assert event.domain == "diplomatic_relations"

    def test_event_minimal(self):
        """Test event with minimal required fields."""
        event = Event(
            event_id="event_1",
            event_type="policy_announce",
            trigger=EventTrigger(text="announced", start_char=20, end_char=29),
            arguments=[],
            metadata=EventMetadata(sentiment="neutral")
        )

        assert event.event_id == "event_1"
        assert len(event.arguments) == 0


class TestProcessedDocument:
    """Test ProcessedDocument model."""

    def test_processed_document_creation(self):
        """Test creating a processed document."""
        doc = ProcessedDocument(
            document_id="doc_001",
            job_id="job_123",
            processed_at=datetime.now().isoformat(),
            normalized_date="2024-12-13T00:00:00Z",
            original_text="Test article text.",
            source_document={"document_id": "doc_001"},
            extracted_entities=[],
            extracted_soa_triplets=[],
            events=[],
            event_linkages=[],
            storylines=[],
            processing_metadata={"processing_time_ms": 100}
        )

        assert doc.document_id == "doc_001"
        assert doc.job_id == "job_123"
        assert len(doc.extracted_entities) == 0


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_event_id(self):
        """Test event ID creation."""
        event_id = create_event_id("doc_456", 2)
        assert event_id == "doc_456_event_2"
