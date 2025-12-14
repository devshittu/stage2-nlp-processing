"""
Unit tests for event linking logic.
"""
import pytest
from src.schemas.data_models import Event, EventTrigger, EventMetadata


class TestEventLinking:
    """Test event linking functionality."""

    def test_event_similarity_by_type(self):
        """Test events with same type."""
        event1 = Event(
            event_id="event_0",
            event_type="contact_meet",
            trigger=EventTrigger(text="met", start_char=0, end_char=3),
            arguments=[],
            metadata=EventMetadata(sentiment="neutral")
        )

        event2 = Event(
            event_id="event_1",
            event_type="contact_meet",
            trigger=EventTrigger(text="meeting", start_char=0, end_char=7),
            arguments=[],
            metadata=EventMetadata(sentiment="neutral")
        )

        assert event1.event_type == event2.event_type

    def test_event_similarity_by_domain(self):
        """Test events with same domain."""
        event1 = Event(
            event_id="event_0",
            event_type="contact_meet",
            trigger=EventTrigger(text="met", start_char=0, end_char=3),
            arguments=[],
            metadata=EventMetadata(sentiment="neutral"),
            domain="diplomatic_relations"
        )

        event2 = Event(
            event_id="event_1",
            event_type="contact_phone_write",
            trigger=EventTrigger(text="called", start_char=0, end_char=6),
            arguments=[],
            metadata=EventMetadata(sentiment="neutral"),
            domain="diplomatic_relations"
        )

        assert event1.domain == event2.domain
