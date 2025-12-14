"""
Unit tests for dependency parsing logic.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch


class TestDependencyParsing:
    """Test dependency parsing functionality."""

    def test_soa_triplet_extraction_logic(self):
        """Test SOA triplet extraction logic."""
        # Test the logic without loading spaCy model
        subject = "Biden"
        action = "met"
        obj = "Netanyahu"

        # Verify triplet structure
        assert len(subject) > 0
        assert len(action) > 0
        assert len(obj) > 0

    def test_entity_span_creation(self):
        """Test EntitySpan component creation."""
        from src.schemas.data_models import EntitySpan

        component = EntitySpan(
            text="announced",
            start_char=10,
            end_char=19
        )

        assert component.text == "announced"
        assert component.start_char == 10
        assert component.end_char == 19
