"""
Shared pytest fixtures for NLP pipeline testing.
"""
import pytest
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_text():
    """Sample text for NER and processing tests."""
    return "Apple CEO Tim Cook met with Microsoft CEO Satya Nadella in Washington on Monday."


@pytest.fixture
def sample_long_text():
    """Longer sample text for comprehensive testing."""
    return """
    The Federal Reserve announced a 0.25% interest rate hike on Wednesday.
    Fed Chair Jerome Powell stated that the move aims to combat inflation.
    President Biden met with Israeli PM Netanyahu in Washington to discuss
    regional security. The meeting lasted two hours at the White House.
    Google announced plans to lay off 12,000 employees as part of a major
    restructuring effort. CEO Sundar Pichai said the cuts are necessary
    to focus on AI development.
    """


@pytest.fixture
def sample_document():
    """Sample Stage 1 document."""
    return {
        "document_id": "test_doc_001",
        "version": "1.0",
        "cleaned_text": "Apple CEO Tim Cook met with Microsoft CEO Satya Nadella.",
        "cleaned_title": "Tech Leaders Meet",
        "cleaned_author": "Test Author",
        "cleaned_publication_date": "2024-12-13T10:00:00Z",
        "cleaned_source_url": "https://example.com/article",
        "cleaned_categories": ["technology"],
        "cleaned_tags": ["Apple", "Microsoft"],
        "cleaned_word_count": 10
    }


@pytest.fixture
def sample_entities():
    """Sample entity list for testing."""
    from src.schemas.data_models import Entity

    return [
        Entity(
            text="Apple",
            type="ORG",
            start_char=0,
            end_char=5,
            confidence=0.95,
            entity_id="test_entity_0"
        ),
        Entity(
            text="Tim Cook",
            type="PER",
            start_char=10,
            end_char=18,
            confidence=0.98,
            entity_id="test_entity_1"
        ),
        Entity(
            text="Cook",
            type="PER",
            start_char=40,
            end_char=44,
            confidence=0.90,
            entity_id="test_entity_2"
        ),
        Entity(
            text="Microsoft",
            type="ORG",
            start_char=50,
            end_char=59,
            confidence=0.96,
            entity_id="test_entity_3"
        )
    ]


@pytest.fixture
def sample_event():
    """Sample event for testing."""
    from src.schemas.data_models import Event, EventTrigger, EventArgument, Entity, EventMetadata

    return Event(
        event_id="test_event_0",
        event_type="contact_meet",
        trigger=EventTrigger(
            text="met",
            start_char=19,
            end_char=22
        ),
        arguments=[
            EventArgument(
                argument_role="agent",
                entity=Entity(
                    text="Tim Cook",
                    type="PER",
                    start_char=10,
                    end_char=18,
                    confidence=0.98
                ),
                confidence=1.0
            )
        ],
        metadata=EventMetadata(
            sentiment="neutral",
            causality="Scheduled business meeting",
            confidence=0.95
        ),
        domain="technology_innovation"
    )


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    from unittest.mock import MagicMock

    config = MagicMock()
    config.general.gpu_enabled = False
    config.general.max_text_length = 10000
    config.ner_service.model_name = "test-model"
    config.ner_service.confidence_threshold = 0.75
    config.ner_service.entity_types = ["PER", "ORG", "LOC", "GPE", "DATE", "TIME", "MONEY", "MISC"]
    config.ner_service.enable_cache = True
    config.caching.enabled = True
    config.caching.redis_url = "redis://localhost:6379/0"

    return config
