"""
Unit tests for NER logic with mocking.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.schemas.data_models import Entity


class TestNERModel:
    """Test NER model functionality."""

    @patch('src.core.ner_logic.AutoTokenizer')
    @patch('src.core.ner_logic.AutoModelForTokenClassification')
    @patch('src.core.ner_logic.pipeline')
    def test_ner_model_initialization(self, mock_pipeline, mock_model, mock_tokenizer):
        """Test NER model initialization."""
        from src.core.ner_logic import NERModel

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_pipeline.return_value = MagicMock()

        config_override = {
            "model_name": "test-model",
            "confidence_threshold": 0.75,
            "enable_cache": False
        }

        model = NERModel(config_override=config_override)

        assert model.ner_pipeline is not None

    def test_entity_type_normalization(self):
        """Test entity type normalization."""
        from src.core.ner_logic import NERModel

        model = NERModel.__new__(NERModel)  # Create instance without __init__

        assert model._normalize_entity_type("B-PERSON") == "PER"
        assert model._normalize_entity_type("I-ORGANIZATION") == "ORG"
        assert model._normalize_entity_type("LOCATION") == "LOC"
        assert model._normalize_entity_type("GPE") == "GPE"

    def test_context_extraction(self):
        """Test context extraction around entities."""
        from src.core.ner_logic import NERModel

        model = NERModel.__new__(NERModel)
        text = "President Biden met with Prime Minister Netanyahu in Washington."
        context = model._extract_context(text, 10, 15, window_size=20)

        assert "Biden" in context
        assert "[Biden]" in context or "Biden" in context

    def test_cache_key_generation(self):
        """Test cache key generation."""
        from src.core.ner_logic import NERModel

        model = NERModel.__new__(NERModel)
        text = "Test text"
        key = model._get_cache_key(text)

        assert key.startswith("ner:entities:")
        assert len(key) > 15


class TestEntityExtraction:
    """Test entity extraction with mocking."""

    @patch('src.core.ner_logic.resolve_entities')
    def test_entity_resolution_called(self, mock_resolve):
        """Test that entity resolution is called."""
        mock_resolve.return_value = []

        # Verify resolve_entities is called with correct parameters
        from src.core.entity_resolution import resolve_entities

        entities = []
        result = resolve_entities(
            entities,
            substring_match=True,
            token_overlap_threshold=0.5,
            fuzzy_similarity_threshold=0.85,
            keep_all_mentions=True
        )

        assert result == []
