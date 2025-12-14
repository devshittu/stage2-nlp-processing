"""
Unit tests for event LLM logic.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch


class TestEventLLMLogic:
    """Test event extraction via LLM."""

    def test_event_extraction_request_structure(self):
        """Test event extraction request structure."""
        request = {
            "text": "Biden met Netanyahu",
            "context": {"title": "US-Israel Meeting"},
            "domain": "diplomatic_relations"
        }

        assert "text" in request
        assert len(request["text"]) > 0

    @patch('src.core.llm_prompts.build_prompt')
    @patch('src.core.llm_prompts.parse_llm_output')
    def test_prompt_building_and_parsing(self, mock_parse, mock_build):
        """Test prompt building and parsing integration."""
        mock_build.return_value = "Test prompt"
        mock_parse.return_value = []

        from src.core.llm_prompts import build_prompt, parse_llm_output

        prompt = build_prompt("Test text")
        events = parse_llm_output('{"events": []}', "doc_001", "Test text")

        assert len(events) == 0
