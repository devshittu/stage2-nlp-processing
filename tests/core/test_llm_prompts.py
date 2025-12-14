"""
Unit tests for LLM prompts and event extraction.
"""
import pytest
import json
from src.core.llm_prompts import (
    build_prompt,
    parse_llm_output,
    _extract_json_from_output,
    _parse_single_event,
    _validate_span,
    get_domain_prompt,
    list_supported_domains,
    list_supported_event_types,
    get_event_type_description
)


class TestPromptBuilding:
    """Test prompt building functionality."""

    def test_build_prompt_basic(self):
        """Test basic prompt building."""
        text = "President Biden met with Prime Minister Netanyahu."
        prompt = build_prompt(text, include_examples=False)

        assert "NEWSWORTHY" in prompt
        assert text in prompt
        assert "Extract ONLY" in prompt or "EXTRACT ONLY" in prompt

    def test_build_prompt_with_context(self):
        """Test prompt with context information."""
        text = "Biden met Netanyahu"
        context = {
            "cleaned_title": "US-Israel Meeting",
            "cleaned_publication_date": "2024-12-13",
            "cleaned_author": "Test Author"
        }

        prompt = build_prompt(text, context=context, include_examples=False)

        assert "US-Israel Meeting" in prompt
        assert "2024-12-13" in prompt
        assert "Test Author" in prompt

    def test_build_prompt_with_domain(self):
        """Test prompt with domain specification."""
        text = "Military strike occurred"
        prompt = build_prompt(text, domain="geopolitical_conflict", include_examples=True)

        assert "geopolitical_conflict" in prompt.lower()
        assert text in prompt

    def test_build_prompt_without_domain(self):
        """Test prompt without specific domain."""
        text = "Test text"
        prompt = build_prompt(text, domain=None, include_examples=False)

        assert text in prompt
        # Should include general domain guidance
        assert len(prompt) > 0


class TestJSONExtraction:
    """Test JSON extraction from LLM output."""

    def test_extract_json_from_clean_output(self):
        """Test extracting JSON from clean output."""
        output = '{"events": []}'
        extracted = _extract_json_from_output(output)

        assert extracted == '{"events": []}'

    def test_extract_json_from_markdown(self):
        """Test extracting JSON from markdown code blocks."""
        output = '```json\n{"events": []}\n```'
        extracted = _extract_json_from_output(output)

        assert json.loads(extracted) == {"events": []}

    def test_extract_json_with_extra_text(self):
        """Test extracting JSON with surrounding text."""
        output = 'Here are the events:\n{"events": [{"event_type": "test"}]}\nDone!'
        extracted = _extract_json_from_output(output)

        data = json.loads(extracted)
        assert "events" in data


class TestSpanValidation:
    """Test character span validation."""

    def test_validate_span_valid(self):
        """Test valid span."""
        assert _validate_span(0, 10, 100) == True
        assert _validate_span(50, 60, 100) == True

    def test_validate_span_invalid_negative(self):
        """Test invalid negative positions."""
        assert _validate_span(-1, 10, 100) == False
        assert _validate_span(0, -5, 100) == False

    def test_validate_span_invalid_order(self):
        """Test invalid span order."""
        assert _validate_span(10, 5, 100) == False
        assert _validate_span(5, 5, 100) == False

    def test_validate_span_out_of_bounds(self):
        """Test out of bounds span."""
        assert _validate_span(0, 150, 100) == False


class TestEventParsing:
    """Test event parsing from LLM output."""

    def test_parse_single_event_complete(self):
        """Test parsing a complete event."""
        event_dict = {
            "event_type": "contact_meet",
            "trigger": {"text": "met", "start_char": 10, "end_char": 13},
            "arguments": [
                {
                    "role": "agent",
                    "text": "Biden",
                    "start_char": 0,
                    "end_char": 5,
                    "type": "PER"
                }
            ],
            "domain": "diplomatic_relations",
            "sentiment": "neutral",
            "causality": "Scheduled meeting"
        }

        original_text = "Biden met Netanyahu"
        event = _parse_single_event(event_dict, "doc_001", 0, original_text)

        assert event.event_type == "contact_meet"
        assert event.trigger.text == "met"
        assert len(event.arguments) == 1
        assert event.domain == "diplomatic_relations"

    def test_parse_single_event_minimal(self):
        """Test parsing event with minimal fields."""
        event_dict = {
            "event_type": "policy_announce",
            "trigger": {"text": "announced", "start_char": 5, "end_char": 14}
        }

        original_text = "Biden announced new policy"
        event = _parse_single_event(event_dict, "doc_001", 0, original_text)

        assert event.event_type == "policy_announce"
        assert len(event.arguments) == 0

    def test_parse_single_event_missing_required(self):
        """Test parsing event missing required fields."""
        event_dict = {
            "trigger": {"text": "met", "start_char": 0, "end_char": 3}
            # Missing event_type
        }

        with pytest.raises(ValueError):
            _parse_single_event(event_dict, "doc_001", 0, "test text")


class TestLLMOutputParsing:
    """Test full LLM output parsing."""

    def test_parse_llm_output_valid(self):
        """Test parsing valid LLM output."""
        output = '''
        {
            "events": [
                {
                    "event_type": "contact_meet",
                    "trigger": {"text": "met", "start_char": 6, "end_char": 9},
                    "arguments": [],
                    "domain": "diplomatic_relations",
                    "sentiment": "neutral"
                }
            ]
        }
        '''

        original_text = "Biden met Netanyahu"
        events = parse_llm_output(output, "doc_001", original_text)

        assert len(events) == 1
        assert events[0].event_type == "contact_meet"

    def test_parse_llm_output_empty_events(self):
        """Test parsing output with no events."""
        output = '{"events": []}'
        events = parse_llm_output(output, "doc_001", "test text")

        assert len(events) == 0

    def test_parse_llm_output_markdown(self):
        """Test parsing markdown-wrapped output."""
        output = '''```json
        {
            "events": [
                {
                    "event_type": "policy_announce",
                    "trigger": {"text": "announced", "start_char": 6, "end_char": 15}
                }
            ]
        }
        ```'''

        events = parse_llm_output(output, "doc_001", "Biden announced policy")
        assert len(events) == 1

    def test_parse_llm_output_invalid_json(self):
        """Test handling of invalid JSON."""
        output = "This is not JSON"

        with pytest.raises(ValueError):
            parse_llm_output(output, "doc_001", "test text")


class TestDomainAndEventTypes:
    """Test domain and event type helpers."""

    def test_list_supported_domains(self):
        """Test listing supported domains."""
        domains = list_supported_domains()

        assert "geopolitical_conflict" in domains
        assert "diplomatic_relations" in domains
        assert "technology_innovation" in domains
        assert len(domains) > 5

    def test_list_supported_event_types(self):
        """Test listing supported event types."""
        types = list_supported_event_types()

        assert "contact_meet" in types
        assert "policy_announce" in types
        assert "conflict_attack" in types
        assert len(types) > 10

    def test_get_domain_prompt(self):
        """Test getting domain-specific prompt."""
        prompt = get_domain_prompt("diplomatic_relations")

        assert prompt is not None
        assert "diplomatic" in prompt.lower()

    def test_get_event_type_description(self):
        """Test getting event type description."""
        desc = get_event_type_description("contact_meet")

        assert desc is not None
        assert "meet" in desc.lower() or "contact" in desc.lower()

    def test_get_invalid_domain(self):
        """Test getting invalid domain."""
        prompt = get_domain_prompt("nonexistent_domain")
        assert prompt is None
