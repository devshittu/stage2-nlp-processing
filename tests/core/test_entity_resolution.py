"""
Unit tests for entity resolution and deduplication.
"""
import pytest
from src.core.entity_resolution import (
    normalize_entity_text,
    get_tokens,
    is_substring_match,
    token_overlap_ratio,
    string_similarity,
    are_entities_similar,
    select_canonical_form,
    group_similar_entities,
    resolve_entities
)
from src.schemas.data_models import Entity


class TestTextNormalization:
    """Test text normalization functions."""

    def test_normalize_entity_text(self):
        """Test entity text normalization."""
        assert normalize_entity_text("  Apple Inc.  ") == "apple inc."
        assert normalize_entity_text("Joe  Biden") == "joe biden"
        assert normalize_entity_text("NEW YORK") == "new york"

    def test_get_tokens(self):
        """Test token extraction."""
        tokens = get_tokens("President Biden")
        assert tokens == {"president", "biden"}

        tokens = get_tokens("Apple Inc.")
        assert "apple" in tokens
        assert "inc" in tokens


class TestStringMatching:
    """Test string matching functions."""

    def test_is_substring_match(self):
        """Test substring matching."""
        assert is_substring_match("Biden", "Joe Biden") == True
        assert is_substring_match("President Biden", "Biden") == True
        assert is_substring_match("Trump", "Biden") == False

    def test_token_overlap_ratio(self):
        """Test token overlap calculation."""
        ratio = token_overlap_ratio("President Biden", "Joe Biden")
        assert ratio > 0.3  # "Biden" is common

        ratio = token_overlap_ratio("Apple", "Microsoft")
        assert ratio == 0.0

    def test_string_similarity(self):
        """Test string similarity."""
        sim = string_similarity("Biden", "Biden")
        assert sim == 1.0

        sim = string_similarity("Biden", "Bden")
        assert sim > 0.7  # Similar

        sim = string_similarity("Biden", "Trump")
        assert sim < 0.5


class TestEntitySimilarity:
    """Test entity similarity matching."""

    def test_are_entities_similar_exact_match(self):
        """Test exact match."""
        e1 = Entity(text="Biden", type="PER", start_char=0, end_char=5, confidence=0.9)
        e2 = Entity(text="Biden", type="PER", start_char=10, end_char=15, confidence=0.9)

        assert are_entities_similar(e1, e2) == True

    def test_are_entities_similar_substring(self):
        """Test substring match."""
        e1 = Entity(text="Joe Biden", type="PER", start_char=0, end_char=9, confidence=0.9)
        e2 = Entity(text="Biden", type="PER", start_char=20, end_char=25, confidence=0.9)

        assert are_entities_similar(e1, e2) == True

    def test_are_entities_similar_different_type(self):
        """Test that different types don't match."""
        e1 = Entity(text="Biden", type="PER", start_char=0, end_char=5, confidence=0.9)
        e2 = Entity(text="Biden", type="ORG", start_char=10, end_char=15, confidence=0.9)

        assert are_entities_similar(e1, e2) == False

    def test_are_entities_similar_token_overlap(self):
        """Test token overlap matching."""
        e1 = Entity(text="President Biden", type="PER", start_char=0, end_char=15, confidence=0.9)
        e2 = Entity(text="Joe Biden", type="PER", start_char=20, end_char=29, confidence=0.9)

        assert are_entities_similar(e1, e2, token_overlap_threshold=0.3) == True


class TestCanonicalFormSelection:
    """Test canonical form selection."""

    def test_select_canonical_form_longest(self):
        """Test selection prefers longest text."""
        entities = [
            Entity(text="Biden", type="PER", start_char=0, end_char=5, confidence=0.9),
            Entity(text="Joe Biden", type="PER", start_char=10, end_char=19, confidence=0.9),
            Entity(text="President Biden", type="PER", start_char=30, end_char=45, confidence=0.9)
        ]

        canonical = select_canonical_form(entities)
        assert canonical.text == "President Biden"

    def test_select_canonical_form_confidence(self):
        """Test selection considers confidence."""
        entities = [
            Entity(text="Apple", type="ORG", start_char=0, end_char=5, confidence=0.7),
            Entity(text="Apple", type="ORG", start_char=10, end_char=15, confidence=0.95)
        ]

        canonical = select_canonical_form(entities)
        assert canonical.confidence == 0.95


class TestEntityGrouping:
    """Test entity grouping."""

    def test_group_similar_entities(self):
        """Test grouping similar entities."""
        entities = [
            Entity(text="Biden", type="PER", start_char=0, end_char=5, confidence=0.9),
            Entity(text="Joe Biden", type="PER", start_char=10, end_char=19, confidence=0.9),
            Entity(text="Trump", type="PER", start_char=30, end_char=35, confidence=0.9),
            Entity(text="Apple", type="ORG", start_char=50, end_char=55, confidence=0.9)
        ]

        groups = group_similar_entities(entities)

        # Should have 3 groups: Biden variants, Trump, Apple
        assert len(groups) == 3

        # Find Biden group
        biden_group = [g for g in groups if any(e.text == "Joe Biden" for e in g)][0]
        assert len(biden_group) == 2


class TestEntityResolution:
    """Test complete entity resolution."""

    def test_resolve_entities_keep_all_mentions(self):
        """Test resolution keeping all mentions with normalized form."""
        entities = [
            Entity(text="Joe Biden", type="PER", start_char=0, end_char=9, confidence=0.9),
            Entity(text="Biden", type="PER", start_char=20, end_char=25, confidence=0.9),
            Entity(text="President Biden", type="PER", start_char=40, end_char=55, confidence=0.9)
        ]

        resolved = resolve_entities(entities, keep_all_mentions=True)

        # Should keep all 3 entities
        assert len(resolved) == 3

        # All should have normalized_form set to canonical form
        for entity in resolved:
            assert entity.normalized_form == "President Biden"

    def test_resolve_entities_canonical_only(self):
        """Test resolution returning only canonical forms."""
        entities = [
            Entity(text="Biden", type="PER", start_char=0, end_char=5, confidence=0.9),
            Entity(text="Joe Biden", type="PER", start_char=10, end_char=19, confidence=0.9),
            Entity(text="Trump", type="PER", start_char=30, end_char=35, confidence=0.9)
        ]

        resolved = resolve_entities(entities, keep_all_mentions=False)

        # Should return 2 canonical entities
        assert len(resolved) == 2

    def test_resolve_entities_empty_list(self):
        """Test resolution with empty list."""
        resolved = resolve_entities([])
        assert resolved == []

    def test_resolve_entities_different_types(self):
        """Test that different entity types are kept separate."""
        entities = [
            Entity(text="Biden", type="PER", start_char=0, end_char=5, confidence=0.9),
            Entity(text="Biden", type="ORG", start_char=10, end_char=15, confidence=0.9)
        ]

        resolved = resolve_entities(entities, keep_all_mentions=False)

        # Should keep both as they are different types
        assert len(resolved) == 2
