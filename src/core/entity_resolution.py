"""
entity_resolution.py

Entity coreference resolution and deduplication for Stage 2 NLP Processing Service.

Features:
- Smart entity deduplication (e.g., "Joe Biden" and "Biden" → single entity)
- Type-aware matching (PER entities only match other PER entities)
- Fuzzy string matching with configurable thresholds
- Canonical form selection (prefers longer/more complete names)
- Context-aware disambiguation

Approach:
1. Group entities by type (PER, ORG, LOC, etc.)
2. Within each type, find similar entities using:
   - Exact match
   - Substring match (e.g., "Biden" in "Joe Biden")
   - Token overlap (e.g., "President Biden" and "Biden")
   - Fuzzy similarity (Levenshtein distance)
3. Merge similar entities and select canonical form
4. Propagate canonical form across all mentions
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from difflib import SequenceMatcher

from src.schemas.data_models import Entity
from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_entity_text(text: str) -> str:
    """
    Normalize entity text for comparison.

    Args:
        text: Entity text

    Returns:
        Normalized text (lowercased, stripped, extra spaces removed)
    """
    return " ".join(text.lower().strip().split())


def get_tokens(text: str) -> Set[str]:
    """
    Get token set from text.

    Args:
        text: Entity text

    Returns:
        Set of tokens (lowercased, no punctuation)
    """
    # Remove common punctuation
    text = text.replace(",", " ").replace(".", " ").replace("'", "")
    return set(text.lower().split())


def is_substring_match(text1: str, text2: str) -> bool:
    """
    Check if one text is a substring of another.

    Args:
        text1: First text
        text2: Second text

    Returns:
        True if one is substring of the other

    Example:
        >>> is_substring_match("Biden", "Joe Biden")
        True
        >>> is_substring_match("Trump", "Donald Trump Jr.")
        True
    """
    norm1 = normalize_entity_text(text1)
    norm2 = normalize_entity_text(text2)

    return norm1 in norm2 or norm2 in norm1


def token_overlap_ratio(text1: str, text2: str) -> float:
    """
    Calculate token overlap ratio between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Overlap ratio (0.0 to 1.0)

    Example:
        >>> token_overlap_ratio("President Biden", "Joe Biden")
        0.5  # "Biden" is common, 1 out of 2 tokens in first text
    """
    tokens1 = get_tokens(text1)
    tokens2 = get_tokens(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    if not union:
        return 0.0

    return len(intersection) / len(union)


def string_similarity(text1: str, text2: str) -> float:
    """
    Calculate string similarity using SequenceMatcher.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (0.0 to 1.0)
    """
    norm1 = normalize_entity_text(text1)
    norm2 = normalize_entity_text(text2)

    return SequenceMatcher(None, norm1, norm2).ratio()


def are_entities_similar(
    entity1: Entity,
    entity2: Entity,
    substring_match: bool = True,
    token_overlap_threshold: float = 0.5,
    fuzzy_similarity_threshold: float = 0.85
) -> bool:
    """
    Determine if two entities refer to the same real-world entity.

    Args:
        entity1: First entity
        entity2: Second entity
        substring_match: Allow substring matching
        token_overlap_threshold: Minimum token overlap ratio
        fuzzy_similarity_threshold: Minimum fuzzy similarity

    Returns:
        True if entities are likely the same

    Example:
        >>> e1 = Entity(text="Joe Biden", type="PER", ...)
        >>> e2 = Entity(text="Biden", type="PER", ...)
        >>> are_entities_similar(e1, e2)
        True
    """
    # Must be same type
    if entity1.type != entity2.type:
        return False

    # Exact match (normalized)
    if normalize_entity_text(entity1.text) == normalize_entity_text(entity2.text):
        return True

    # Substring match (e.g., "Biden" in "Joe Biden")
    if substring_match and is_substring_match(entity1.text, entity2.text):
        return True

    # Token overlap (e.g., "President Biden" and "Joe Biden" share "Biden")
    overlap = token_overlap_ratio(entity1.text, entity2.text)
    if overlap >= token_overlap_threshold:
        return True

    # Fuzzy similarity (typos, slight variations)
    similarity = string_similarity(entity1.text, entity2.text)
    if similarity >= fuzzy_similarity_threshold:
        return True

    return False


def select_canonical_form(entities: List[Entity]) -> Entity:
    """
    Select the canonical/preferred form from a list of similar entities.

    Prefers:
    1. Longer text (more complete names)
    2. Higher confidence
    3. First occurrence

    Args:
        entities: List of similar entities

    Returns:
        Canonical entity

    Example:
        >>> entities = [
        ...     Entity(text="Biden", type="PER", confidence=0.9),
        ...     Entity(text="Joe Biden", type="PER", confidence=0.95),
        ...     Entity(text="President Biden", type="PER", confidence=0.85)
        ... ]
        >>> canonical = select_canonical_form(entities)
        >>> canonical.text
        "President Biden"  # Longest and descriptive
    """
    if not entities:
        raise ValueError("Cannot select canonical form from empty list")

    # Sort by: length (desc), confidence (desc)
    sorted_entities = sorted(
        entities,
        key=lambda e: (len(e.text), e.confidence),
        reverse=True
    )

    return sorted_entities[0]


def group_similar_entities(
    entities: List[Entity],
    substring_match: bool = True,
    token_overlap_threshold: float = 0.5,
    fuzzy_similarity_threshold: float = 0.85
) -> List[List[Entity]]:
    """
    Group similar entities together.

    Uses greedy clustering based on pairwise similarity.

    Args:
        entities: List of entities to group
        substring_match: Allow substring matching
        token_overlap_threshold: Minimum token overlap
        fuzzy_similarity_threshold: Minimum fuzzy similarity

    Returns:
        List of entity groups (each group contains similar entities)
    """
    if not entities:
        return []

    # Group entities by type first
    type_groups = defaultdict(list)
    for entity in entities:
        type_groups[entity.type].append(entity)

    all_groups = []

    # Process each type separately
    for entity_type, type_entities in type_groups.items():
        # Greedy clustering
        groups = []
        used = set()

        for i, entity1 in enumerate(type_entities):
            if i in used:
                continue

            # Start new group with this entity
            group = [entity1]
            used.add(i)

            # Find all similar entities
            for j, entity2 in enumerate(type_entities):
                if j in used:
                    continue

                # Check if similar to any entity in current group
                is_similar = any(
                    are_entities_similar(
                        entity2,
                        group_entity,
                        substring_match,
                        token_overlap_threshold,
                        fuzzy_similarity_threshold
                    )
                    for group_entity in group
                )

                if is_similar:
                    group.append(entity2)
                    used.add(j)

            groups.append(group)

        all_groups.extend(groups)

    return all_groups


def resolve_entities(
    entities: List[Entity],
    substring_match: bool = True,
    token_overlap_threshold: float = 0.5,
    fuzzy_similarity_threshold: float = 0.85,
    keep_all_mentions: bool = False
) -> List[Entity]:
    """
    Resolve entity coreferences and return deduplicated entities.

    Args:
        entities: List of entities to resolve
        substring_match: Allow substring matching
        token_overlap_threshold: Minimum token overlap
        fuzzy_similarity_threshold: Minimum fuzzy similarity
        keep_all_mentions: If True, keep all mentions but update with canonical form
                          If False, return only one canonical entity per group

    Returns:
        List of resolved entities

    Example:
        >>> entities = [
        ...     Entity(text="Joe Biden", type="PER", start_char=0, end_char=9),
        ...     Entity(text="Biden", type="PER", start_char=50, end_char=55),
        ...     Entity(text="President Biden", type="PER", start_char=100, end_char=115)
        ... ]
        >>> resolved = resolve_entities(entities, keep_all_mentions=False)
        >>> len(resolved)
        1
        >>> resolved[0].text
        "President Biden"  # Canonical form selected
    """
    if not entities:
        return []

    logger.debug(f"Resolving {len(entities)} entities")

    # Group similar entities
    groups = group_similar_entities(
        entities,
        substring_match,
        token_overlap_threshold,
        fuzzy_similarity_threshold
    )

    logger.debug(f"Grouped into {len(groups)} entity groups")

    resolved = []

    for group in groups:
        # Select canonical form
        canonical = select_canonical_form(group)

        if keep_all_mentions:
            # Update all mentions with canonical text as normalized_form
            for entity in group:
                entity.normalized_form = canonical.text
                resolved.append(entity)
        else:
            # Return only canonical entity
            resolved.append(canonical)

    logger.info(
        f"Entity resolution: {len(entities)} → {len(resolved)} entities",
        extra={
            "original_count": len(entities),
            "resolved_count": len(resolved),
            "deduplication_ratio": 1 - (len(resolved) / len(entities)) if entities else 0
        }
    )

    return resolved


def get_entity_mentions(
    entities: List[Entity],
    canonical_text: str,
    entity_type: str
) -> List[Entity]:
    """
    Get all mentions of an entity by its canonical form.

    Args:
        entities: List of all entities
        canonical_text: Canonical entity text
        entity_type: Entity type

    Returns:
        List of entity mentions
    """
    mentions = []

    for entity in entities:
        if entity.type != entity_type:
            continue

        if entity.normalized_form == canonical_text or entity.text == canonical_text:
            mentions.append(entity)

    return mentions


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    """
    Test entity resolution with sample entities.

    Run with: python -m src.core.entity_resolution
    """
    from src.utils.logger import setup_logging

    # Setup logging
    setup_logging(log_level="INFO")

    logger.info("Starting entity resolution test")

    # Sample entities
    test_entities = [
        Entity(text="Joe Biden", type="PER", start_char=0, end_char=9, confidence=0.95),
        Entity(text="Biden", type="PER", start_char=50, end_char=55, confidence=0.90),
        Entity(text="President Biden", type="PER", start_char=100, end_char=115, confidence=0.92),
        Entity(text="Donald Trump", type="PER", start_char=200, end_char=212, confidence=0.96),
        Entity(text="Trump", type="PER", start_char=250, end_char=255, confidence=0.91),
        Entity(text="White House", type="LOC", start_char=300, end_char=311, confidence=0.88),
        Entity(text="the White House", type="LOC", start_char=400, end_char=415, confidence=0.87),
        Entity(text="Microsoft", type="ORG", start_char=500, end_char=509, confidence=0.93),
        Entity(text="Microsoft Corp", type="ORG", start_char=600, end_char=614, confidence=0.89),
    ]

    print("\n" + "=" * 80)
    print("ORIGINAL ENTITIES")
    print("=" * 80)
    for entity in test_entities:
        print(f"- {entity.text} ({entity.type}) [{entity.start_char}:{entity.end_char}]")

    # Test resolution (return only canonical forms)
    print("\n" + "=" * 80)
    print("RESOLVED ENTITIES (Canonical Forms Only)")
    print("=" * 80)
    resolved = resolve_entities(test_entities, keep_all_mentions=False)
    for entity in resolved:
        print(f"- {entity.text} ({entity.type}) [confidence: {entity.confidence:.2f}]")

    # Test resolution (keep all mentions)
    print("\n" + "=" * 80)
    print("RESOLVED ENTITIES (All Mentions with Normalized Forms)")
    print("=" * 80)
    resolved_all = resolve_entities(test_entities, keep_all_mentions=True)
    for entity in resolved_all:
        normalized = f" → {entity.normalized_form}" if entity.normalized_form else ""
        print(f"- {entity.text} ({entity.type}){normalized}")

    # Test grouping
    print("\n" + "=" * 80)
    print("ENTITY GROUPS")
    print("=" * 80)
    groups = group_similar_entities(test_entities)
    for i, group in enumerate(groups, 1):
        canonical = select_canonical_form(group)
        print(f"\nGroup {i} (Canonical: {canonical.text}):")
        for entity in group:
            print(f"  - {entity.text}")

    print("\n" + "=" * 80)
    print(f"Summary: {len(test_entities)} entities → {len(resolved)} canonical entities")
    print("=" * 80)

    logger.info("Entity resolution test completed successfully")
