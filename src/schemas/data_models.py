"""
data_models.py

Pydantic data models for Stage 2 NLP Processing Service.
Defines input/output schemas for API endpoints, internal processing, and storage.

Schema Design:
- Input: Compatible with Stage 1 (Cleaning Service) output
- Output: Compatible with Stage 3 (Embedding Generation) input
- Internal: Rich structures for NLP processing and event linking
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, date
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class EntityType(str, Enum):
    """Entity types for Named Entity Recognition."""
    PERSON = "PER"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    GEOPOLITICAL_ENTITY = "GPE"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    MISCELLANEOUS = "MISC"
    EVENT = "EVENT"


class DomainType(str, Enum):
    """Domain types for storyline distinction."""
    GEOPOLITICAL_CONFLICT = "geopolitical_conflict"
    DIPLOMATIC_RELATIONS = "diplomatic_relations"
    ECONOMIC_POLICY = "economic_policy"
    DOMESTIC_POLICY = "domestic_policy"
    ELECTIONS_POLITICS = "elections_politics"
    TECHNOLOGY_INNOVATION = "technology_innovation"
    SOCIAL_MOVEMENTS = "social_movements"
    ENVIRONMENTAL_CLIMATE = "environmental_climate"
    HEALTH_PANDEMIC = "health_pandemic"
    LEGAL_JUDICIAL = "legal_judicial"
    CORPORATE_BUSINESS = "corporate_business"
    CULTURAL_ENTERTAINMENT = "cultural_entertainment"


class SentimentType(str, Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


# =============================================================================
# INPUT MODELS (from Stage 1)
# =============================================================================

class Stage1Document(BaseModel):
    """
    Input document schema from Stage 1 (Cleaning Service).

    This matches the PreprocessSingleResponse from Stage 1.
    """
    document_id: str = Field(..., description="Unique document identifier")
    version: str = Field(default="1.0", description="Schema version")

    # Text fields (primary sources for NLP processing)
    original_text: Optional[str] = Field(None, description="Original unprocessed text")
    cleaned_text: str = Field(..., description="Cleaned and normalized text")

    # Metadata fields (used for context enrichment)
    cleaned_title: Optional[str] = Field(None, description="Cleaned article title")
    cleaned_excerpt: Optional[str] = Field(None, description="Cleaned excerpt/summary")
    cleaned_author: Optional[str] = Field(None, description="Cleaned author name")
    cleaned_publication_date: Optional[str] = Field(None, description="Publication date (ISO 8601)")
    cleaned_revision_date: Optional[str] = Field(None, description="Revision date")
    cleaned_source_url: Optional[str] = Field(None, description="Source URL")
    cleaned_categories: Optional[List[str]] = Field(None, description="Article categories")
    cleaned_tags: Optional[List[str]] = Field(None, description="Article tags")
    cleaned_word_count: Optional[int] = Field(None, description="Word count")
    cleaned_publisher: Optional[str] = Field(None, description="Publisher name")

    # Additional fields
    temporal_metadata: Optional[str] = Field(None, description="Normalized temporal reference")
    entities: Optional[List[Dict[str, Any]]] = Field(None, description="Basic entities from Stage 1")
    cleaned_additional_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# =============================================================================
# ENTITY MODELS
# =============================================================================

class EntitySpan(BaseModel):
    """Represents a text span with position information."""
    text: str = Field(..., description="The entity text")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position (exclusive)")

    @field_validator('end_char')
    @classmethod
    def validate_end_char(cls, v, info):
        """Ensure end_char > start_char."""
        if 'start_char' in info.data and v <= info.data['start_char']:
            raise ValueError("end_char must be greater than start_char")
        return v


class Entity(BaseModel):
    """
    Represents an extracted named entity.

    Includes entity type, position, confidence, and optional context.
    """
    text: str = Field(..., description="Entity text")
    type: str = Field(..., description="Entity type (PER, ORG, LOC, etc.)")
    start_char: int = Field(..., description="Start position in document")
    end_char: int = Field(..., description="End position in document")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")

    # Context for disambiguation
    context: Optional[str] = Field(None, description="Surrounding context for disambiguation")
    normalized_form: Optional[str] = Field(None, description="Normalized/canonical form")
    entity_id: Optional[str] = Field(None, description="Unique entity identifier (for linking)")


# =============================================================================
# DEPENDENCY PARSING MODELS
# =============================================================================

class DependencyRelation(BaseModel):
    """Represents a dependency relation between tokens."""
    head: str = Field(..., description="Head token")
    head_pos: str = Field(..., description="Head part-of-speech")
    dependent: str = Field(..., description="Dependent token")
    dependent_pos: str = Field(..., description="Dependent part-of-speech")
    relation: str = Field(..., description="Dependency relation type")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SOATriplet(BaseModel):
    """
    Subject-Object-Action triplet extracted from dependency parsing.

    Represents basic relational information in the text.
    """
    subject: EntitySpan = Field(..., description="Subject entity")
    action: EntitySpan = Field(..., description="Action/verb")
    object: EntitySpan = Field(..., description="Object entity")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Optional context
    sentence: Optional[str] = Field(None, description="Source sentence")
    dependencies: Optional[List[DependencyRelation]] = Field(None, description="Underlying dependencies")


# =============================================================================
# EVENT EXTRACTION MODELS
# =============================================================================

class EventArgument(BaseModel):
    """
    Represents an argument of an event (participant, location, time, etc.).
    """
    argument_role: str = Field(..., description="Role (agent, patient, time, place, etc.)")
    entity: Entity = Field(..., description="Entity filling this role")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class EventTrigger(BaseModel):
    """Represents the trigger word/phrase that indicates an event."""
    text: str = Field(..., description="Trigger text")
    start_char: int = Field(..., description="Start position")
    end_char: int = Field(..., description="End position")
    lemma: Optional[str] = Field(None, description="Lemmatized form")


class EventMetadata(BaseModel):
    """Additional metadata for an event."""
    sentiment: Optional[str] = Field(None, description="Event sentiment")
    causality: Optional[str] = Field(None, description="Causal explanation or link")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall event confidence")
    source_sentence: Optional[str] = Field(None, description="Source sentence")
    source_chunk_index: Optional[int] = Field(None, description="Chunk index (for long documents)")


class Event(BaseModel):
    """
    Represents an extracted event with type, trigger, arguments, and metadata.

    This is the core output of the Event LLM service.
    """
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Event type (ACE 2005 + extended)")
    trigger: EventTrigger = Field(..., description="Event trigger")
    arguments: List[EventArgument] = Field(default_factory=list, description="Event arguments")
    metadata: EventMetadata = Field(..., description="Event metadata")

    # Domain classification for storyline distinction
    domain: Optional[str] = Field(None, description="Domain type for storyline grouping")
    domain_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Temporal information
    temporal_reference: Optional[str] = Field(None, description="Temporal reference (ISO 8601 or relative)")

    # Linking information (populated by event linker)
    linked_event_ids: Optional[List[str]] = Field(None, description="IDs of linked co-referent events")
    storyline_id: Optional[str] = Field(None, description="Storyline/cluster identifier")


# =============================================================================
# EVENT LINKING MODELS
# =============================================================================

class EventLinkage(BaseModel):
    """Represents a link between two events."""
    source_event_id: str = Field(..., description="Source event ID")
    target_event_id: str = Field(..., description="Target event ID")
    link_type: str = Field(..., description="Link type (coreference, causality, temporal_sequence, etc.)")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")

    # Multi-dimensional scores
    semantic_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    entity_overlap: Optional[float] = Field(None, ge=0.0, le=1.0)
    temporal_proximity: Optional[float] = Field(None, ge=0.0, le=1.0)
    domain_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)


class Storyline(BaseModel):
    """
    Represents a distinct storyline containing related events.

    Used for storyline distinction (e.g., separate Trump+Israel vs Trump+Qatar).
    """
    storyline_id: str = Field(..., description="Unique storyline identifier")
    event_ids: List[str] = Field(..., description="Event IDs in this storyline")

    # Storyline characteristics
    primary_entities: List[str] = Field(..., description="Key entities defining this storyline")
    domain: Optional[str] = Field(None, description="Primary domain")
    temporal_span: Optional[Tuple[str, str]] = Field(None, description="(start_date, end_date)")

    # Summary information
    storyline_summary: Optional[str] = Field(None, description="Brief summary of storyline")
    key_events: Optional[List[str]] = Field(None, description="IDs of key/pivotal events")


# =============================================================================
# SERVICE RESPONSE MODELS
# =============================================================================

class NERServiceResponse(BaseModel):
    """Response from NER service."""
    document_id: str = Field(..., description="Document ID")
    entities: List[Entity] = Field(..., description="Extracted entities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_name: str = Field(..., description="NER model used")


class DPServiceResponse(BaseModel):
    """Response from Dependency Parsing service."""
    document_id: str = Field(..., description="Document ID")
    soa_triplets: List[SOATriplet] = Field(..., description="Subject-Object-Action triplets")
    dependencies: Optional[List[DependencyRelation]] = Field(None, description="Raw dependencies")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_name: str = Field(..., description="DP model used")


class EventLLMServiceResponse(BaseModel):
    """Response from Event LLM service."""
    document_id: str = Field(..., description="Document ID")
    events: List[Event] = Field(..., description="Extracted events")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_name: str = Field(..., description="LLM model used")
    chunks_processed: int = Field(default=1, description="Number of chunks processed")


class EventLinkingResponse(BaseModel):
    """Response from Event Linking service."""
    batch_id: str = Field(..., description="Batch ID")
    linkages: List[EventLinkage] = Field(..., description="Event linkages")
    storylines: List[Storyline] = Field(..., description="Identified storylines")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# =============================================================================
# ORCHESTRATOR OUTPUT (Stage 2 Output to Stage 3)
# =============================================================================

class ProcessedDocument(BaseModel):
    """
    Final output from Stage 2 NLP Processing Service.

    This is the guaranteed contract for Stage 3 (Embedding Generation).
    """
    # Identifiers
    document_id: str = Field(..., description="Unique document identifier")
    job_id: Optional[str] = Field(None, description="Processing job identifier")

    # Timestamps
    processed_at: str = Field(..., description="Processing timestamp (ISO 8601)")
    normalized_date: Optional[str] = Field(None, description="Normalized publication date")

    # Original content (preserved from Stage 1)
    original_text: str = Field(..., description="Original cleaned text from Stage 1")
    source_document: Dict[str, Any] = Field(..., description="Complete Stage 1 document (pass-through)")

    # Extracted NLP artifacts
    extracted_entities: List[Entity] = Field(..., description="Extracted named entities")
    extracted_soa_triplets: List[SOATriplet] = Field(..., description="Subject-Object-Action triplets")
    events: List[Event] = Field(..., description="Extracted events with arguments and metadata")

    # Linking and storyline information
    event_linkages: Optional[List[EventLinkage]] = Field(None, description="Event linkages")
    storylines: Optional[List[Storyline]] = Field(None, description="Identified storylines")

    # Processing metadata
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics and metadata")


# =============================================================================
# API REQUEST/RESPONSE MODELS
# =============================================================================

class ProcessDocumentRequest(BaseModel):
    """Request to process a single document."""
    document: Stage1Document = Field(..., description="Document from Stage 1")
    options: Optional[Dict[str, Any]] = Field(None, description="Processing options")


class ProcessDocumentResponse(BaseModel):
    """Response from processing a single document."""
    success: bool = Field(..., description="Processing success status")
    document_id: str = Field(..., description="Document ID")
    result: Optional[ProcessedDocument] = Field(None, description="Processed document")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(..., description="Total processing time")


class ProcessBatchRequest(BaseModel):
    """Request to process a batch of documents."""
    documents: List[Stage1Document] = Field(..., description="Documents to process")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")
    options: Optional[Dict[str, Any]] = Field(None, description="Processing options")


class ProcessBatchResponse(BaseModel):
    """Response from batch processing submission."""
    success: bool = Field(..., description="Submission success")
    batch_id: str = Field(..., description="Batch identifier")
    job_id: str = Field(..., description="Celery job ID for tracking")
    document_count: int = Field(..., description="Number of documents submitted")
    message: str = Field(..., description="Status message")


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status (PENDING, STARTED, SUCCESS, FAILURE)")
    progress: Optional[float] = Field(None, ge=0.0, le=100.0, description="Progress percentage")
    documents_processed: Optional[int] = Field(None, description="Number of documents processed")
    documents_total: Optional[int] = Field(None, description="Total documents in job")
    result: Optional[Dict[str, Any]] = Field(None, description="Result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status (ok, degraded, error)")
    services: Dict[str, Dict[str, Any]] = Field(..., description="Individual service statuses")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(default="1.0.0", description="API version")


# =============================================================================
# UTILITY MODELS
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_document_id(document_id: str) -> bool:
    """
    Validate document ID format.

    Args:
        document_id: Document identifier

    Returns:
        True if valid, False otherwise
    """
    if not document_id or len(document_id) < 1:
        return False
    # Add additional validation as needed
    return True


def create_event_id(document_id: str, event_index: int) -> str:
    """
    Create unique event ID.

    Args:
        document_id: Document identifier
        event_index: Event index within document

    Returns:
        Unique event ID
    """
    return f"{document_id}_event_{event_index}"


def create_storyline_id(batch_id: str, cluster_index: int) -> str:
    """
    Create unique storyline ID.

    Args:
        batch_id: Batch identifier
        cluster_index: Cluster index

    Returns:
        Unique storyline ID
    """
    return f"{batch_id}_storyline_{cluster_index}"


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    # Test model creation and validation
    import json

    # Test Entity model
    entity = Entity(
        text="Donald Trump",
        type="PER",
        start_char=0,
        end_char=12,
        confidence=0.95,
        context="Donald Trump announced a new policy"
    )
    print("Entity model:", json.dumps(entity.model_dump(), indent=2))

    # Test Event model
    trigger = EventTrigger(text="announced", start_char=13, end_char=22)
    argument = EventArgument(
        argument_role="agent",
        entity=entity,
        confidence=0.9
    )
    event = Event(
        event_id="doc1_event_0",
        event_type="policy_announce",
        trigger=trigger,
        arguments=[argument],
        metadata=EventMetadata(sentiment="neutral", confidence=0.85),
        domain="domestic_policy"
    )
    print("\nEvent model:", json.dumps(event.model_dump(), indent=2))

    print("\nAll models validated successfully!")
