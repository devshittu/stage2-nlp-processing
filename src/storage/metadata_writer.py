"""
Metadata writer integration for Stage 2 NLP Processing.

Writes extracted metadata to the shared metadata registry for cross-stage access.
This is a DUAL-WRITE pattern: existing storage backends continue to work unchanged.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

import structlog

# Import shared metadata registry
if TYPE_CHECKING:
    from shared_metadata_registry import RegistryConfig

try:
    from shared_metadata_registry import (
        DocumentMetadata,
        EntityMetadata,
        EventMetadata,
        JobRegistration,
        JobStatus,
        MetadataRegistry,
        RegistryConfig,
        StorylineMetadata,
    )

    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    # Define all types as None when import fails to prevent NameError
    DocumentMetadata = None
    EntityMetadata = None
    EventMetadata = None
    JobRegistration = None
    JobStatus = None
    MetadataRegistry = None
    RegistryConfig = None
    StorylineMetadata = None
    structlog.get_logger(__name__).warning(
        "shared_metadata_registry not installed - metadata registry disabled"
    )

logger = structlog.get_logger(__name__)


class MetadataWriter:
    """
    Metadata writer for Stage 2 NLP Processing.

    Integrates with shared metadata registry to write extracted metadata
    for consumption by downstream stages (especially Stage 5 Graph Service).

    Features:
    - Dual-write pattern (existing backends + registry)
    - Graceful degradation if registry unavailable
    - Zero-regression guarantee
    - Configurable via environment variables
    """

    def __init__(self, config: Optional[RegistryConfig] = None):
        """
        Initialize metadata writer.

        Args:
            config: Optional registry configuration (defaults to env-based)
        """
        self.registry: Optional[MetadataRegistry] = None
        self.enabled = False
        self.initialized = False

        if not REGISTRY_AVAILABLE:
            logger.warning("Metadata registry library not available - writer disabled")
            return

        # Check if enabled via environment
        self.enabled = os.getenv("METADATA_REGISTRY_ENABLED", "false").lower() == "true"

        if not self.enabled:
            logger.info("Metadata registry disabled by configuration")
            return

        # Initialize registry
        try:
            self.registry = MetadataRegistry(config)
            logger.info("Metadata writer initialized (registry pending connection)")
        except Exception as e:
            logger.error(f"Failed to initialize metadata writer: {e}")
            self.enabled = False

    async def connect(self) -> bool:
        """
        Connect to metadata registry backends.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.enabled or not self.registry:
            return False

        try:
            success = await self.registry.initialize()
            self.initialized = success

            if success:
                logger.info("Metadata writer connected to registry")
            else:
                logger.warning("Metadata registry initialization failed - continuing without")

            return success
        except Exception as e:
            logger.error(f"Failed to connect metadata writer: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from metadata registry."""
        if self.registry and self.initialized:
            await self.registry.shutdown()
            logger.info("Metadata writer disconnected")

    async def register_job(
        self,
        job_id: UUID,
        batch_id: Optional[str] = None,
        parent_job_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register Stage 2 job in the metadata registry.

        Args:
            job_id: Stage 2 job identifier
            batch_id: Shared batch identifier across stages
            parent_job_id: Optional parent job (from Stage 1)
            metadata: Additional job metadata

        Returns:
            True if registration successful, False otherwise
        """
        if not self._is_ready():
            return False

        try:
            job = JobRegistration(
                job_id=job_id,
                batch_id=batch_id,
                stage=2,
                stage_name="nlp",
                parent_job_id=parent_job_id,
                status=JobStatus.RUNNING,
                metadata=metadata or {},
            )

            success = await self.registry.register_job(job)

            if success:
                logger.info(
                    "job_registered_in_metadata_registry",
                    job_id=str(job_id),
                    batch_id=batch_id,
                    stage=2,
                )
            else:
                logger.warning(
                    "job_registration_failed",
                    job_id=str(job_id),
                    batch_id=batch_id,
                )

            return success
        except Exception as e:
            logger.error(f"Failed to register job {job_id}: {e}")
            return False

    async def update_job_status(
        self,
        job_id: UUID,
        status: str,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update job status in registry.

        Args:
            job_id: Job identifier
            status: New status (queued, running, completed, failed, etc.)
            error_message: Optional error message for failed jobs

        Returns:
            True if update successful
        """
        if not self._is_ready():
            return False

        try:
            success = await self.registry.update_job_status(job_id, status, error_message)

            if success:
                logger.info(
                    "job_status_updated",
                    job_id=str(job_id),
                    status=status,
                )

            return success
        except Exception as e:
            logger.error(f"Failed to update job status {job_id}: {e}")
            return False

    async def write_document_metadata(
        self,
        document_id: str,
        job_id: UUID,
        batch_id: Optional[str],
        source_document: Dict[str, Any],
    ) -> bool:
        """
        Write document metadata to registry.

        Args:
            document_id: Document identifier
            job_id: Processing job ID
            batch_id: Batch identifier
            source_document: Full source document data

        Returns:
            True if write successful
        """
        if not self._is_ready():
            return False

        try:
            # Parse publication date if present
            pub_date = source_document.get("cleaned_publication_date")
            if pub_date and isinstance(pub_date, str):
                try:
                    pub_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                except ValueError:
                    pub_date = None

            document = DocumentMetadata(
                document_id=document_id,
                job_id=job_id,
                batch_id=batch_id,
                title=source_document.get("cleaned_title"),
                author=source_document.get("cleaned_author"),
                publication_date=pub_date,
                source_url=source_document.get("cleaned_url"),
                publisher=source_document.get("cleaned_publisher"),
                word_count=source_document.get("word_count"),
                categories=source_document.get("cleaned_categories", []),
                tags=source_document.get("cleaned_tags", []),
                full_data=source_document,
            )

            success = await self.registry.store_document(document)

            if success:
                logger.debug(
                    "document_metadata_written",
                    document_id=document_id,
                    job_id=str(job_id),
                )

            return success
        except Exception as e:
            logger.error(f"Failed to write document metadata {document_id}: {e}")
            return False

    async def write_event_metadata(
        self,
        event_id: str,
        document_id: str,
        job_id: UUID,
        batch_id: Optional[str],
        event_data: Dict[str, Any],
    ) -> bool:
        """
        Write event metadata to registry.

        Args:
            event_id: Event identifier
            document_id: Parent document ID
            job_id: Processing job ID
            batch_id: Batch identifier
            event_data: Full event data from NLP extraction

        Returns:
            True if write successful
        """
        if not self._is_ready():
            return False

        try:
            # Parse event date if present
            event_date = event_data.get("date")
            if event_date and isinstance(event_date, str):
                try:
                    event_date = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
                except ValueError:
                    event_date = None

            event = EventMetadata(
                event_id=event_id,
                document_id=document_id,
                job_id=job_id,
                batch_id=batch_id,
                event_type=event_data.get("type"),
                event_date=event_date,
                event_date_text=event_data.get("date_text"),
                description=event_data.get("description"),
                participants=event_data.get("participants", []),
                location=event_data.get("location"),
                full_data=event_data,
            )

            success = await self.registry.store_event(event)

            if success:
                logger.debug(
                    "event_metadata_written",
                    event_id=event_id,
                    document_id=document_id,
                    job_id=str(job_id),
                )

            return success
        except Exception as e:
            logger.error(f"Failed to write event metadata {event_id}: {e}")
            return False

    async def write_entity_metadata(
        self,
        entity_id: str,
        document_id: str,
        job_id: UUID,
        batch_id: Optional[str],
        entity_data: Dict[str, Any],
    ) -> bool:
        """
        Write entity metadata to registry.

        Args:
            entity_id: Entity identifier
            document_id: Parent document ID
            job_id: Processing job ID
            batch_id: Batch identifier
            entity_data: Full entity data from NER

        Returns:
            True if write successful
        """
        if not self._is_ready():
            return False

        try:
            entity = EntityMetadata(
                entity_id=entity_id,
                document_id=document_id,
                job_id=job_id,
                batch_id=batch_id,
                entity_type=entity_data.get("type"),
                name=entity_data.get("name", entity_data.get("text", "Unknown")),
                mentions=entity_data.get("mentions", 1),
                confidence=entity_data.get("confidence"),
                full_data=entity_data,
            )

            success = await self.registry.store_entity(entity)

            if success:
                logger.debug(
                    "entity_metadata_written",
                    entity_id=entity_id,
                    document_id=document_id,
                    job_id=str(job_id),
                )

            return success
        except Exception as e:
            logger.error(f"Failed to write entity metadata {entity_id}: {e}")
            return False

    async def write_storyline_metadata(
        self,
        storyline_id: str,
        document_id: str,
        job_id: UUID,
        batch_id: Optional[str],
        storyline_data: Dict[str, Any],
    ) -> bool:
        """
        Write storyline metadata to registry.

        Args:
            storyline_id: Storyline identifier
            document_id: Parent document ID
            job_id: Processing job ID
            batch_id: Batch identifier
            storyline_data: Full storyline data

        Returns:
            True if write successful
        """
        if not self._is_ready():
            return False

        try:
            storyline = StorylineMetadata(
                storyline_id=storyline_id,
                document_id=document_id,
                job_id=job_id,
                batch_id=batch_id,
                title=storyline_data.get("title"),
                summary=storyline_data.get("summary"),
                key_entities=storyline_data.get("key_entities", []),
                event_count=storyline_data.get("event_count", 0),
                full_data=storyline_data,
            )

            success = await self.registry.store_storyline(storyline)

            if success:
                logger.debug(
                    "storyline_metadata_written",
                    storyline_id=storyline_id,
                    document_id=document_id,
                    job_id=str(job_id),
                )

            return success
        except Exception as e:
            logger.error(f"Failed to write storyline metadata {storyline_id}: {e}")
            return False

    async def write_batch(
        self,
        documents: List[DocumentMetadata],
        events: List[EventMetadata],
        entities: List[EntityMetadata],
    ) -> Dict[str, int]:
        """
        Write batch of metadata in bulk (optimized).

        Args:
            documents: List of document metadata
            events: List of event metadata
            entities: List of entity metadata

        Returns:
            Dictionary with counts of successfully written items
        """
        if not self._is_ready():
            return {"documents": 0, "events": 0, "entities": 0}

        counts = {"documents": 0, "events": 0, "entities": 0}

        try:
            if documents:
                counts["documents"] = await self.registry.bulk_store_documents(documents)

            if events:
                counts["events"] = await self.registry.bulk_store_events(events)

            if entities:
                counts["entities"] = await self.registry.bulk_store_entities(entities)

            logger.info(
                "batch_metadata_written",
                documents=counts["documents"],
                events=counts["events"],
                entities=counts["entities"],
            )

            return counts
        except Exception as e:
            logger.error(f"Failed to write batch metadata: {e}")
            return counts

    def _is_ready(self) -> bool:
        """Check if writer is ready to write."""
        return self.enabled and self.initialized and self.registry is not None


# Singleton instance (optional, can be instantiated per-task)
_writer_instance: Optional[MetadataWriter] = None


def get_metadata_writer() -> MetadataWriter:
    """
    Get or create singleton metadata writer instance.

    Returns:
        MetadataWriter instance
    """
    global _writer_instance

    if _writer_instance is None:
        _writer_instance = MetadataWriter()

    return _writer_instance
