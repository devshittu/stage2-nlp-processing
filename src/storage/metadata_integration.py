"""
Metadata registry integration hooks for Stage 2.

Provides async wrapper functions to write metadata to shared registry
alongside existing storage backends (dual-write pattern).

ZERO-REGRESSION GUARANTEE:
- All writes wrapped in try-catch
- Failures logged but don't affect main pipeline
- Completely optional (can be disabled via config)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog

from src.schemas.data_models import (
    ProcessedDocument,
    Event,
    Entity,
    Storyline,
    EventLinkage
)

# Import metadata writer (graceful if not available)
try:
    from src.storage.metadata_writer import get_metadata_writer

    WRITER_AVAILABLE = True
except ImportError:
    WRITER_AVAILABLE = False
    structlog.get_logger(__name__).warning("Metadata writer not available")

logger = structlog.get_logger(__name__)


async def write_job_to_registry(
    job_id: UUID,
    batch_id: Optional[str] = None,
    parent_job_id: Optional[UUID] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Write job registration to metadata registry.

    Args:
        job_id: Stage 2 job ID
        batch_id: Batch identifier
        parent_job_id: Optional parent job from Stage 1
        metadata: Additional job metadata

    Returns:
        True if successful (or disabled), False if error occurred
    """
    if not WRITER_AVAILABLE:
        return True  # Not an error, just not available

    try:
        writer = get_metadata_writer()
        await writer.connect()
        success = await writer.register_job(job_id, batch_id, parent_job_id, metadata)
        return success
    except Exception as e:
        logger.error(f"Failed to write job to metadata registry: {e}")
        return False


async def write_documents_to_registry(
    processed_docs: List[ProcessedDocument],
) -> int:
    """
    Write processed documents to metadata registry.

    Args:
        processed_docs: List of processed documents

    Returns:
        Number of documents successfully written
    """
    if not WRITER_AVAILABLE or not processed_docs:
        return 0

    count = 0

    try:
        writer = get_metadata_writer()

        for doc in processed_docs:
            try:
                # Write document metadata
                success = await writer.write_document_metadata(
                    document_id=doc.document_id,
                    job_id=doc.job_id,
                    batch_id=None,  # Extract from doc if available
                    source_document=doc.source_document,
                )

                if success:
                    count += 1

                    # Write events
                    for event in doc.events:
                        await writer.write_event_metadata(
                            event_id=event.event_id,
                            document_id=doc.document_id,
                            job_id=doc.job_id,
                            batch_id=None,
                            event_data=event.dict(),
                        )

                    # Write entities
                    for entity in doc.extracted_entities:
                        # Generate entity_id if not present
                        entity_id = getattr(entity, "entity_id", None)
                        if not entity_id:
                            entity_id = f"{doc.document_id}_entity_{entity.name}_{entity.type}"

                        await writer.write_entity_metadata(
                            entity_id=entity_id,
                            document_id=doc.document_id,
                            job_id=doc.job_id,
                            batch_id=None,
                            entity_data=entity.dict(),
                        )

                    # Write storylines (if available)
                    for storyline in doc.storylines:
                        storyline_id = getattr(storyline, "storyline_id", None)
                        if storyline_id:
                            await writer.write_storyline_metadata(
                                storyline_id=storyline_id,
                                document_id=doc.document_id,
                                job_id=doc.job_id,
                                batch_id=None,
                                storyline_data=storyline.dict(),
                            )

            except Exception as e:
                logger.error(
                    f"Failed to write document {doc.document_id} to metadata registry: {e}"
                )
                # Continue with next document (don't fail entire batch)

        logger.info(
            f"Metadata registry: wrote {count}/{len(processed_docs)} documents"
        )

        return count

    except Exception as e:
        logger.error(f"Failed to write documents to metadata registry: {e}")
        return 0


def sync_write_job_to_registry(
    job_id: UUID,
    batch_id: Optional[str] = None,
    parent_job_id: Optional[UUID] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Synchronous wrapper for write_job_to_registry.

    For use in synchronous Celery tasks.

    Args:
        job_id: Stage 2 job ID
        batch_id: Batch identifier
        parent_job_id: Optional parent job from Stage 1
        metadata: Additional job metadata

    Returns:
        True if successful (or disabled), False if error occurred
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(
            write_job_to_registry(job_id, batch_id, parent_job_id, metadata)
        )
    except Exception as e:
        logger.error(f"Sync write job failed: {e}")
        return False


def sync_write_documents_to_registry(processed_docs: List[ProcessedDocument]) -> int:
    """
    Synchronous wrapper for write_documents_to_registry.

    For use in synchronous Celery tasks.

    Args:
        processed_docs: List of processed documents

    Returns:
        Number of documents successfully written
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(write_documents_to_registry(processed_docs))
    except Exception as e:
        logger.error(f"Sync write documents failed: {e}")
        return 0


def update_job_status_in_registry(
    job_id: UUID,
    status: str,
    error_message: Optional[str] = None,
) -> bool:
    """
    Update job status in metadata registry.

    Args:
        job_id: Job identifier
        status: New status (completed, failed, etc.)
        error_message: Optional error message

    Returns:
        True if successful (or disabled)
    """
    if not WRITER_AVAILABLE:
        return True

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        async def _update():
            writer = get_metadata_writer()
            await writer.connect()
            return await writer.update_job_status(job_id, status, error_message)

        return loop.run_until_complete(_update())
    except Exception as e:
        logger.error(f"Failed to update job status in registry: {e}")
        return False
