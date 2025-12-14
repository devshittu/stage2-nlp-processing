"""
celery_tasks.py

Celery tasks with Dask integration for parallel batch processing.
Coordinates the full NLP pipeline: NER → DP → Event LLM → Event Linking → Storage.

Features:
- Celery app initialization with Redis broker
- Dask LocalCluster for parallel document processing
- Process single documents synchronously
- Process batches with Dask parallelism
- Event linking across batch for storyline identification
- Multi-backend storage with error handling
- Comprehensive progress tracking and metrics
- Graceful error handling per document (batch continues on failures)

Architecture:
- process_document_task: Single document processing (synchronous)
- process_batch_task: Batch processing with Dask parallelism
  - Creates Dask cluster from config
  - Processes docs in parallel
  - Links events across batch
  - Saves to storage backends
  - Returns summary statistics
"""

import os
import json
import uuid
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed as futures_as_completed

# Celery
from celery import Celery, Task
from celery.utils.log import get_task_logger

# Dask
from dask.distributed import LocalCluster, Client, as_completed
from dask import delayed
import dask

# Internal imports
from src.utils.config_manager import get_settings
from src.utils.logger import (
    get_logger,
    PerformanceLogger,
    log_exception,
    log_info_with_metrics,
    initialize_logging_from_config
)
from src.utils.document_processor import DocumentProcessor
from src.schemas.data_models import (
    Stage1Document,
    ProcessedDocument,
    Entity,
    SOATriplet,
    Event,
    EventLinkage,
    Storyline
)
from src.core.event_linker import get_event_linker
from src.storage.backends import MultiBackendWriter


# =============================================================================
# Module Initialization
# =============================================================================

# Initialize logging
initialize_logging_from_config()
logger = get_logger(__name__, service="celery_worker")

# Load settings
settings = get_settings()
celery_config = settings.celery

# Initialize event publisher for inter-stage communication
event_publisher = None
try:
    from src.events.publisher import create_event_publisher
    event_publisher = create_event_publisher(settings)
    if settings.events.enabled:
        logger.info(f"Event publisher initialized with backend: {settings.events.backend}")
    else:
        logger.info("Event publisher disabled (events.enabled = false)")
except Exception as e:
    logger.error(f"Failed to initialize event publisher: {e}", exc_info=True)
    event_publisher = None

# =============================================================================
# Processing Constants
# =============================================================================

# Worker configuration
MAX_WORKERS_FALLBACK = min(os.cpu_count() or 4, 8)  # Use up to 8 cores for fallback processing

# Progress reporting
BATCH_CHUNK_SIZE = 10  # Log progress every N documents
PROGRESS_LOG_INTERVAL = 10  # Log progress every 10 documents

# Event linking thresholds
MIN_EVENTS_FOR_LINKING = 2  # Minimum events needed to perform linking

# Dask configuration
DASK_DASHBOARD_PORT = 8787  # Dask dashboard port (if enabled)

# Timeouts (in seconds)
HTTP_CLIENT_TIMEOUT = 300  # 5 minutes for downstream service calls
HTTP_CONNECT_TIMEOUT = 10  # 10 seconds to establish connection

# =============================================================================
# Event Linker Cache (only non-HTTP service)
# =============================================================================

class EventLinkerCache:
    """
    Cache for event linker (used for cross-document event linking).
    NER, DP, and Event LLM are accessed via HTTP, not loaded in-process.
    """
    _instance = None
    _event_linker = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventLinkerCache, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        """Initialize event linker."""
        if self._initialized:
            logger.debug("Event linker already initialized")
            return

        logger.info("Initializing event linker for batch processing...")

        try:
            self._event_linker = get_event_linker()
            logger.info("✓ Event linker loaded")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize event linker: {e}", exc_info=True)
            raise

    @property
    def event_linker(self):
        """Get event linker instance."""
        if not self._initialized:
            self.initialize()
        return self._event_linker


# Global event linker cache
linker_cache = EventLinkerCache()


# =============================================================================
# Celery App Configuration
# =============================================================================

app = Celery(
    "nlp_processing_tasks",
    broker=celery_config.broker_url,
    backend=celery_config.result_backend
)

# Configure Celery from settings
app.conf.update(
    task_serializer=celery_config.task_serializer,
    result_serializer=celery_config.result_serializer,
    accept_content=celery_config.accept_content,
    task_routes=celery_config.task_routes,
    task_time_limit=celery_config.task_time_limit,
    task_soft_time_limit=celery_config.task_soft_time_limit,
    result_expires=celery_config.result_expires,
    worker_prefetch_multiplier=1,  # Disable prefetching for better memory control
    task_acks_late=True,  # Acknowledge after task completion
    task_reject_on_worker_lost=True,  # Reject tasks if worker lost
)

logger.info(
    "Celery app configured",
    extra={
        "broker": celery_config.broker_url.split('@')[-1],  # Hide credentials
        "backend": celery_config.result_backend.split('@')[-1],
        "task_time_limit": celery_config.task_time_limit,
        "dask_enabled": celery_config.dask_enabled
    }
)


# =============================================================================
# Celery Worker Signals
# =============================================================================

from celery.signals import worker_ready, worker_shutdown


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """
    Initialize event linker when worker starts.
    NER, DP, and Event LLM are accessed via HTTP (no in-process loading).
    """
    logger.info("Worker ready signal received. Initializing event linker...")
    try:
        linker_cache.initialize()
        logger.info("Worker ready with event linker loaded (HTTP mode for NER/DP/LLM)")
    except Exception as e:
        logger.error(f"Failed to initialize event linker on worker startup: {e}", exc_info=True)
        # Don't raise - allow worker to start, linker will lazy-load


@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs):
    """Cleanup on worker shutdown."""
    logger.info("Worker shutting down...")


# =============================================================================
# Helper Functions
# =============================================================================

def create_dask_cluster() -> Tuple[LocalCluster, Client]:
    """
    Create Dask LocalCluster with configured settings.

    Returns:
        Tuple of (cluster, client)

    Raises:
        RuntimeError: If cluster creation fails
    """
    logger.info("Creating Dask LocalCluster")

    try:
        cluster = LocalCluster(
            n_workers=celery_config.dask_local_cluster_n_workers,
            threads_per_worker=celery_config.dask_local_cluster_threads_per_worker,
            memory_limit=celery_config.dask_local_cluster_memory_limit,
            processes=False,  # Use threads to avoid daemon process error in Celery workers
            silence_logs=logging.ERROR,  # Reduce Dask logging noise
            dashboard_address=None,  # Disable dashboard in production
        )

        client = Client(cluster)

        logger.info(
            "Dask cluster created successfully",
            extra={
                "n_workers": celery_config.dask_local_cluster_n_workers,
                "threads_per_worker": celery_config.dask_local_cluster_threads_per_worker,
                "memory_limit": celery_config.dask_local_cluster_memory_limit,
                "dashboard_link": client.dashboard_link
            }
        )

        return cluster, client

    except Exception as e:
        logger.error(f"Failed to create Dask cluster: {e}", exc_info=True)
        raise RuntimeError(f"Dask cluster creation failed: {e}")


def process_single_document_pipeline(
    document_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single document through the full NLP pipeline using HTTP requests.

    This function calls the microservices (NER, DP, Event LLM) via HTTP for scalability.
    This architecture ensures vLLM is used for fast event extraction.

    Features:
    - HTTP retries with exponential backoff
    - Comprehensive error logging for traceability
    - Handles long documents via chunking in Event LLM service

    Args:
        document_dict: Document as dictionary (Stage1Document serialized)

    Returns:
        Dictionary with processing results or error information

    Note:
        This function handles errors gracefully and returns error info
        rather than raising exceptions (to prevent batch failure).
    """
    import httpx
    from httpx import Timeout, Limits

    document_id = document_dict.get("document_id", "unknown")
    start_time = datetime.utcnow()

    # Service endpoints
    NER_SERVICE_URL = "http://ner-service:8001/api/v1/extract"
    DP_SERVICE_URL = "http://dp-service:8002/api/v1/parse"
    EVENT_LLM_SERVICE_URL = "http://event-llm-service:8003/api/v1/extract"

    # HTTP client configuration with retries
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    def make_request_with_retry(client, url, data, service_name):
        """Make HTTP request with retry logic."""
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = client.post(url, json=data)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"[{document_id}] {service_name} request failed (attempt {attempt+1}/{MAX_RETRIES}), "
                        f"retrying in {wait_time}s: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"[{document_id}] {service_name} request failed after {MAX_RETRIES} attempts: {str(e)}")

        raise last_error

    try:
        # Parse document
        document = Stage1Document(**document_dict)

        # Extract text
        doc_processor = DocumentProcessor()
        text = doc_processor.extract_text(document)
        context = doc_processor.extract_context_fields(document)

        logger.info(
            f"Processing document via HTTP: {document_id}",
            extra={
                "document_id": document_id,
                "text_length": len(text),
                "context_fields": list(context.keys())
            }
        )

        # Step 1: NER via HTTP with retries
        logger.debug(f"[{document_id}] Calling NER service")
        with httpx.Client(timeout=HTTP_CLIENT_TIMEOUT) as client:
            ner_data = make_request_with_retry(
                client,
                NER_SERVICE_URL,
                {"text": text, "document_id": document_id},
                "NER"
            )
            entities = [Entity(**e) for e in ner_data.get("entities", [])]
            logger.debug(f"[{document_id}] NER extracted {len(entities)} entities")

        # Step 2: Dependency Parsing via HTTP with retries
        logger.debug(f"[{document_id}] Calling DP service")
        with httpx.Client(timeout=HTTP_CLIENT_TIMEOUT) as client:
            dp_data = make_request_with_retry(
                client,
                DP_SERVICE_URL,
                {"text": text, "document_id": document_id},
                "DP"
            )
            soa_triplets = [SOATriplet(**t) for t in dp_data.get("soa_triplets", [])]
            logger.debug(f"[{document_id}] DP extracted {len(soa_triplets)} triplets")

        # Step 3: Event LLM Extraction via HTTP (with vLLM!) with retries
        logger.debug(f"[{document_id}] Calling Event LLM service (vLLM)")
        with httpx.Client(timeout=HTTP_CLIENT_TIMEOUT) as client:
            event_data = make_request_with_retry(
                client,
                EVENT_LLM_SERVICE_URL,
                {
                    "text": text,
                    "document_id": document_id,
                    "entities": [e.model_dump() for e in entities],
                    "context": context
                },
                "Event LLM"
            )
            events = [Event(**e) for e in event_data.get("events", [])]
            logger.debug(f"[{document_id}] Event LLM extracted {len(events)} events")

        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000

        # Build result
        result = {
            "success": True,
            "document_id": document_id,
            "entities": [e.model_dump() for e in entities],
            "soa_triplets": [t.model_dump() for t in soa_triplets],
            "events": [e.model_dump() for e in events],
            "source_document": document_dict,
            "processing_time_ms": processing_time_ms,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "error": None
        }

        logger.info(
            f"Document processed successfully via HTTP: {document_id}",
            extra={
                "document_id": document_id,
                "entities_count": len(entities),
                "events_count": len(events),
                "processing_time_ms": processing_time_ms
            }
        )

        return result

    except httpx.HTTPError as e:
        error_msg = f"HTTP error processing document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)

        return {
            "success": False,
            "document_id": document_id,
            "entities": [],
            "soa_triplets": [],
            "events": [],
            "source_document": document_dict,
            "processing_time_ms": 0,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "error": error_msg,
            "error_traceback": traceback.format_exc()
        }
    except Exception as e:
        error_msg = f"Failed to process document {document_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)

        return {
            "success": False,
            "document_id": document_id,
            "entities": [],
            "soa_triplets": [],
            "events": [],
            "source_document": document_dict,
            "processing_time_ms": 0,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "error": error_msg,
            "error_traceback": traceback.format_exc()
        }


def link_events_across_batch(
    processed_results: List[Dict[str, Any]],
    batch_id: str,
    event_linker: Any  # EventLinker type from event_linker module
) -> Tuple[List[Dict[str, Any]], List[EventLinkage], List[Storyline]]:
    """
    Link events across batch and assign storyline IDs.

    Args:
        processed_results: List of processing results from pipeline
        batch_id: Batch identifier
        event_linker: EventLinker instance

    Returns:
        Tuple of (updated_results, linkages, storylines)
    """
    logger.info(f"Linking events across batch: {batch_id}")

    try:
        # Collect all events from successful documents
        all_events = []
        document_event_map = defaultdict(list)  # Maps doc_id -> event indices

        for result in processed_results:
            if result["success"] and result["events"]:
                doc_id = result["document_id"]
                for event_dict in result["events"]:
                    event = Event(**event_dict)
                    event_idx = len(all_events)
                    all_events.append(event)
                    document_event_map[doc_id].append(event_idx)

        if len(all_events) < MIN_EVENTS_FOR_LINKING:
            logger.info(
                f"Less than {MIN_EVENTS_FOR_LINKING} events in batch, skipping event linking",
                extra={"event_count": len(all_events)}
            )
            return processed_results, [], []

        logger.info(f"Linking {len(all_events)} events from {len(document_event_map)} documents")

        # Perform event linking
        linkages, storylines = event_linker.link_events(all_events, batch_id=batch_id)

        # Update events in results with storyline IDs
        for result in processed_results:
            if result["success"]:
                doc_id = result["document_id"]
                if doc_id in document_event_map:
                    # Update events with storyline assignments
                    for event_dict in result["events"]:
                        event_id = event_dict["event_id"]

                        # Find corresponding Event object
                        matching_event = next(
                            (e for e in all_events if e.event_id == event_id),
                            None
                        )

                        if matching_event and matching_event.storyline_id:
                            event_dict["storyline_id"] = matching_event.storyline_id

                            # Add linked event IDs
                            if matching_event.linked_event_ids:
                                event_dict["linked_event_ids"] = matching_event.linked_event_ids

        logger.info(
            f"Event linking complete: {len(linkages)} linkages, {len(storylines)} storylines",
            extra={
                "batch_id": batch_id,
                "linkages_count": len(linkages),
                "storylines_count": len(storylines)
            }
        )

        return processed_results, linkages, storylines

    except Exception as e:
        logger.error(f"Event linking failed for batch {batch_id}: {e}", exc_info=True)
        # Return original results without linking
        return processed_results, [], []


def save_processed_documents(
    processed_results: List[Dict[str, Any]],
    linkages: List[EventLinkage],
    storylines: List[Storyline],
    storage_writer: MultiBackendWriter
) -> Dict[str, int]:
    """
    Save processed documents to storage backends.

    Args:
        processed_results: List of processing results
        linkages: Event linkages
        storylines: Storylines
        storage_writer: Storage backend writer

    Returns:
        Dictionary with save statistics
    """
    logger.info(f"Saving {len(processed_results)} processed documents (including failures for traceability)")

    # Convert results to ProcessedDocument objects
    processed_docs = []

    for result in processed_results:
        # CRITICAL: Save ALL documents (success AND failures) for traceability
        try:
            # Build ProcessedDocument
            doc = ProcessedDocument(
                document_id=result["document_id"],
                job_id=result.get("job_id"),
                processed_at=result["processed_at"],
                normalized_date=result["source_document"].get("cleaned_publication_date"),
                original_text=result["source_document"].get("cleaned_text", ""),
                source_document=result["source_document"],
                extracted_entities=[Entity(**e) for e in result["entities"]],
                extracted_soa_triplets=[SOATriplet(**t) for t in result["soa_triplets"]],
                events=[Event(**e) for e in result["events"]],
                event_linkages=linkages,
                storylines=storylines,
                processing_metadata={
                    "processing_time_ms": result["processing_time_ms"],
                    "entities_count": len(result["entities"]),
                    "events_count": len(result["events"]),
                    "soa_triplets_count": len(result["soa_triplets"])
                }
            )

            processed_docs.append(doc)

        except Exception as e:
            logger.error(
                f"Failed to convert result to ProcessedDocument: {result['document_id']}",
                exc_info=True
            )

    # Save batch
    if processed_docs:
        save_results = storage_writer.save_batch(processed_docs)

        logger.info(
            f"Saved {len(processed_docs)} documents to storage",
            extra={"save_results": save_results}
        )

        return {
            "documents_saved": len(processed_docs),
            "backend_results": save_results
        }
    else:
        logger.warning("No documents to save")
        return {"documents_saved": 0, "backend_results": {}}


# =============================================================================
# Celery Tasks
# =============================================================================

@app.task(name="process_document_task", bind=True)
def process_document_task(self, document_json: str) -> Dict[str, Any]:
    """
    Process a single document through the NLP pipeline.

    Args:
        document_json: JSON string of Stage1Document

    Returns:
        Dictionary with ProcessedDocument data or error info

    Example:
        result = process_document_task.delay(json.dumps(document.model_dump()))
        processed = result.get(timeout=60)
    """
    task_id = self.request.id
    logger.info(f"Starting document processing task: {task_id}")

    start_time = datetime.utcnow()

    try:
        # Parse document with error handling
        try:
            document_dict = json.loads(document_json)
        except (json.JSONDecodeError, TypeError) as e:
            error_msg = f"Invalid JSON document: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "task_id": task_id,
                "error": error_msg,
                "error_traceback": traceback.format_exc()
            }

        document_id = document_dict.get("document_id", "unknown")

        logger.info(
            f"Processing single document: {document_id}",
            extra={"task_id": task_id, "document_id": document_id}
        )

        # Use HTTP requests to microservices (no model loading needed)
        storage_writer = MultiBackendWriter()

        # Process document via HTTP
        result = process_single_document_pipeline(
            document_dict=document_dict
        )

        if not result["success"]:
            logger.error(
                f"Document processing failed: {document_id}",
                extra={"error": result["error"]}
            )
            return result

        # Add job ID
        result["job_id"] = task_id

        # Event linking (single document - no linking needed, but process for consistency)
        events = [Event(**e) for e in result["events"]] if result["events"] else []
        linkages = []
        storylines = []

        # Build ProcessedDocument
        processed_doc = ProcessedDocument(
            document_id=result["document_id"],
            job_id=task_id,
            processed_at=result["processed_at"],
            normalized_date=document_dict.get("cleaned_publication_date"),
            original_text=document_dict.get("cleaned_text", ""),
            source_document=document_dict,
            extracted_entities=[Entity(**e) for e in result["entities"]],
            extracted_soa_triplets=[SOATriplet(**t) for t in result["soa_triplets"]],
            events=events,
            event_linkages=linkages,
            storylines=storylines,
            processing_metadata={
                "processing_time_ms": result["processing_time_ms"],
                "entities_count": len(result["entities"]),
                "events_count": len(result["events"]),
                "soa_triplets_count": len(result["soa_triplets"])
            }
        )

        # Save to storage
        save_results = storage_writer.save(processed_doc)

        # Calculate total time
        end_time = datetime.utcnow()
        total_time_ms = (end_time - start_time).total_seconds() * 1000

        logger.info(
            f"Document processing task complete: {document_id}",
            extra={
                "task_id": task_id,
                "document_id": document_id,
                "total_time_ms": total_time_ms,
                "save_results": save_results
            }
        )

        # Return processed document as dict
        return {
            "success": True,
            "task_id": task_id,
            "document": processed_doc.model_dump(),
            "total_time_ms": total_time_ms,
            "save_results": save_results
        }

    except Exception as e:
        error_msg = f"Task failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        return {
            "success": False,
            "task_id": task_id,
            "error": error_msg,
            "error_traceback": traceback.format_exc()
        }


@app.task(name="process_batch_task", bind=True)
def process_batch_task(
    self,
    documents: List[Dict[str, Any]],
    batch_id: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a batch of documents with Dask parallelism.

    Pipeline:
    1. Validate input documents
    2. Create Dask LocalCluster
    3. Process documents in parallel (NER → DP → LLM)
    4. Link events across batch
    5. Assign storyline IDs
    6. Save to storage backends
    7. Return summary statistics

    Args:
        documents: List of Stage1Document dictionaries
        batch_id: Optional batch identifier (auto-generated if not provided)
        options: Optional processing options

    Returns:
        Dictionary with batch processing results and statistics

    Example:
        documents = [doc1.model_dump(), doc2.model_dump(), ...]
        result = process_batch_task.delay(documents=documents, batch_id="batch_001")
        batch_result = result.get(timeout=3600)
    """
    task_id = self.request.id
    batch_id = batch_id or f"batch_{task_id}"

    logger.info(
        f"Starting batch processing task: {task_id}",
        extra={"task_id": task_id, "batch_id": batch_id}
    )

    start_time = datetime.utcnow()
    cluster = None
    client = None

    try:
        # Validate documents input
        if not isinstance(documents, list):
            raise ValueError(f"Expected list of documents, got {type(documents)}")

        documents_list = documents
        total_docs = len(documents_list)

        logger.info(
            f"Processing batch: {batch_id} with {total_docs} documents",
            extra={
                "task_id": task_id,
                "batch_id": batch_id,
                "document_count": total_docs
            }
        )

        if total_docs == 0:
            return {
                "success": True,
                "batch_id": batch_id,
                "task_id": task_id,
                "documents_total": 0,
                "documents_processed": 0,
                "documents_failed": 0,
                "message": "Empty batch"
            }

        # Publish batch.started event
        if event_publisher and settings.events.enabled and settings.events.publish_events.batch_started:
            try:
                event_publisher.publish_batch_started(
                    job_id=task_id,
                    total_documents=total_docs,
                    metadata={
                        "batch_id": batch_id,
                        "started_at": start_time.isoformat() + "Z"
                    }
                )
                logger.info(f"Published batch.started event for job {task_id}")
            except Exception as e:
                logger.warning(f"Failed to publish batch.started event: {e}")

        # Use cached models (loaded once per worker)
        logger.info("Using cached NLP models")
        storage_writer = MultiBackendWriter()

        # Create Dask cluster if enabled
        if celery_config.dask_enabled:
            logger.info("Creating Dask cluster for parallel processing")
            cluster, client = create_dask_cluster()
        else:
            logger.info("Dask disabled, processing sequentially")

        # Process documents
        logger.info(f"Processing {total_docs} documents")

        if celery_config.dask_enabled and client:
            # Parallel processing with Dask
            logger.info("Submitting documents to Dask cluster")

            # Create delayed tasks (HTTP-based, no models needed)
            futures = []
            for doc_dict in documents_list:
                future = client.submit(
                    process_single_document_pipeline,
                    doc_dict
                )
                futures.append(future)

            # Collect results with progress tracking
            processed_results = []
            documents_processed = 0
            documents_failed = 0

            for future in as_completed(futures):
                try:
                    result = future.result()
                    processed_results.append(result)

                    if result["success"]:
                        documents_processed += 1
                    else:
                        documents_failed += 1

                    # Update task progress
                    progress = (len(processed_results) / total_docs) * 100
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'current': len(processed_results),
                            'total': total_docs,
                            'progress': progress,
                            'documents_processed': documents_processed,
                            'documents_failed': documents_failed
                        }
                    )

                    if len(processed_results) % PROGRESS_LOG_INTERVAL == 0:
                        logger.info(
                            f"Progress: {len(processed_results)}/{total_docs} documents",
                            extra={
                                "batch_id": batch_id,
                                "progress": progress,
                                "success": documents_processed,
                                "failed": documents_failed
                            }
                        )

                except Exception as e:
                    logger.error(f"Failed to process document in Dask: {e}", exc_info=True)
                    documents_failed += 1

        else:
            # Parallel processing with ThreadPoolExecutor (fallback when Dask disabled)
            # Note: Using threads instead of processes because GPU models cannot be pickled
            logger.info(
                f"Processing documents in parallel with ThreadPoolExecutor "
                f"({MAX_WORKERS_FALLBACK} workers)"
            )
            processed_results = []
            documents_processed = 0
            documents_failed = 0

            # Use ThreadPoolExecutor for parallel processing (HTTP-based)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS_FALLBACK) as executor:
                # Submit all tasks
                future_to_doc = {
                    executor.submit(process_single_document_pipeline, doc_dict): doc_dict
                    for doc_dict in documents_list
                }

                # Collect results as they complete
                for future in futures_as_completed(future_to_doc):
                    doc_dict = future_to_doc[future]
                    document_id = doc_dict.get("document_id", "unknown")

                    try:
                        result = future.result()
                        processed_results.append(result)

                        if result["success"]:
                            documents_processed += 1
                            logger.info(f"✓ Document {document_id} processed successfully")
                        else:
                            documents_failed += 1
                            logger.error(f"✗ Document {document_id} processing failed: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        # CRITICAL: Record failed document for traceability
                        error_msg = f"Exception processing document {document_id}: {str(e)}"
                        logger.error(error_msg, exc_info=True)

                        # Add failed result to ensure traceability
                        failed_result = {
                            "success": False,
                            "document_id": document_id,
                            "entities": [],
                            "soa_triplets": [],
                            "events": [],
                            "source_document": doc_dict,
                            "processing_time_ms": 0,
                            "processed_at": datetime.utcnow().isoformat() + "Z",
                            "error": error_msg,
                            "error_traceback": traceback.format_exc()
                        }
                        processed_results.append(failed_result)
                        documents_failed += 1

                    # Update progress
                    progress = (len(processed_results) / total_docs) * 100
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'current': len(processed_results),
                            'total': total_docs,
                            'progress': progress,
                            'documents_processed': documents_processed,
                            'documents_failed': documents_failed
                        }
                    )

                    # Log progress periodically
                    if len(processed_results) % PROGRESS_LOG_INTERVAL == 0:
                        logger.info(
                            f"Progress: {len(processed_results)}/{total_docs} documents",
                            extra={
                                "batch_id": batch_id,
                                "progress": progress,
                                "success": documents_processed,
                                "failed": documents_failed
                            }
                        )

        # Add job_id to all results
        for result in processed_results:
            result["job_id"] = task_id

        # Event linking across batch with cached linker
        logger.info(f"Linking events across batch: {batch_id}")
        processed_results, linkages, storylines = link_events_across_batch(
            processed_results,
            batch_id,
            linker_cache.event_linker
        )

        # Save to storage
        logger.info("Saving processed documents to storage")
        save_stats = save_processed_documents(
            processed_results,
            linkages,
            storylines,
            storage_writer
        )

        # Calculate statistics
        end_time = datetime.utcnow()
        total_time_ms = (end_time - start_time).total_seconds() * 1000

        # Count success/failure
        documents_processed = sum(1 for r in processed_results if r["success"])
        documents_failed = sum(1 for r in processed_results if not r["success"])

        # Event statistics
        total_events = sum(len(r["events"]) for r in processed_results if r["success"])
        total_entities = sum(len(r["entities"]) for r in processed_results if r["success"])

        result = {
            "success": True,
            "batch_id": batch_id,
            "task_id": task_id,
            "documents_total": total_docs,
            "documents_processed": documents_processed,
            "documents_failed": documents_failed,
            "total_events": total_events,
            "total_entities": total_entities,
            "linkages_count": len(linkages),
            "storylines_count": len(storylines),
            "processing_time_ms": total_time_ms,
            "completed_at": end_time.isoformat() + "Z",
            "save_statistics": save_stats,
            "message": f"Processed {documents_processed}/{total_docs} documents successfully"
        }

        log_info_with_metrics(
            logger.logger,
            f"Batch processing complete: {batch_id}",
            metrics={
                "documents_total": total_docs,
                "documents_processed": documents_processed,
                "documents_failed": documents_failed,
                "total_events": total_events,
                "linkages": len(linkages),
                "storylines": len(storylines),
                "processing_time_seconds": total_time_ms / 1000
            },
            batch_id=batch_id,
            task_id=task_id
        )

        # Publish batch.completed event
        if event_publisher and settings.events.enabled and settings.events.publish_events.batch_completed:
            try:
                # Determine output locations
                output_locations = {}
                if save_stats:
                    if save_stats.get("jsonl"):
                        output_locations["jsonl"] = f"file:///app/data/extracted_events_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
                    if save_stats.get("postgresql"):
                        output_locations["postgresql"] = f"postgresql://db/batch/{batch_id}"
                    if save_stats.get("elasticsearch"):
                        output_locations["elasticsearch"] = f"http://es:9200/batch/{batch_id}"

                event_publisher.publish_batch_completed(
                    job_id=task_id,
                    total_documents=total_docs,
                    successful=documents_processed,
                    failed=documents_failed,
                    duration_seconds=total_time_ms / 1000.0,
                    started_at=start_time,
                    output_locations=output_locations,
                    aggregate_metrics={
                        "total_events": total_events,
                        "total_entities": total_entities,
                        "linkages_count": len(linkages),
                        "storylines_count": len(storylines),
                        "avg_processing_time_ms": total_time_ms / total_docs if total_docs > 0 else 0
                    }
                )
                logger.info(f"Published batch.completed event for job {task_id}")
            except Exception as e:
                logger.warning(f"Failed to publish batch.completed event: {e}")

        return result

    except Exception as e:
        error_msg = f"Batch processing failed: {str(e)}"
        log_exception(
            logger.logger,
            error_msg,
            batch_id=batch_id,
            task_id=task_id
        )

        return {
            "success": False,
            "batch_id": batch_id,
            "task_id": task_id,
            "error": error_msg,
            "error_traceback": traceback.format_exc()
        }

    finally:
        # Clean up Dask cluster
        if client:
            try:
                logger.info("Closing Dask client")
                client.close()
            except Exception as e:
                logger.warning(f"Error closing Dask client: {e}")

        if cluster:
            try:
                logger.info("Closing Dask cluster")
                cluster.close()
            except Exception as e:
                logger.warning(f"Error closing Dask cluster: {e}")


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    """
    Test Celery tasks configuration.

    Note: Actual task execution requires running Celery worker:
        celery -A src.core.celery_tasks worker --loglevel=info
    """
    import sys

    print("Celery Tasks Module Test")
    print("=" * 60)

    print("\n1. Configuration:")
    print(f"   Broker: {celery_config.broker_url.split('@')[-1]}")
    print(f"   Backend: {celery_config.result_backend.split('@')[-1]}")
    print(f"   Task time limit: {celery_config.task_time_limit}s")
    print(f"   Dask enabled: {celery_config.dask_enabled}")

    if celery_config.dask_enabled:
        print(f"\n2. Dask Configuration:")
        print(f"   Workers: {celery_config.dask_local_cluster_n_workers}")
        print(f"   Threads per worker: {celery_config.dask_local_cluster_threads_per_worker}")
        print(f"   Memory per worker: {celery_config.dask_local_cluster_memory_limit}")
        print(f"   Total memory: {celery_config.dask_cluster_total_memory}")

    print(f"\n3. Registered Tasks:")
    for task_name in app.tasks:
        if not task_name.startswith("celery."):
            print(f"   - {task_name}")

    print("\n" + "=" * 60)
    print("To run tasks, start Celery worker:")
    print("  celery -A src.core.celery_tasks worker --loglevel=info")
    print("\nTo monitor with Flower:")
    print("  celery -A src.core.celery_tasks flower")
    print("=" * 60)
