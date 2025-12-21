"""
orchestrator_service.py

Main orchestrator FastAPI service for Stage 2 NLP Processing Pipeline.
Coordinates the entire NLP pipeline: NER → DP → Event LLM → Event Linking → Storage.

Features:
- Single document synchronous processing
- Batch processing via Celery workers
- Health monitoring of all services
- Retry logic with exponential backoff
- Comprehensive error handling
- Structured logging
- CORS middleware
- Stage 1 integration (input) and Stage 3 contract (output)

Architecture:
- Orchestrates HTTP calls to microservices (NER, DP, Event LLM)
- Coordinates event linking and storage
- Submits batch jobs to Celery for parallel processing
- Provides job status tracking via Redis

Pipeline Flow (Single Document):
1. Validate Stage1Document input
2. Extract text using DocumentProcessor
3. Call NER service → Extract entities
4. Call DP service → Extract SOA triplets
5. Call Event LLM service → Extract events
6. Link events using EventLinker
7. Build ProcessedDocument output
8. Save to storage backends
9. Return ProcessedDocument to client
"""

import time
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.utils.middleware import setup_middleware
from src.schemas.data_models import (
    Stage1Document,
    ProcessedDocument,
    Entity,
    SOATriplet,
    Event,
    EventLinkage,
    Storyline,
    NERServiceResponse,
    DPServiceResponse,
    EventLLMServiceResponse,
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    ProcessBatchRequest,
    ProcessBatchResponse,
    JobStatusResponse,
    HealthCheckResponse,
    ErrorResponse,
)
from src.utils.config_manager import get_settings
from src.utils.logger import get_logger, initialize_logging_from_config, PerformanceLogger
from src.utils.document_processor import get_document_processor
from src.core.event_linker import get_event_linker
from src.storage.backends import MultiBackendWriter
from src.core.resource_lifecycle_manager import get_resource_manager


# =============================================================================
# Logging Setup
# =============================================================================

config = get_settings()
initialize_logging_from_config(config)
logger = get_logger(__name__, service="orchestrator_service")


# =============================================================================
# Service Configuration
# =============================================================================

class ServiceConfig:
    """Configuration for downstream services."""

    def __init__(self):
        settings = get_settings()
        self.ner_url = f"http://ner-service:{settings.ner_service.port}"
        self.dp_url = f"http://dp-service:{settings.dp_service.port}"
        self.event_llm_url = f"http://event-llm-service:{settings.event_llm_service.port}"

        self.ner_timeout = settings.orchestrator_service.ner_service_timeout
        self.dp_timeout = settings.orchestrator_service.dp_service_timeout
        self.event_llm_timeout = settings.orchestrator_service.event_llm_service_timeout

        self.max_retries = settings.orchestrator_service.max_retries
        self.retry_backoff = settings.orchestrator_service.retry_backoff_seconds


service_config = ServiceConfig()


# =============================================================================
# Global Resources (Module-level for lifespan access)
# =============================================================================

# HTTP client for async service calls
http_client: Optional[httpx.AsyncClient] = None

# Document processor for field extraction
document_processor = get_document_processor()

# Event linker for event co-reference
event_linker = get_event_linker()

# Storage writer
storage_writer: Optional[MultiBackendWriter] = None

# Event publisher for inter-stage communication
event_publisher: Optional[Any] = None


# =============================================================================
# FastAPI Application Setup with Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown logic.
    """
    global http_client, storage_writer, event_publisher

    # Startup
    logger.info("Starting orchestrator service...")

    # Configure HTTP client with connection pooling and limits
    http_pool_limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0
    )

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        limits=http_pool_limits,
        http2=False,  # Disabled - requires h2 package
        follow_redirects=True
    )
    logger.info("HTTP client initialized with connection pooling (max_connections=100, max_keepalive=20)")

    # Initialize storage writer
    try:
        storage_writer = MultiBackendWriter()
        logger.info("Storage writer initialized")
    except Exception as e:
        logger.error(f"Failed to initialize storage writer: {e}", exc_info=True)
        storage_writer = None

    # Initialize event publisher for inter-stage communication
    try:
        from src.events.publisher import create_event_publisher
        event_publisher = create_event_publisher(config)
        if config.events.enabled:
            logger.info(f"Event publisher initialized with backend: {config.events.backend}")
        else:
            logger.info("Event publisher disabled (events.enabled = false)")
    except Exception as e:
        logger.error(f"Failed to initialize event publisher: {e}", exc_info=True)
        event_publisher = None

    logger.info("Orchestrator service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down orchestrator service...")

    if http_client:
        await http_client.aclose()
        logger.info("HTTP client closed")

    if storage_writer:
        storage_writer.close()
        logger.info("Storage writer closed")

    if event_publisher:
        event_publisher.close()
        logger.info("Event publisher closed")

    logger.info("Orchestrator service shutdown complete")


# =============================================================================
# API Version Configuration
# =============================================================================

API_VERSION = "v1"
API_VERSION_PREFIX = f"/api/{API_VERSION}"

app = FastAPI(
    title="NLP Processing Orchestrator",
    description="Stage 2 NLP Processing Service - Orchestrates event and entity extraction pipeline",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=f"{API_VERSION_PREFIX}/docs",
    redoc_url=f"{API_VERSION_PREFIX}/redoc",
    openapi_url=f"{API_VERSION_PREFIX}/openapi.json",
)

# Setup secure middleware (CORS, logging, error handlers)
setup_middleware(app, "orchestrator_service", allow_cors_credentials=False)

# Create versioned router
api_v1_router = APIRouter(prefix=API_VERSION_PREFIX, tags=["v1"])


# =============================================================================
# Service Call Utilities
# =============================================================================

async def call_service_with_retry(
    url: str,
    payload: Dict[str, Any],
    service_name: str,
    timeout: float,
    max_retries: int = 3,
    backoff_seconds: float = 2.0
) -> Dict[str, Any]:
    """
    Call a microservice with retry logic.

    Args:
        url: Service endpoint URL
        payload: Request payload
        service_name: Service name for logging
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        backoff_seconds: Backoff multiplier between retries

    Returns:
        Response JSON

    Raises:
        HTTPException: If all retries fail
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            logger.debug(
                f"Calling {service_name} (attempt {attempt + 1}/{max_retries})",
                extra={"url": url, "attempt": attempt + 1}
            )

            response = await http_client.post(
                url,
                json=payload,
                timeout=timeout
            )

            response.raise_for_status()

            logger.debug(
                f"Successfully called {service_name}",
                extra={"url": url, "status_code": response.status_code}
            )

            return response.json()

        except httpx.TimeoutException as e:
            last_error = f"Timeout calling {service_name}: {e}"
            logger.warning(last_error, extra={"attempt": attempt + 1})

        except httpx.HTTPStatusError as e:
            last_error = f"HTTP error from {service_name}: {e.response.status_code}"
            logger.warning(last_error, extra={"attempt": attempt + 1})

        except Exception as e:
            last_error = f"Error calling {service_name}: {str(e)}"
            logger.warning(last_error, extra={"attempt": attempt + 1})

        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = backoff_seconds * (2 ** attempt)
            logger.debug(f"Waiting {wait_time}s before retry")
            await asyncio.sleep(wait_time)

    # All retries failed
    logger.error(f"All retries failed for {service_name}: {last_error}")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Failed to call {service_name} after {max_retries} attempts: {last_error}"
    )


async def call_ner_service(text: str, document_id: str) -> List[Entity]:
    """
    Call NER service to extract entities.

    Args:
        text: Text to process
        document_id: Document identifier

    Returns:
        List of extracted entities
    """
    payload = {
        "text": text,
        "document_id": document_id
    }

    response = await call_service_with_retry(
        url=f"{service_config.ner_url}/api/v1/extract",
        payload=payload,
        service_name="NER Service",
        timeout=service_config.ner_timeout,
        max_retries=service_config.max_retries,
        backoff_seconds=service_config.retry_backoff
    )

    # Parse response
    ner_response = NERServiceResponse(**response)
    return ner_response.entities


async def call_dp_service(text: str, document_id: str) -> List[SOATriplet]:
    """
    Call Dependency Parsing service to extract SOA triplets.

    Args:
        text: Text to process
        document_id: Document identifier

    Returns:
        List of SOA triplets
    """
    payload = {
        "text": text,
        "document_id": document_id
    }

    response = await call_service_with_retry(
        url=f"{service_config.dp_url}/api/v1/parse",
        payload=payload,
        service_name="DP Service",
        timeout=service_config.dp_timeout,
        max_retries=service_config.max_retries,
        backoff_seconds=service_config.retry_backoff
    )

    # Parse response
    dp_response = DPServiceResponse(**response)
    return dp_response.soa_triplets


async def call_event_llm_service(
    text: str,
    document_id: str,
    entities: List[Entity],
    context: Dict[str, Any]
) -> List[Event]:
    """
    Call Event LLM service to extract events.

    Args:
        text: Text to process
        document_id: Document identifier
        entities: Pre-extracted entities for context
        context: Additional context fields

    Returns:
        List of extracted events
    """
    payload = {
        "text": text,
        "document_id": document_id,
        "entities": [e.model_dump() for e in entities],
        "context": context
    }

    response = await call_service_with_retry(
        url=f"{service_config.event_llm_url}/api/v1/extract",
        payload=payload,
        service_name="Event LLM Service",
        timeout=service_config.event_llm_timeout,
        max_retries=service_config.max_retries,
        backoff_seconds=service_config.retry_backoff
    )

    # Parse response
    llm_response = EventLLMServiceResponse(**response)
    return llm_response.events


# =============================================================================
# Pipeline Processing
# =============================================================================

async def process_single_document(document: Stage1Document) -> ProcessedDocument:
    """
    Process a single document through the complete NLP pipeline.

    Pipeline stages:
    1. Document validation and field extraction
    2. NER: Entity extraction
    3. DP: Dependency parsing and SOA triplet extraction
    4. Event LLM: Event extraction with arguments and metadata
    5. Event Linking: Co-reference resolution (single doc = no linking)
    6. Storage: Save to configured backends

    Args:
        document: Stage 1 document

    Returns:
        ProcessedDocument matching Stage 3 contract

    Raises:
        HTTPException: If processing fails
    """
    document_id = document.document_id
    start_time = time.time()

    logger.info(f"Starting pipeline for document: {document_id}")

    try:
        # Step 1: Validate and extract fields
        is_valid, error_msg = document_processor.validate_document(document)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid document: {error_msg}"
            )

        text, context_fields, preserved_fields = document_processor.process_document(document)

        logger.info(
            f"Document validated and processed",
            extra={
                "document_id": document_id,
                "text_length": len(text),
                "context_fields": list(context_fields.keys())
            }
        )

        # Step 2: Call NER service
        with PerformanceLogger("ner_extraction", logger.logger, document_id=document_id):
            entities = await call_ner_service(text, document_id)

        logger.info(
            f"Extracted {len(entities)} entities",
            extra={"document_id": document_id, "entity_count": len(entities)}
        )

        # Step 3: Call DP service
        with PerformanceLogger("dp_extraction", logger.logger, document_id=document_id):
            soa_triplets = await call_dp_service(text, document_id)

        logger.info(
            f"Extracted {len(soa_triplets)} SOA triplets",
            extra={"document_id": document_id, "triplet_count": len(soa_triplets)}
        )

        # Step 4: Call Event LLM service
        with PerformanceLogger("event_llm_extraction", logger.logger, document_id=document_id):
            events = await call_event_llm_service(text, document_id, entities, context_fields)

        logger.info(
            f"Extracted {len(events)} events",
            extra={"document_id": document_id, "event_count": len(events)}
        )

        # Step 5: Event linking (single document - no cross-doc linking)
        # For single documents, we skip linking as it requires multiple events across documents
        linkages: List[EventLinkage] = []
        storylines: List[Storyline] = []

        # Step 6: Build ProcessedDocument
        processed_at = datetime.utcnow().isoformat() + "Z"
        normalized_date = preserved_fields.get("normalized_date")

        processing_time_ms = (time.time() - start_time) * 1000

        processed_doc = ProcessedDocument(
            document_id=document_id,
            job_id=None,  # Single document has no job ID
            processed_at=processed_at,
            normalized_date=normalized_date,
            original_text=text,
            source_document=document.model_dump(),
            extracted_entities=entities,
            extracted_soa_triplets=soa_triplets,
            events=events,
            event_linkages=linkages if linkages else None,
            storylines=storylines if storylines else None,
            processing_metadata={
                "pipeline_version": "1.0.0",
                "processing_time_ms": round(processing_time_ms, 2),
                "entity_count": len(entities),
                "triplet_count": len(soa_triplets),
                "event_count": len(events),
                "timestamp": processed_at,
            }
        )

        # Step 7: Save to storage backends
        output_locations = {}
        if storage_writer:
            try:
                save_results = storage_writer.save(processed_doc)
                logger.info(
                    f"Document saved to storage",
                    extra={"document_id": document_id, "backends": save_results}
                )
                # Build output locations dict for event
                if save_results.get("jsonl"):
                    output_locations["jsonl"] = f"file:///app/data/extracted_events_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
                if save_results.get("postgresql"):
                    output_locations["postgresql"] = f"postgresql://db/documents/{document_id}"
                if save_results.get("elasticsearch"):
                    output_locations["elasticsearch"] = f"http://es:9200/documents/{document_id}"
            except Exception as e:
                logger.error(
                    f"Failed to save document to storage: {e}",
                    exc_info=True,
                    extra={"document_id": document_id}
                )
                # Don't fail the request if storage fails

        # Step 8: Publish document.processed event
        if event_publisher and config.events.enabled and config.events.publish_events.document_processed:
            try:
                event_publisher.publish_document_processed(
                    document_id=document_id,
                    job_id=processed_doc.job_id or "single-document",
                    processing_time_seconds=processing_time_ms / 1000.0,
                    output_locations=output_locations,
                    metrics={
                        "event_count": len(events),
                        "entity_count": len(entities),
                        "soa_triplet_count": len(soa_triplets)
                    },
                    metadata={
                        "pipeline_version": "1.0.0",
                        "model_versions": {
                            "ner": config.ner_service.model_name,
                            "dp": config.dp_service.model_name,
                            "event_extraction": config.event_llm_service.model_name
                        }
                    }
                )
            except Exception as e:
                # Log but don't fail - event publishing is non-critical
                logger.warning(f"Failed to publish document.processed event: {e}")

        logger.info(
            f"Pipeline completed successfully for document: {document_id}",
            extra={
                "document_id": document_id,
                "processing_time_ms": round(processing_time_ms, 2)
            }
        )

        return processed_doc

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Pipeline failed for document: {document_id}",
            exc_info=True,
            extra={"document_id": document_id, "error": str(e)}
        )

        # Publish document.failed event
        if event_publisher and config.events.enabled and config.events.publish_events.document_failed:
            try:
                event_publisher.publish_document_failed(
                    document_id=document_id,
                    job_id="single-document",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    retry_count=0
                )
            except Exception as publish_error:
                logger.warning(f"Failed to publish document.failed event: {publish_error}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {str(e)}"
        )


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Service information endpoint."""
    return {
        "service": "NLP Processing Orchestrator",
        "version": "1.0.0",
        "stage": "Stage 2 - NLP Processing Service",
        "description": "Orchestrates event and entity extraction pipeline",
        "endpoints": {
            "health": "/health",
            "process_document": "/v1/documents",
            "process_batch": "/v1/documents/batch",
            "job_status": "/v1/jobs/{job_id}"
        }
    }


@app.get("/health", response_model=HealthCheckResponse)  # Unversioned for backward compatibility
@api_v1_router.get("/health", response_model=HealthCheckResponse)  # Versioned endpoint
async def health_check():
    """
    Health check endpoint - verifies all downstream services.

    Returns:
        HealthCheckResponse with status of all services
    """
    logger.debug("Health check requested")

    services_status = {}
    overall_status = "ok"

    # Check NER service
    try:
        response = await http_client.get(
            f"{service_config.ner_url}/api/v1/health",
            timeout=5.0
        )
        services_status["ner_service"] = {
            "status": "ok" if response.status_code == 200 else "degraded",
            "response_time_ms": response.elapsed.total_seconds() * 1000
        }
    except Exception as e:
        services_status["ner_service"] = {"status": "error", "error": str(e)}
        overall_status = "degraded"

    # Check DP service
    try:
        response = await http_client.get(
            f"{service_config.dp_url}/api/v1/health",
            timeout=5.0
        )
        services_status["dp_service"] = {
            "status": "ok" if response.status_code == 200 else "degraded",
            "response_time_ms": response.elapsed.total_seconds() * 1000
        }
    except Exception as e:
        services_status["dp_service"] = {"status": "error", "error": str(e)}
        overall_status = "degraded"

    # Check Event LLM service
    try:
        response = await http_client.get(
            f"{service_config.event_llm_url}/api/v1/health",
            timeout=5.0
        )
        services_status["event_llm_service"] = {
            "status": "ok" if response.status_code == 200 else "degraded",
            "response_time_ms": response.elapsed.total_seconds() * 1000
        }
    except Exception as e:
        services_status["event_llm_service"] = {"status": "error", "error": str(e)}
        overall_status = "degraded"

    # Check storage
    services_status["storage"] = {
        "status": "ok" if storage_writer else "error",
        "backends_count": len(storage_writer.backends) if storage_writer else 0
    }

    if not storage_writer:
        overall_status = "degraded"

    return HealthCheckResponse(
        status=overall_status,
        services=services_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0"
    )


@app.get("/metrics/resources")  # Unversioned for backward compatibility
@api_v1_router.get("/metrics/resources")  # Versioned endpoint
async def get_resource_metrics():
    """
    Get resource lifecycle metrics and current usage statistics.

    Returns detailed information about:
    - GPU VRAM usage
    - System RAM usage
    - Service idle states
    - Cleanup statistics
    - Resource pressure indicators

    Returns:
        Dict with comprehensive resource metrics
    """
    try:
        resource_manager = get_resource_manager()
        metrics = resource_manager.get_metrics()

        # Add timestamp
        metrics["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Calculate memory pressure status
        gpu_percent = metrics.get("gpu_memory_percent", 0)
        system_percent = metrics.get("system_memory_percent", 0)

        if gpu_percent >= 95 or system_percent >= 90:
            pressure_status = "critical"
        elif gpu_percent >= 85 or system_percent >= 80:
            pressure_status = "high"
        elif gpu_percent >= 70 or system_percent >= 65:
            pressure_status = "moderate"
        else:
            pressure_status = "normal"

        metrics["memory_pressure_status"] = pressure_status

        # Add recommendations if pressure is high
        if pressure_status in ["high", "critical"]:
            metrics["recommendations"] = [
                "Consider triggering manual cleanup via cleanup endpoints",
                "Review idle timeouts in settings.yaml",
                "Monitor batch job sizes to prevent memory spikes"
            ]

        logger.debug(
            "Resource metrics requested",
            extra={
                "gpu_usage": f"{gpu_percent:.1f}%",
                "system_usage": f"{system_percent:.1f}%",
                "pressure_status": pressure_status
            }
        )

        return metrics

    except Exception as e:
        logger.error(f"Failed to get resource metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve resource metrics: {str(e)}"
        )


@app.post("/admin/cleanup/{service_name}")  # Unversioned for backward compatibility
@api_v1_router.post("/admin/cleanup/{service_name}")  # Versioned endpoint
async def trigger_cleanup(
    service_name: str,
    force: bool = False,
    strategy: Optional[str] = None
):
    """
    Manually trigger resource cleanup for a specific service.

    Args:
        service_name: Service to cleanup (event_llm_service, ner_service, dp_service, orchestrator_service)
        force: Force cleanup regardless of idle state (default: False)
        strategy: Cleanup strategy override (aggressive, balanced, conservative)

    Returns:
        Dict with cleanup results
    """
    valid_services = ['event_llm_service', 'ner_service', 'dp_service', 'orchestrator_service']

    if service_name not in valid_services:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid service name. Must be one of: {', '.join(valid_services)}"
        )

    valid_strategies = ['aggressive', 'balanced', 'conservative', None]
    if strategy and strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Must be one of: aggressive, balanced, conservative"
        )

    try:
        resource_manager = get_resource_manager()

        from src.core.resource_lifecycle_manager import CleanupStrategy
        strategy_enum = CleanupStrategy(strategy) if strategy else None

        result = resource_manager.cleanup_service(
            service_name=service_name,
            strategy=strategy_enum,
            force=force
        )

        logger.info(
            f"Manual cleanup triggered for {service_name}",
            extra={
                "service": service_name,
                "force": force,
                "strategy": strategy,
                "result": result
            }
        )

        return {
            "success": True,
            "service": service_name,
            "cleanup_result": result,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    except Exception as e:
        logger.error(f"Failed to trigger cleanup for {service_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger cleanup: {str(e)}"
        )


@api_v1_router.post("/documents", response_model=ProcessDocumentResponse)
async def process_document(request: ProcessDocumentRequest):
    """
    Process a single document through the NLP pipeline.

    Args:
        request: ProcessDocumentRequest with Stage1Document

    Returns:
        ProcessDocumentResponse with ProcessedDocument result
    """
    start_time = time.time()
    document = request.document
    document_id = document.document_id

    logger.info(
        f"Received document processing request",
        extra={"document_id": document_id}
    )

    try:
        processed_doc = await process_single_document(document)
        processing_time_ms = (time.time() - start_time) * 1000

        return ProcessDocumentResponse(
            success=True,
            document_id=document_id,
            result=processed_doc,
            error=None,
            processing_time_ms=round(processing_time_ms, 2)
        )

    except HTTPException as e:
        processing_time_ms = (time.time() - start_time) * 1000
        return ProcessDocumentResponse(
            success=False,
            document_id=document_id,
            result=None,
            error=str(e.detail),
            processing_time_ms=round(processing_time_ms, 2)
        )

    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Unexpected error processing document: {e}", exc_info=True)
        return ProcessDocumentResponse(
            success=False,
            document_id=document_id,
            result=None,
            error=f"Internal server error: {str(e)}",
            processing_time_ms=round(processing_time_ms, 2)
        )


@api_v1_router.post("/documents/batch", response_model=ProcessBatchResponse)
async def process_batch(request: ProcessBatchRequest):
    """
    Submit a batch of documents for asynchronous processing.

    Args:
        request: ProcessBatchRequest with list of Stage1Documents

    Returns:
        ProcessBatchResponse with job_id for tracking
    """
    documents = request.documents
    batch_id = request.batch_id or f"batch_{uuid.uuid4().hex[:12]}"

    # Validate batch size
    batch_size = len(documents)
    max_batch_size = config.orchestrator_service.max_batch_size

    if batch_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch cannot be empty. Please provide at least one document."
        )

    if batch_size > max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size ({batch_size}) exceeds maximum allowed ({max_batch_size}). "
                   f"Please split into smaller batches."
        )

    logger.info(
        f"Received batch processing request (validated)",
        extra={"batch_id": batch_id, "document_count": batch_size, "max_allowed": max_batch_size}
    )

    try:
        # Import Celery task
        from src.core.celery_tasks import process_batch_task

        # Convert documents to dict format
        documents_dicts = [doc.model_dump() for doc in documents]

        # Submit to Celery
        task = process_batch_task.delay(
            documents=documents_dicts,
            batch_id=batch_id,
            options=request.options or {}
        )

        logger.info(
            f"Batch submitted to Celery",
            extra={"batch_id": batch_id, "job_id": task.id, "document_count": len(documents)}
        )

        return ProcessBatchResponse(
            success=True,
            batch_id=batch_id,
            job_id=task.id,
            document_count=len(documents),
            message=f"Batch processing started. Track progress with job_id: {task.id}"
        )

    except ImportError:
        logger.error("Celery tasks not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Batch processing not available. Celery worker may not be running."
        )

    except Exception as e:
        logger.error(f"Failed to submit batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit batch for processing: {str(e)}"
        )


@api_v1_router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of a batch processing job.

    Args:
        job_id: Celery task ID

    Returns:
        JobStatusResponse with job status and progress
    """
    logger.debug(f"Job status requested", extra={"job_id": job_id})

    try:
        from celery.result import AsyncResult

        result = AsyncResult(job_id)

        # Map Celery states to response
        status_map = {
            "PENDING": "PENDING",
            "STARTED": "STARTED",
            "SUCCESS": "SUCCESS",
            "FAILURE": "FAILURE",
            "RETRY": "STARTED",
            "REVOKED": "FAILURE"
        }

        job_status = status_map.get(result.state, result.state)

        response_data = {
            "job_id": job_id,
            "status": job_status,
            "progress": None,
            "documents_processed": None,
            "documents_total": None,
            "result": None,
            "error": None
        }

        # Add result data if available
        if result.state == "SUCCESS":
            response_data["result"] = result.result
            response_data["progress"] = 100.0
        elif result.state == "FAILURE":
            response_data["error"] = str(result.info)
        elif result.state == "STARTED":
            # Try to get progress info
            if result.info:
                response_data["progress"] = result.info.get("progress")
                response_data["documents_processed"] = result.info.get("documents_processed")
                response_data["documents_total"] = result.info.get("documents_total")

        return JobStatusResponse(**response_data)

    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job status not available. Celery not configured."
        )

    except Exception as e:
        logger.error(f"Failed to get job status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job status: {str(e)}"
        )


# =============================================================================
# Error Handlers
# =============================================================================
# Error handlers are now managed by shared middleware (src/utils/middleware.py)


# =============================================================================
# Include Versioned Router
# =============================================================================
# Note: This must be done AFTER all endpoints are defined
# Router will be included at the end of the file, before main()


# =============================================================================
# Include Versioned Router
# =============================================================================
# Include the versioned API router
app.include_router(api_v1_router)

logger.info(f"API {API_VERSION} routes registered at {API_VERSION_PREFIX}")


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    # Get orchestrator settings
    settings = get_settings()
    port = settings.orchestrator_service.port

    logger.info(f"Starting orchestrator service on port {port}")

    uvicorn.run(
        "src.api.orchestrator_service:app",
        host="0.0.0.0",
        port=port,
        reload=settings.development.reload_on_change,
        log_config=None  # Use our custom logging
    )
