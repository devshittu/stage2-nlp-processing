"""
event_llm_service.py

FastAPI application for Event LLM Service (Port 8003).
Handles event extraction from text using vLLM-optimized inference.

Features:
- POST /extract endpoint for event extraction
- GET /health endpoint for health checks
- Comprehensive error handling
- Structured JSON logging
- CORS middleware support
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.utils.middleware import setup_middleware
from src.core.event_llm_logic import get_event_llm_model
from src.schemas.data_models import Event, EventLLMServiceResponse
from src.utils.config_manager import get_settings
from src.utils.logger import get_logger, initialize_logging_from_config
from src.core.resource_lifecycle_manager import get_resource_manager
from src.core.cleanup_hooks import create_cleanup_hook


# =============================================================================
# Logging Setup
# =============================================================================

initialize_logging_from_config()
logger = get_logger(__name__, service="event_llm_service")


# =============================================================================
# Request/Response Models
# =============================================================================

class ExtractEventRequest(BaseModel):
    """Request model for event extraction endpoint."""
    text: str = Field(..., description="Text to extract events from", min_length=1)
    document_id: str = Field(..., description="Unique document identifier")
    entities: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Pre-extracted entities from NER service (for context enrichment)"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional context (title, author, publication_date, etc.)"
    )
    domain_hint: Optional[str] = Field(
        None,
        description="Optional domain hint for focused extraction"
    )

    @field_validator('text')
    @classmethod
    def validate_text_length(cls, v):
        """Validate text is within max length."""
        settings = get_settings()
        if len(v) > settings.general.max_text_length:
            raise ValueError(
                f"Text exceeds maximum length of {settings.general.max_text_length} characters"
            )
        return v


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status (ok, degraded, error)")
    service: str = Field(..., description="Service name")
    timestamp: str = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: str = Field(..., description="LLM model name")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# =============================================================================
# FastAPI Application Setup
# =============================================================================

# Global model instance
_model_instance = None

# Resource lifecycle manager
_resource_manager = None
_cleanup_hook = None

# =============================================================================
# Backpressure Control (2026 Best Practice)
# =============================================================================
# Track active requests to implement 429 backpressure signaling
import threading
import asyncio

_active_requests = 0
_active_requests_lock = threading.Lock()
MAX_CONCURRENT_REQUESTS = 2  # Match throttler limit
BACKPRESSURE_RETRY_AFTER = 15  # Seconds to wait before retry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown logic.
    """
    # Startup
    logger.info("Event LLM Service starting up")
    try:
        global _model_instance, _resource_manager, _cleanup_hook

        # Initialize resource lifecycle manager
        _resource_manager = get_resource_manager()
        logger.info("Resource lifecycle manager initialized")

        # Load Event LLM model
        _model_instance = get_event_llm_model()
        logger.info(
            "Event LLM model loaded successfully",
            extra={
                "model_name": _model_instance.settings.model_name,
                "use_vllm": _model_instance.use_vllm
            }
        )

        # Register vLLM cleanup hook
        _cleanup_hook = create_cleanup_hook('vllm', _model_instance)
        _resource_manager.register_cleanup_callback(
            service_name='event_llm_service',
            callback=_cleanup_hook.cleanup,
            priority=100  # High priority for GPU cleanup
        )
        logger.info("vLLM cleanup hook registered")

    except Exception as e:
        logger.error("Failed to load Event LLM model during startup", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Event LLM Service shutting down")

    # Stop resource monitoring
    if _resource_manager:
        _resource_manager.stop_monitoring()


# API Versioning Configuration
API_VERSION = "v1"
API_VERSION_PREFIX = f"/api/{API_VERSION}"

# Create FastAPI app
app = FastAPI(
    title="Event LLM Service",
    description="Event extraction service using vLLM-optimized LLM inference",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=f"{API_VERSION_PREFIX}/docs",
    redoc_url=f"{API_VERSION_PREFIX}/redoc",
    openapi_url=f"{API_VERSION_PREFIX}/openapi.json",
)

# Create versioned API router
api_v1_router = APIRouter(prefix=API_VERSION_PREFIX, tags=["v1"])

# Setup secure middleware (CORS, logging, error handlers)
setup_middleware(app, "event_llm_service", allow_cors_credentials=False)


# Request/response middleware is now handled by shared middleware (src/utils/middleware.py)


# =============================================================================
# Health Check Endpoint
# =============================================================================

@api_v1_router.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Health check"
)
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Health check (unversioned for backward compatibility)"
)
async def health_check():
    """
    Check service health and model status.

    Returns:
        HealthCheckResponse with service status and model info
    """
    try:
        if _model_instance is None:
            return HealthCheckResponse(
                status="degraded",
                service="event_llm_service",
                timestamp=datetime.utcnow().isoformat() + "Z",
                model_loaded=False,
                model_name="unknown"
            )

        return HealthCheckResponse(
            status="ok",
            service="event_llm_service",
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_loaded=True,
            model_name=_model_instance.settings.model_name
        )
    except Exception as e:
        logger.error("Health check failed", exc_info=True)
        raise HTTPException(status_code=503, detail="Service unavailable")


# =============================================================================
# Kubernetes-style Liveness and Readiness Probes (2026 Best Practice)
# =============================================================================

@api_v1_router.get(
    "/live",
    tags=["Health"],
    summary="Liveness probe (is process alive?)"
)
@app.get(
    "/live",
    tags=["Health"],
    summary="Liveness probe (unversioned)"
)
async def liveness_probe():
    """
    Kubernetes liveness probe - lightweight check that process is running.
    Should ALWAYS return 200 if the process is alive (no dependency checks).

    Returns:
        {"status": "alive"} with 200 OK
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat() + "Z"}


@api_v1_router.get(
    "/ready",
    tags=["Health"],
    summary="Readiness probe (can handle requests?)"
)
@app.get(
    "/ready",
    tags=["Health"],
    summary="Readiness probe (unversioned)"
)
async def readiness_probe():
    """
    Kubernetes readiness probe - checks if service can handle traffic.

    Returns:
        200 OK if model loaded AND service has capacity
        503 Service Unavailable if not ready

    Checks:
    1. Model is loaded
    2. Service is not at capacity (backpressure check)
    """
    # Check 1: Model loaded?
    if _model_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded - service not ready"
        )

    # Check 2: At capacity? (backpressure)
    with _active_requests_lock:
        if _active_requests >= MAX_CONCURRENT_REQUESTS:
            raise HTTPException(
                status_code=503,
                detail=f"At capacity ({_active_requests}/{MAX_CONCURRENT_REQUESTS}) - not accepting new requests",
                headers={"Retry-After": str(BACKPRESSURE_RETRY_AFTER)}
            )

    return {
        "status": "ready",
        "model_loaded": True,
        "active_requests": _active_requests,
        "max_concurrent": MAX_CONCURRENT_REQUESTS,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# =============================================================================
# Event Extraction Endpoint
# =============================================================================

@api_v1_router.post(
    "/extract",
    response_model=EventLLMServiceResponse,
    tags=["Event Extraction"],
    summary="Extract events from text"
)
async def extract_events(request: ExtractEventRequest):
    """
    Extract events from text using LLM.

    Args:
        request: ExtractEventRequest containing text, document_id, context, and domain_hint

    Returns:
        EventLLMServiceResponse with extracted events and metadata

    Raises:
        HTTPException: If extraction fails or model is not loaded
        HTTPException(429): If service is at capacity (backpressure)
    """
    global _active_requests
    start_time = time.time()
    document_id = request.document_id

    # =========================================================================
    # BACKPRESSURE CHECK: Return 429 if at capacity
    # =========================================================================
    with _active_requests_lock:
        if _active_requests >= MAX_CONCURRENT_REQUESTS:
            logger.warning(
                f"Backpressure triggered: {_active_requests} active requests",
                extra={
                    "document_id": document_id,
                    "active_requests": _active_requests,
                    "max_concurrent": MAX_CONCURRENT_REQUESTS
                }
            )
            raise HTTPException(
                status_code=429,
                detail=f"Service at capacity ({_active_requests}/{MAX_CONCURRENT_REQUESTS} requests). Retry later.",
                headers={"Retry-After": str(BACKPRESSURE_RETRY_AFTER)}
            )
        _active_requests += 1

    try:
        logger.info(
            f"Extracting events from document: {document_id}",
            extra={
                "document_id": document_id,
                "text_length": len(request.text),
                "has_context": request.context is not None,
                "domain_hint": request.domain_hint,
                "active_requests": _active_requests
            }
        )

        # Validate model is loaded
        if _model_instance is None:
            logger.error("Event LLM model not loaded")
            raise HTTPException(
                status_code=503,
                detail="Event LLM model not loaded. Service starting up."
            )

        # Track resource activity for this task
        with _resource_manager.track_task("event_llm_service"):
            # Extract events (entities can be used for context enrichment if needed)
            events: List[Event] = _model_instance.extract_events(
                text=request.text,
                document_id=document_id,
                context=request.context,
                domain_hint=request.domain_hint
            )

            # Note: entities parameter is available in request.entities if needed for future enhancements

            # Calculate chunks processed
            chunks = _model_instance.chunk_text(request.text)
            chunks_processed = len(chunks)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Successfully extracted {len(events)} events from {chunks_processed} chunks",
                extra={
                    "document_id": document_id,
                    "event_count": len(events),
                    "chunks_processed": chunks_processed,
                    "processing_time_ms": round(processing_time_ms, 2)
                }
            )

            return EventLLMServiceResponse(
                document_id=document_id,
                events=events,
                processing_time_ms=round(processing_time_ms, 2),
                model_name=_model_instance.settings.model_name,
                chunks_processed=chunks_processed
            )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(
            f"Validation error for document {document_id}: {str(e)}",
            extra={"document_id": document_id, "error": str(e)}
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Event extraction failed for document {document_id}",
            exc_info=True,
            extra={
                "document_id": document_id,
                "processing_time_ms": round(processing_time_ms, 2)
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Event extraction failed: {str(e)}"
        )
    finally:
        # Always decrement active requests counter (backpressure cleanup)
        with _active_requests_lock:
            _active_requests -= 1
            logger.debug(
                f"Request completed, active requests: {_active_requests}",
                extra={"active_requests": _active_requests}
            )


# =============================================================================
# Error Handling
# =============================================================================
# Error handlers are now managed by shared middleware (src/utils/middleware.py)


# =============================================================================
# Router Registration
# =============================================================================

# Include the versioned API router
app.include_router(api_v1_router)
logger.info(f"API {API_VERSION} routes registered at {API_VERSION_PREFIX}")


# =============================================================================
# Root Endpoint
# =============================================================================

@api_v1_router.get(
    "/",
    tags=["Info"],
    summary="Service information"
)
async def root():
    """Service information endpoint."""
    settings = get_settings()
    return {
        "service": "Event LLM Service",
        "version": "1.0.0",
        "api_version": API_VERSION,
        "port": settings.event_llm_service.port,
        "model": settings.event_llm_service.model_name,
        "use_vllm": settings.event_llm_service.use_vllm,
        "endpoints": {
            "health": f"{API_VERSION_PREFIX}/health",
            "extract": f"{API_VERSION_PREFIX}/extract",
            "docs": f"{API_VERSION_PREFIX}/docs"
        }
    }


# =============================================================================
# Module Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    port = settings.event_llm_service.port

    logger.info(
        f"Starting Event LLM Service on port {port}",
        extra={
            "port": port,
            "model": settings.event_llm_service.model_name,
            "use_vllm": settings.event_llm_service.use_vllm
        }
    )

    uvicorn.run(
        "src.api.event_llm_service:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
