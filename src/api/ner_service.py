"""
ner_service.py

Named Entity Recognition FastAPI service for Stage 2 NLP Processing.
Provides HTTP endpoints for entity extraction using transformer-based NER models.

Features:
- GPU-accelerated entity extraction
- Comprehensive error handling
- Structured logging
- CORS middleware
- Health check endpoint
- Request/response validation
"""

import time
import uvicorn
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.utils.middleware import setup_middleware
from src.core.ner_logic import get_ner_model
from src.schemas.data_models import Entity, NERServiceResponse
from src.utils.config_manager import get_settings
from src.utils.logger import get_logger, initialize_logging_from_config


# =============================================================================
# Logging Setup
# =============================================================================

# Initialize logging from configuration
config = get_settings()
initialize_logging_from_config(config)
logger = get_logger(__name__, service="ner_service")


# =============================================================================
# Request/Response Models
# =============================================================================

class NERExtractRequest(BaseModel):
    """Request model for entity extraction."""
    text: str = Field(..., min_length=1, description="Text to extract entities from")
    document_id: Optional[str] = Field(None, description="Optional document identifier")


class NERExtractResponse(BaseModel):
    """Response model for entity extraction."""
    document_id: str = Field(..., description="Document identifier")
    entities: List[Entity] = Field(..., description="Extracted entities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_name: str = Field(..., description="NER model name")


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status (ok, degraded, error)")
    service: str = Field(default="ner_service", description="Service name")
    timestamp: str = Field(..., description="Health check timestamp")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")


# =============================================================================
# Global Resources (Module-level for lifespan access)
# =============================================================================

# Global NER model instance
_ner_model = None


def get_ner_model_instance():
    """Get or initialize NER model instance."""
    global _ner_model
    if _ner_model is None:
        logger.info("Initializing NER model instance")
        _ner_model = get_ner_model()
    return _ner_model


# =============================================================================
# FastAPI Application Setup with Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown logic.
    """
    # Startup
    logger.info(
        "NER service starting up",
        extra={
            "version": "1.0.0",
            "port": config.ner_service.port,
            "model_name": config.ner_service.model_name
        }
    )

    try:
        # Pre-load model on startup
        model = get_ner_model_instance()
        model_info = model.get_model_info()
        logger.info(
            "NER model loaded successfully at startup",
            extra={"model_info": model_info}
        )
    except Exception as e:
        logger.error(
            "Failed to load NER model at startup",
            extra={"error": str(e)},
            exc_info=True
        )

    yield

    # Shutdown
    logger.info("NER service shutting down")


# API Versioning Configuration
API_VERSION = "v1"
API_VERSION_PREFIX = f"/api/{API_VERSION}"

app = FastAPI(
    title="Named Entity Recognition Service",
    description="Entity extraction service for Stage 2 NLP Processing Pipeline",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=f"{API_VERSION_PREFIX}/docs",
    redoc_url=f"{API_VERSION_PREFIX}/redoc",
    openapi_url=f"{API_VERSION_PREFIX}/openapi.json",
)

# Create versioned API router
api_v1_router = APIRouter(prefix=API_VERSION_PREFIX, tags=["v1"])

# Setup secure middleware (CORS, logging, error handlers)
setup_middleware(app, "ner_service", allow_cors_credentials=False)


# =============================================================================
# API Endpoints
# =============================================================================

@api_v1_router.get("/health")
@app.get("/health")  # Backward compatibility: unversioned endpoint
async def health_check() -> HealthStatus:
    """
    Health check endpoint.

    Returns:
        HealthStatus: Service health information

    Example:
        GET /health
        {
            "status": "ok",
            "service": "ner_service",
            "timestamp": "2024-01-15T10:30:00Z",
            "model_info": {...}
        }
    """
    try:
        model = get_ner_model_instance()
        model_info = model.get_model_info()

        logger.debug("Health check performed", extra={"status": "ok"})

        return HealthStatus(
            status="ok",
            service="ner_service",
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_info=model_info
        )

    except Exception as e:
        logger.error(
            "Health check failed",
            extra={"error": str(e)},
            exc_info=True
        )

        return HealthStatus(
            status="error",
            service="ner_service",
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_info=None
        )


@api_v1_router.post("/extract", response_model=NERExtractResponse)
async def extract_entities(request: NERExtractRequest) -> NERExtractResponse:
    """
    Extract named entities from text.

    Args:
        request: NERExtractRequest containing text and optional document_id

    Returns:
        NERExtractResponse: Extracted entities with metadata

    Raises:
        HTTPException: If extraction fails

    Example:
        POST /extract
        {
            "text": "Donald Trump met Angela Merkel in Berlin on Monday.",
            "document_id": "doc123"
        }

        Response:
        {
            "document_id": "doc123",
            "entities": [
                {
                    "text": "Donald Trump",
                    "type": "PER",
                    "start_char": 0,
                    "end_char": 12,
                    "confidence": 0.95,
                    "context": "[Donald Trump] met Angela Merkel",
                    "entity_id": "doc123_entity_0"
                }
            ],
            "processing_time_ms": 1234.56,
            "model_name": "Babelscape/wikineural-multilingual-ner"
        }
    """
    # Validate input
    if not request.text or not request.text.strip():
        logger.warning(
            "Empty text provided to extract endpoint",
            extra={"document_id": request.document_id}
        )
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    # Generate document ID if not provided
    if not request.document_id:
        request.document_id = f"auto_{int(time.time() * 1000)}"

    try:
        # Record start time
        start_time = time.time()

        # Extract entities
        model = get_ner_model_instance()
        entities = model.extract_entities(
            text=request.text,
            document_id=request.document_id,
            use_cache=True
        )

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Entities extracted successfully",
            extra={
                "document_id": request.document_id,
                "entity_count": len(entities),
                "text_length": len(request.text),
                "processing_time_ms": round(processing_time_ms, 2)
            }
        )

        return NERExtractResponse(
            document_id=request.document_id,
            entities=entities,
            processing_time_ms=round(processing_time_ms, 2),
            model_name=config.ner_service.model_name
        )

    except Exception as e:
        logger.error(
            "Entity extraction failed",
            extra={
                "document_id": request.document_id,
                "text_length": len(request.text),
                "error": str(e)
            },
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Entity extraction failed: {str(e)}"
        )


@api_v1_router.get("/model-info")
async def get_model_info() -> Dict[str, Any]:
    """
    Get detailed model information and statistics.

    Returns:
        Dict with model information including name, device, entity types, etc.

    Example:
        GET /model-info
        {
            "model_name": "Babelscape/wikineural-multilingual-ner",
            "device": "cuda",
            "entity_types": ["PER", "ORG", "LOC", ...],
            "confidence_threshold": 0.75,
            "cache_enabled": true,
            "batch_size": 16,
            "max_length": 512,
            "model_parameters": 12345678
        }
    """
    try:
        model = get_ner_model_instance()
        model_info = model.get_model_info()

        logger.debug(
            "Model info requested",
            extra={"model_name": model_info.get("model_name")}
        )

        return model_info

    except Exception as e:
        logger.error(
            "Failed to retrieve model information",
            extra={"error": str(e)},
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model information"
        )


@api_v1_router.post("/clear-cache")
async def clear_cache() -> Dict[str, Any]:
    """
    Clear the entity cache.

    Returns:
        Dict with cache clear status

    Example:
        POST /clear-cache
        {
            "status": "success",
            "message": "Cache cleared successfully",
            "timestamp": "2024-01-15T10:30:00Z"
        }
    """
    try:
        model = get_ner_model_instance()
        success = model.clear_cache()

        if success:
            logger.info("Entity cache cleared successfully")
            return {
                "status": "success",
                "message": "Cache cleared successfully",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        else:
            logger.warning("Cache clearing failed or cache not enabled")
            return {
                "status": "warning",
                "message": "Cache not enabled or clearing failed",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    except Exception as e:
        logger.error(
            "Cache clearing failed",
            extra={"error": str(e)},
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Cache clearing failed: {str(e)}"
        )


# =============================================================================
# Error Handlers
# =============================================================================
# Error handlers are now managed by shared middleware (src/utils/middleware.py)


# =============================================================================
# Router Registration
# =============================================================================

# Include the versioned API router
app.include_router(api_v1_router)
logger.info(f"API {API_VERSION} routes registered at {API_VERSION_PREFIX}")


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    """
    Run the NER service with uvicorn.

    Usage:
        python -m src.api.ner_service

    The service will start on port 8001 by default (configurable via settings).
    """
    port = config.ner_service.port

    logger.info(
        f"Starting NER FastAPI service on port {port}",
        extra={
            "port": port,
            "model": config.ner_service.model_name,
            "gpu_enabled": config.general.gpu_enabled
        }
    )

    uvicorn.run(
        "src.api.ner_service:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level=config.general.log_level.lower(),
        access_log=True,
        workers=1  # Single worker for GPU model
    )
