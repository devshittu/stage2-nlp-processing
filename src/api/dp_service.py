"""
dp_service.py

Dependency Parsing (DP) FastAPI service for Stage 2 NLP Processing Service.
Handles HTTP requests for dependency parsing and SOA triplet extraction.

Endpoints:
- POST /parse - Parse text and extract SOA triplets and dependencies
- GET /health - Health check endpoint
- GET /model-info - Get information about loaded model

Features:
- FastAPI with async request handling
- GPU-accelerated dependency parsing
- CORS middleware for cross-origin requests
- Comprehensive error handling and logging
- Request validation with Pydantic
"""

import logging
import time
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.utils.middleware import setup_middleware
from src.core.dp_logic import get_dependency_parser, DependencyParser
from src.schemas.data_models import DPServiceResponse, SOATriplet, DependencyRelation
from src.utils.config_manager import get_settings
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__, service="dp_service")

# =============================================================================
# Configuration
# =============================================================================

settings = get_settings()
dp_settings = settings.dp_service

# =============================================================================
# Request/Response Models
# =============================================================================


class ParseRequest(BaseModel):
    """Request model for text parsing."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=settings.general.max_text_length,
        description="Text to parse"
    )
    document_id: str = Field(
        default="",
        description="Optional document identifier"
    )


class ParseResponse(BaseModel):
    """Response model for parsing request."""
    document_id: str = Field(..., description="Document ID")
    soa_triplets: list[SOATriplet] = Field(..., description="Subject-Object-Action triplets")
    dependencies: Optional[list[DependencyRelation]] = Field(None, description="Dependency relations")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_name: str = Field(..., description="Model name")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: str = Field(..., description="Model name")
    timestamp: str = Field(..., description="Check timestamp")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Model name")
    loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device (cuda or cpu)")
    batch_size: int = Field(..., description="Batch size")
    active_pipes: Optional[list[str]] = Field(None, description="Active spaCy pipes")
    vocab_size: Optional[int] = Field(None, description="Vocabulary size")
    gpu_enabled: bool = Field(..., description="GPU enabled")


# =============================================================================
# Global State
# =============================================================================

parser: Optional[DependencyParser] = None


# =============================================================================
# Lifespan Events
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan.

    Load model on startup, release on shutdown.
    """
    global parser

    # Startup
    logger.info(f"Starting DP Service on port {dp_settings.port}")
    try:
        parser = get_dependency_parser()
        logger.info("Dependency parser initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize parser: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down DP Service")


# =============================================================================
# FastAPI Application
# =============================================================================

# API Versioning Configuration
API_VERSION = "v1"
API_VERSION_PREFIX = f"/api/{API_VERSION}"

app = FastAPI(
    title="Dependency Parsing Service",
    description="NLP service for dependency parsing and SOA triplet extraction",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=f"{API_VERSION_PREFIX}/docs",
    redoc_url=f"{API_VERSION_PREFIX}/redoc",
    openapi_url=f"{API_VERSION_PREFIX}/openapi.json",
)

# Create versioned API router
api_v1_router = APIRouter(prefix=API_VERSION_PREFIX, tags=["v1"])

# Setup secure middleware (CORS, logging, error handlers)
setup_middleware(app, "dp_service", allow_cors_credentials=False)


# Request logging is now handled by shared middleware (src/utils/middleware.py)


# =============================================================================
# Endpoints
# =============================================================================


@api_v1_router.get("/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)  # Backward compatibility: unversioned endpoint
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse with service status
    """
    try:
        is_loaded = parser is not None and parser.nlp is not None

        return HealthResponse(
            status="ok" if is_loaded else "degraded",
            model_loaded=is_loaded,
            model_name=dp_settings.model_name,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Health check failed")


@api_v1_router.post("/parse", response_model=ParseResponse)
async def parse(request: ParseRequest) -> ParseResponse:
    """
    Parse text to extract SOA triplets and dependency relations.

    Args:
        request: ParseRequest containing text and optional document_id

    Returns:
        ParseResponse with extracted triplets and dependencies

    Raises:
        HTTPException: If parsing fails
    """
    if not parser:
        logger.error("Parser not initialized")
        raise HTTPException(status_code=503, detail="Parser not initialized")

    try:
        start_time = time.time()

        # Validate input
        if not request.text or not request.text.strip():
            raise ValueError("Empty text provided")

        if len(request.text) > settings.general.max_text_length:
            raise ValueError(
                f"Text exceeds maximum length of {settings.general.max_text_length} characters"
            )

        # Parse text
        soa_triplets, dependencies = parser.parse(request.text)

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Text parsed successfully",
            extra={
                "document_id": request.document_id,
                "text_length": len(request.text),
                "triplets": len(soa_triplets),
                "dependencies": len(dependencies),
                "processing_time_ms": f"{processing_time_ms:.2f}"
            }
        )

        return ParseResponse(
            document_id=request.document_id or "unknown",
            soa_triplets=soa_triplets,
            dependencies=dependencies if dependencies else None,
            processing_time_ms=processing_time_ms,
            model_name=dp_settings.model_name
        )

    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(
            f"Parsing failed: {e}",
            extra={"document_id": request.document_id},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@api_v1_router.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """
    Get information about loaded model.

    Returns:
        ModelInfoResponse with model details
    """
    if not parser:
        raise HTTPException(status_code=503, detail="Parser not initialized")

    try:
        info = parser.get_model_info()

        return ModelInfoResponse(
            model_name=info.get("model_name", "unknown"),
            loaded=info.get("loaded", False),
            device=info.get("device", "unknown"),
            batch_size=info.get("batch_size", 0),
            active_pipes=info.get("active_pipes"),
            vocab_size=info.get("vocab_size"),
            gpu_enabled=info.get("gpu_enabled", False)
        )
    except Exception as e:
        logger.error(f"Failed to get model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get model info")


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
# Root Endpoint
# =============================================================================


@api_v1_router.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Dependency Parsing Service",
        "version": "1.0.0",
        "api_version": API_VERSION,
        "endpoints": {
            "health": f"{API_VERSION_PREFIX}/health",
            "parse": f"{API_VERSION_PREFIX}/parse",
            "model_info": f"{API_VERSION_PREFIX}/model-info"
        }
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Setup logger
    setup_logging(
        log_level=settings.general.log_level,
        log_file=None,
        log_format="json"
    )

    logger.info(f"Starting Dependency Parsing Service on port {dp_settings.port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=dp_settings.port,
        workers=1,
        log_level=settings.general.log_level.lower()
    )
