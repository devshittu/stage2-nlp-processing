"""
middleware.py

Shared middleware components for FastAPI services.
Provides CORS configuration, error handling, and logging middleware.

Features:
- Secure CORS configuration with environment-based origins
- Standardized error handlers
- Request/response logging middleware
- Performance tracking
"""

import os
import time
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


# =============================================================================
# CORS Configuration
# =============================================================================

def get_allowed_origins() -> List[str]:
    """
    Get allowed CORS origins from environment variable.

    Returns:
        List of allowed origins

    Examples:
        CORS_ORIGINS="http://localhost:3000,https://app.example.com"
    """
    origins_env = os.getenv("CORS_ORIGINS", "")

    if not origins_env:
        # Default to localhost for development
        logger.warning(
            "CORS_ORIGINS not set. Defaulting to localhost. "
            "Set CORS_ORIGINS environment variable for production."
        )
        return [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8000",
        ]

    # Parse comma-separated origins
    origins = [origin.strip() for origin in origins_env.split(",")]
    logger.info(f"CORS configured for origins: {origins}")
    return origins


def configure_cors(app, allow_credentials: bool = False):
    """
    Configure CORS middleware with secure defaults.

    Args:
        app: FastAPI application instance
        allow_credentials: Whether to allow credentials (default: False)

    Security Notes:
        - Origins are restricted based on CORS_ORIGINS environment variable
        - Credentials are disabled by default
        - Only essential methods and headers are allowed
    """
    allowed_origins = get_allowed_origins()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,  # Restricted to configured origins
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-Request-ID",
            "X-API-Key"
        ],
        expose_headers=["X-Request-ID", "X-Process-Time"],
        max_age=600,  # Cache preflight requests for 10 minutes
    )

    logger.info(
        f"CORS middleware configured: "
        f"origins={len(allowed_origins)}, credentials={allow_credentials}"
    )


# =============================================================================
# Logging Middleware
# =============================================================================

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.
    Adds performance metrics and request tracking.
    """

    async def dispatch(self, request: Request, call_next):
        """Process request and log details."""
        request_id = request.headers.get("x-request-id", "unknown")
        start_time = time.time()

        # Log incoming request
        logger.info(
            f"→ {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown"
            }
        )

        try:
            # Process request
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000

            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

            # Log response
            logger.info(
                f"← {request.method} {request.url.path} {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time_ms": round(process_time, 2)
                }
            )

            return response

        except Exception as e:
            process_time = (time.time() - start_time) * 1000
            logger.error(
                f"✗ {request.method} {request.url.path} - Error: {str(e)}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "process_time_ms": round(process_time, 2)
                }
            )
            raise


# =============================================================================
# Error Handlers
# =============================================================================

class ErrorResponse:
    """Standard error response structure."""

    @staticmethod
    def create(
        error_type: str,
        message: str,
        status_code: int = 500,
        request_id: Optional[str] = None,
        details: Optional[dict] = None
    ) -> dict:
        """
        Create standardized error response.

        Args:
            error_type: Type of error (e.g., "ValidationError", "NotFound")
            message: Human-readable error message
            status_code: HTTP status code
            request_id: Optional request tracking ID
            details: Optional additional error details

        Returns:
            Dictionary formatted as error response
        """
        response = {
            "error": error_type,
            "message": message,
            "status_code": status_code,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        if request_id:
            response["request_id"] = request_id

        if details:
            response["details"] = details

        return response


async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTPException with standardized response.

    Args:
        request: FastAPI request
        exc: HTTPException instance

    Returns:
        JSONResponse with error details
    """
    request_id = request.headers.get("x-request-id", "unknown")

    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": request.url.path
        }
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse.create(
            error_type="HTTPException",
            message=str(exc.detail),
            status_code=exc.status_code,
            request_id=request_id
        )
    )


async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions with standardized response.

    Args:
        request: FastAPI request
        exc: Exception instance

    Returns:
        JSONResponse with error details
    """
    request_id = request.headers.get("x-request-id", "unknown")

    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)}",
        exc_info=True,
        extra={
            "request_id": request_id,
            "exception_type": type(exc).__name__,
            "path": request.url.path,
            "method": request.method
        }
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse.create(
            error_type="InternalServerError",
            message="An unexpected error occurred. Please contact support if this persists.",
            status_code=500,
            request_id=request_id
        )
    )


async def validation_exception_handler(request: Request, exc: Exception):
    """
    Handle validation errors with detailed response.

    Args:
        request: FastAPI request
        exc: Validation exception

    Returns:
        JSONResponse with validation error details
    """
    request_id = request.headers.get("x-request-id", "unknown")

    logger.warning(
        f"Validation error: {str(exc)}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "error": str(exc)
        }
    )

    return JSONResponse(
        status_code=422,
        content=ErrorResponse.create(
            error_type="ValidationError",
            message="Request validation failed",
            status_code=422,
            request_id=request_id,
            details={"validation_errors": str(exc)}
        )
    )


# =============================================================================
# Middleware Setup Helper
# =============================================================================

def setup_middleware(app, service_name: str, allow_cors_credentials: bool = False):
    """
    Setup all middleware for a FastAPI application.

    Args:
        app: FastAPI application instance
        service_name: Name of the service (for logging)
        allow_cors_credentials: Whether to allow CORS credentials

    Example:
        from src.utils.middleware import setup_middleware

        app = FastAPI(title="My Service")
        setup_middleware(app, "my_service")
    """
    # Configure CORS
    configure_cors(app, allow_credentials=allow_cors_credentials)

    # Add logging middleware
    app.add_middleware(LoggingMiddleware)

    # Register exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    app.add_exception_handler(ValueError, validation_exception_handler)

    logger.info(f"Middleware configured for {service_name}")


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("Middleware Module Test")
    print("=" * 60)

    # Test origin parsing
    os.environ["CORS_ORIGINS"] = "http://localhost:3000,https://app.example.com"
    origins = get_allowed_origins()
    print(f"\n1. Parsed CORS origins: {origins}")

    # Test error response creation
    error = ErrorResponse.create(
        error_type="TestError",
        message="This is a test error",
        status_code=400,
        request_id="test-123"
    )
    print(f"\n2. Error response structure: {error}")

    print("\n" + "=" * 60)
    print("Middleware module test completed!")
