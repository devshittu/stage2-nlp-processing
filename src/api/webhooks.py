"""
src/api/webhooks.py

CloudEvents v1.0 Webhook Receiver for Stage 1 → Stage 2 Integration.

Implements 2026 REST API standards for webhook endpoints:
- CloudEvents Binary Content Mode (ce-* headers)
- CloudEvents Structured Content Mode (application/cloudevents+json)
- Idempotency via ce-id header
- Rate limiting via middleware
- HMAC signature verification (optional)

Endpoint: POST /api/v1/webhooks/stage1/events
"""

import json
import logging
import hashlib
import hmac
import os
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, Request, HTTPException, Header, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("stage2_nlp")


# =============================================================================
# Pydantic Models (2026 CloudEvents v1.0 Spec)
# =============================================================================

class CloudEventPayload(BaseModel):
    """CloudEvents v1.0 structured content mode payload."""
    specversion: str = Field(default="1.0", description="CloudEvents spec version")
    type: str = Field(..., description="Event type (reverse-DNS format)")
    source: str = Field(..., description="Event source URI")
    id: str = Field(..., description="Unique event identifier")
    time: Optional[str] = Field(None, description="Event timestamp (ISO 8601)")
    subject: Optional[str] = Field(None, description="Event subject")
    datacontenttype: Optional[str] = Field("application/json", description="Data content type")
    data: Optional[Dict[str, Any]] = Field(None, description="Event payload")


class WebhookResponse(BaseModel):
    """Standard webhook response (2026 API design)."""
    accepted: bool = Field(..., description="Whether event was accepted for processing")
    event_id: str = Field(..., description="CloudEvent ID")
    message: str = Field(..., description="Human-readable status message")
    processing_status: str = Field(..., description="pending|queued|duplicate|rejected")
    timestamp: str = Field(..., description="Response timestamp (ISO 8601)")


# =============================================================================
# Webhook Router
# =============================================================================

router = APIRouter(
    prefix="/api/v1/webhooks",
    tags=["webhooks"],
    responses={
        202: {"description": "Event accepted for processing"},
        400: {"description": "Invalid CloudEvents format"},
        401: {"description": "Invalid signature"},
        409: {"description": "Duplicate event (idempotency)"},
        429: {"description": "Rate limit exceeded"},
    }
)

# In-memory deduplication cache (use Redis in production for distributed systems)
_processed_event_ids: Dict[str, datetime] = {}
_DEDUP_TTL_HOURS = 24


def _check_duplicate(event_id: str) -> bool:
    """Check if event was already processed (idempotency)."""
    if event_id in _processed_event_ids:
        return True
    return False


def _mark_processed(event_id: str):
    """Mark event as processed for deduplication."""
    _processed_event_ids[event_id] = datetime.utcnow()
    # Cleanup old entries (simple TTL cleanup)
    cutoff = datetime.utcnow()
    to_remove = [
        eid for eid, ts in _processed_event_ids.items()
        if (cutoff - ts).total_seconds() > _DEDUP_TTL_HOURS * 3600
    ]
    for eid in to_remove:
        del _processed_event_ids[eid]


def _verify_hmac_signature(
    payload: bytes,
    signature: Optional[str],
    secret: Optional[str]
) -> bool:
    """
    Verify HMAC-SHA256 signature for webhook security.

    Args:
        payload: Raw request body
        signature: X-Signature-256 header value
        secret: Shared secret key

    Returns:
        True if signature is valid or verification is disabled
    """
    if not secret:
        # Signature verification disabled
        return True

    if not signature:
        return False

    # Calculate expected signature
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    # Compare with provided signature (constant-time comparison)
    return hmac.compare_digest(f"sha256={expected}", signature)


async def _trigger_nlp_processing(event_data: Dict[str, Any], event_type: str):
    """
    Trigger NLP processing based on event type.

    This is called as a background task to avoid blocking the webhook response.
    """
    try:
        # Import here to avoid circular imports
        from src.services.event_consumer_service import EventConsumerService

        service = EventConsumerService()

        if event_type == "com.storytelling.cleaning.job.completed":
            success = await service.handle_job_completed(event_data)
            if success:
                logger.info(
                    f"webhook_triggered_job_processing: job_id={event_data.get('job_id')}"
                )
            else:
                logger.error(
                    f"webhook_job_processing_failed: job_id={event_data.get('job_id')}"
                )

        elif event_type == "com.storytelling.cleaning.document.cleaned":
            success = await service.handle_document_cleaned(event_data)
            if success:
                logger.info(
                    f"webhook_triggered_document_processing: doc_id={event_data.get('document_id')}"
                )

        else:
            logger.info(f"webhook_event_received_no_handler: type={event_type}")

    except Exception as e:
        logger.error(f"webhook_processing_error: {e}", exc_info=True)


@router.post(
    "/stage1/events",
    response_model=WebhookResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Receive Stage 1 CloudEvents",
    description="""
    Webhook endpoint for receiving CloudEvents from Stage 1 Data Cleaning Service.

    **Supported Event Types:**
    - `com.storytelling.cleaning.job.completed` - Batch job completion
    - `com.storytelling.cleaning.job.started` - Batch job started
    - `com.storytelling.cleaning.job.progress` - Batch job progress update
    - `com.storytelling.cleaning.document.cleaned` - Individual document cleaned

    **Content Modes:**
    - Binary: CloudEvents attributes in HTTP headers (ce-*)
    - Structured: Full CloudEvent JSON with `application/cloudevents+json`

    **Security:**
    - Optional HMAC-SHA256 signature verification via X-Signature-256 header
    - Idempotency enforced via ce-id header (duplicates return 409)
    """
)
async def receive_stage1_event(
    request: Request,
    background_tasks: BackgroundTasks,
    # CloudEvents Binary Mode Headers
    ce_specversion: Optional[str] = Header(None, alias="ce-specversion"),
    ce_type: Optional[str] = Header(None, alias="ce-type"),
    ce_source: Optional[str] = Header(None, alias="ce-source"),
    ce_id: Optional[str] = Header(None, alias="ce-id"),
    ce_time: Optional[str] = Header(None, alias="ce-time"),
    ce_subject: Optional[str] = Header(None, alias="ce-subject"),
    # Security Headers
    x_signature_256: Optional[str] = Header(None, alias="X-Signature-256"),
    # Pipeline Headers
    x_stage_source: Optional[str] = Header(None, alias="X-Stage-Source"),
):
    """
    Receive CloudEvents from Stage 1 cleaning service.

    Supports both Binary and Structured content modes per CloudEvents v1.0 spec.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Read raw body
    body = await request.body()
    content_type = request.headers.get("content-type", "")

    # Verify HMAC signature (if configured)
    webhook_secret = os.getenv("WEBHOOK_SECRET")
    if webhook_secret and not _verify_hmac_signature(body, x_signature_256, webhook_secret):
        logger.warning(
            "webhook_invalid_signature",
            extra={"source": x_stage_source}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature"
        )

    # Parse CloudEvent based on content mode
    event_id: str
    event_type: str
    event_source: str
    event_data: Dict[str, Any]

    if "application/cloudevents+json" in content_type:
        # Structured Content Mode - full CloudEvent in body
        try:
            cloud_event = CloudEventPayload(**json.loads(body))
            event_id = cloud_event.id
            event_type = cloud_event.type
            event_source = cloud_event.source
            event_data = cloud_event.data or {}
        except Exception as e:
            logger.error(f"webhook_invalid_cloudevents_json: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid CloudEvents structured format: {str(e)}"
            )

    elif ce_type and ce_source and ce_id:
        # Binary Content Mode - attributes in headers, data in body
        event_id = ce_id
        event_type = ce_type
        event_source = ce_source

        try:
            event_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            event_data = {}

    else:
        # Missing required CloudEvents attributes
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required CloudEvents attributes. Provide either "
                   "ce-type/ce-source/ce-id headers (binary mode) or "
                   "application/cloudevents+json content type (structured mode)"
        )

    # Idempotency check
    if _check_duplicate(event_id):
        logger.info(
            f"webhook_duplicate_event: event_id={event_id}, type={event_type}"
        )
        return WebhookResponse(
            accepted=True,
            event_id=event_id,
            message="Event already processed (idempotent)",
            processing_status="duplicate",
            timestamp=timestamp
        )

    # Mark as processed
    _mark_processed(event_id)

    # Log received event
    logger.info(
        f"webhook_event_received: type={event_type}, id={event_id}, "
        f"source={event_source}, stage_source={x_stage_source}"
    )

    # Trigger processing in background (non-blocking)
    background_tasks.add_task(_trigger_nlp_processing, event_data, event_type)

    # Return 202 Accepted immediately
    return WebhookResponse(
        accepted=True,
        event_id=event_id,
        message=f"Event {event_type} accepted for processing",
        processing_status="queued",
        timestamp=timestamp
    )


@router.get(
    "/stage1/events/health",
    summary="Webhook Health Check",
    description="Health check endpoint for webhook receiver"
)
async def webhook_health():
    """Health check for webhook endpoint."""
    return {
        "status": "healthy",
        "endpoint": "/api/v1/webhooks/stage1/events",
        "supported_events": [
            "com.storytelling.cleaning.job.completed",
            "com.storytelling.cleaning.job.started",
            "com.storytelling.cleaning.job.progress",
            "com.storytelling.cleaning.document.cleaned",
        ],
        "content_modes": ["binary", "structured"],
        "dedup_ttl_hours": _DEDUP_TTL_HOURS,
        "events_in_dedup_cache": len(_processed_event_ids),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
