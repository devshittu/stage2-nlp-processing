"""
src/services/event_consumer_service.py

Background service for consuming Stage 1 events and triggering NLP processing.

Responsibilities:
- Run event consumer in background
- Handle consumed events
- Trigger NLP processing for cleaned documents
- Manage service lifecycle (startup, shutdown)
- Health monitoring

Usage:
    python -m src.services.event_consumer_service

Environment Variables:
    EVENT_CONSUMER_ENABLED - Enable/disable consumer (default: false)
    STAGE1_EVENT_STREAM - Source stream name (default: stage1:cleaning:events)
    AUTO_PROCESS - Auto-trigger processing (default: true)
"""

import asyncio
import logging
import os
import signal
from typing import Dict, Any, Optional
from datetime import datetime

from src.events.consumer import get_event_consumer, CloudEventConsumer
from src.utils.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("event_consumer_service")


class EventConsumerService:
    """
    Background service for event consumption.

    Features:
    - Consumes Stage 1 CloudEvents
    - Triggers NLP processing
    - Graceful shutdown
    - Health monitoring
    """

    def __init__(self):
        """Initialize service."""
        self.consumer: Optional[CloudEventConsumer] = None
        self.running = False
        self._shutdown_event = asyncio.Event()

        # Load configuration
        try:
            self.settings = ConfigManager.get_settings()
        except Exception as e:
            logger.error(f"failed_to_load_settings: {e}")
            self.settings = None

    async def initialize(self) -> bool:
        """
        Initialize event consumer.

        Returns:
            True if successful
        """
        logger.info("initializing_event_consumer_service")

        # Get consumer instance
        self.consumer = get_event_consumer()

        # Initialize consumer
        success = await self.consumer.initialize()

        if not success:
            logger.error("failed_to_initialize_consumer")
            return False

        # Set event handler
        self.consumer.set_event_handler(self.handle_event)

        logger.info("event_consumer_service_initialized")
        return True

    async def handle_event(self, event: Dict[str, Any]) -> bool:
        """
        Handle consumed event from Stage 1.

        Args:
            event: CloudEvent dictionary

        Returns:
            True if processing successful
        """
        event_type = event.get("type")
        event_id = event.get("id")
        event_data = event.get("data", {})

        logger.info(
            f"handling_event: type={event_type}, id={event_id}"
        )

        try:
            # Route based on event type
            if event_type == "com.storytelling.cleaning.document.cleaned":
                return await self.handle_document_cleaned(event_data)

            elif event_type == "com.storytelling.cleaning.job.completed":
                return await self.handle_job_completed(event_data)

            else:
                logger.warning(f"unknown_event_type: {event_type}")
                return True  # Not an error, just unknown

        except Exception as e:
            logger.error(
                f"event_handling_error: event_id={event_id}, error={e}",
                exc_info=True
            )
            return False

    async def handle_document_cleaned(self, data: Dict[str, Any]) -> bool:
        """
        Handle individual document cleaned event.

        Args:
            data: Event data

        Returns:
            True if processing successful
        """
        document_id = data.get("document_id")

        logger.info(f"document_cleaned_event_received: document_id={document_id}")

        # TODO: Implement individual document processing
        # Options:
        # 1. Read document from PostgreSQL
        # 2. Read from /shared/stage1/ if file path provided
        # 3. Process embedded text if included in event

        # For now, just log
        logger.info(f"document_ready_for_nlp: document_id={document_id}")

        return True

    async def handle_job_completed(self, data: Dict[str, Any]) -> bool:
        """
        Handle batch job completed event.

        Args:
            data: Event data with job details

        Returns:
            True if processing successful
        """
        job_id = data.get("job_id")
        batch_id = data.get("batch_id")
        documents_processed = data.get("documents_processed", 0)

        logger.info(
            f"job_completed_event_received: job_id={job_id}, "
            f"batch_id={batch_id}, documents={documents_processed}"
        )

        if documents_processed == 0:
            logger.warning(f"job_has_no_documents: job_id={job_id}")
            return True

        # Trigger NLP batch processing
        try:
            success = await self.trigger_nlp_batch_processing(
                job_id=job_id,
                batch_id=batch_id,
                document_count=documents_processed
            )

            if success:
                logger.info(
                    f"nlp_batch_processing_triggered: job_id={job_id}, "
                    f"batch_id={batch_id}"
                )
            else:
                logger.error(
                    f"failed_to_trigger_nlp_processing: job_id={job_id}"
                )

            return success

        except Exception as e:
            logger.error(
                f"nlp_processing_trigger_error: job_id={job_id}, error={e}",
                exc_info=True
            )
            return False

    async def trigger_nlp_batch_processing(
        self,
        job_id: str,
        batch_id: str,
        document_count: int
    ) -> bool:
        """
        Trigger NLP batch processing for cleaned documents.

        Args:
            job_id: Stage 1 job ID
            batch_id: Batch identifier
            document_count: Number of documents

        Returns:
            True if triggered successfully
        """
        logger.info(
            f"triggering_nlp_batch: job_id={job_id}, batch_id={batch_id}, "
            f"count={document_count}"
        )

        # TODO: Implement actual trigger logic
        # Options:
        # 1. Call internal API endpoint
        # 2. Submit Celery task directly
        # 3. Read from /shared/stage1/ and process

        # For now, simulate processing
        await asyncio.sleep(0.1)  # Simulate work

        logger.info(
            f"nlp_batch_processing_queued: job_id={job_id}, "
            f"stage1_job_id={job_id}, documents={document_count}"
        )

        return True

    async def run(self):
        """
        Main service loop.

        Runs consumer and handles graceful shutdown.
        """
        if not self.consumer or not self.consumer.enabled:
            logger.error("consumer_not_enabled_exiting")
            return

        self.running = True
        logger.info("event_consumer_service_starting")

        # Start consumer
        await self.consumer.start()

        # Log startup info
        logger.info(
            f"event_consumer_service_running: "
            f"stream={self.consumer.source_stream}, "
            f"group={self.consumer.consumer_group}, "
            f"consumer={self.consumer.consumer_name}, "
            f"auto_process={self.consumer.auto_process}"
        )

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        logger.info("event_consumer_service_shutting_down")

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("shutdown_signal_received")

        self.running = False
        self._shutdown_event.set()

        # Stop consumer
        if self.consumer:
            await self.consumer.stop()
            await self.consumer.close()

        logger.info("event_consumer_service_stopped")

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check.

        Returns:
            Health status dictionary
        """
        if not self.consumer:
            return {
                "healthy": False,
                "reason": "consumer_not_initialized"
            }

        consumer_health = await self.consumer.health_check()

        return {
            "healthy": consumer_health.get("healthy", False),
            "service_running": self.running,
            "consumer": consumer_health
        }


# Global service instance
service: Optional[EventConsumerService] = None


def handle_shutdown_signal(sig, frame):
    """Handle shutdown signals (SIGTERM, SIGINT)."""
    logger.info(f"received_signal: {signal.Signals(sig).name}")
    if service:
        asyncio.create_task(service.shutdown())


async def main():
    """
    Main entry point.

    Runs event consumer service.
    """
    global service

    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)

    # Create service
    service = EventConsumerService()

    # Initialize
    success = await service.initialize()

    if not success:
        logger.error("service_initialization_failed_exiting")
        return 1

    # Run service
    try:
        await service.run()
        return 0
    except Exception as e:
        logger.error(f"service_error: {e}", exc_info=True)
        return 1
    finally:
        await service.shutdown()


if __name__ == "__main__":
    """
    Run as standalone service.

    Usage:
        python -m src.services.event_consumer_service
    """
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("service_interrupted_by_user")
        exit(0)
