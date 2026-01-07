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
import json
import logging
import os
import signal
from typing import Dict, Any, List, Optional
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
        # 2. Read from shared_volume/stage1/output/ if file path provided
        # 3. Process embedded text if included in event

        # For now, just log
        logger.info(f"document_ready_for_nlp: document_id={document_id}")

        return True

    async def handle_job_completed(self, data: Dict[str, Any]) -> bool:
        """
        Handle batch job completed event from Stage 1.

        Event data structure (from Stage 1):
        {
            "job_id": "uuid",
            "batch_id": "batch_2026-01-07",
            "documents_processed": 100,
            "documents_failed": 0,
            "documents_total": 100,
            "processing_time_ms": 120000,
            "output_files": ["/app/data/output/processed_2026-01-07_10-30-45.jsonl"]
        }

        Args:
            data: Event data with job details

        Returns:
            True if processing successful
        """
        job_id = data.get("job_id")
        batch_id = data.get("batch_id")
        documents_processed = data.get("documents_processed", 0)
        output_files = data.get("output_files", [])

        logger.info(
            f"job_completed_event_received: job_id={job_id}, "
            f"batch_id={batch_id}, documents={documents_processed}, "
            f"output_files={output_files}"
        )

        if documents_processed == 0:
            logger.warning(f"job_has_no_documents: job_id={job_id}")
            return True

        # Trigger NLP batch processing with output file paths
        try:
            success = await self.trigger_nlp_batch_processing(
                job_id=job_id,
                batch_id=batch_id,
                document_count=documents_processed,
                output_files=output_files
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
        document_count: int,
        output_files: Optional[List[str]] = None
    ) -> bool:
        """
        Trigger NLP batch processing for cleaned documents.

        Strategy (flexible, configuration-driven):
        1. Try reading from shared JSONL files (fastest, if available)
        2. Fallback to PostgreSQL query (Stage 1 database)
        3. Submit to Celery for async processing

        Args:
            job_id: Stage 1 job ID
            batch_id: Batch identifier
            document_count: Number of documents
            output_files: Optional list of output file paths from Stage 1

        Returns:
            True if triggered successfully
        """
        logger.info(
            f"triggering_nlp_batch: job_id={job_id}, batch_id={batch_id}, "
            f"count={document_count}, files={output_files}"
        )

        try:
            # Strategy 1: Read from shared JSONL files (preferred)
            documents = await self._read_documents_from_files(output_files, job_id, batch_id)

            if not documents:
                # Strategy 2: Fallback to Stage 1 PostgreSQL
                documents = await self._read_documents_from_stage1_db(job_id, batch_id)

            if not documents:
                logger.warning(
                    f"no_documents_found_for_nlp: job_id={job_id}, batch_id={batch_id}"
                )
                return False

            logger.info(
                f"loaded_documents_for_nlp: count={len(documents)}, "
                f"job_id={job_id}, batch_id={batch_id}"
            )

            # Strategy 3: Submit to Celery for async processing
            success = await self._submit_to_celery(documents, job_id, batch_id)

            if success:
                logger.info(
                    f"nlp_batch_processing_queued: job_id={job_id}, "
                    f"stage1_job_id={job_id}, documents={len(documents)}"
                )
            else:
                logger.error(
                    f"failed_to_queue_nlp_batch: job_id={job_id}"
                )

            return success

        except Exception as e:
            logger.error(
                f"trigger_nlp_batch_error: job_id={job_id}, error={e}",
                exc_info=True
            )
            return False

    async def _read_documents_from_files(
        self,
        output_files: Optional[List[str]],
        job_id: str,
        batch_id: str
    ) -> List[Dict[str, Any]]:
        """
        Read cleaned documents from Stage 1 output JSONL files.

        Supports multiple file locations:
        - Explicit paths from event data
        - Shared directory discovery
        - Volume mount paths
        """
        documents = []

        # Get shared directory from environment
        shared_dir = os.getenv("SHARED_STAGE1_DIR", "/shared/stage1")

        # File paths to try
        file_paths = []

        # 1. Use explicit output_files from event
        if output_files:
            file_paths.extend(output_files)

        # 2. Try common naming patterns in shared directory
        if os.path.isdir(shared_dir):
            import glob
            # Match Stage 1 output naming convention: processed_YYYY-MM-DD_HH-MM-SS*.jsonl
            # SHARED_STAGE1_DIR points directly to output directory
            patterns = [
                f"{shared_dir}/processed_*.jsonl",
                f"{shared_dir}/*.jsonl",
            ]
            for pattern in patterns:
                matches = glob.glob(pattern)
                # Sort by modification time (newest first)
                matches.sort(key=os.path.getmtime, reverse=True)
                file_paths.extend(matches[:5])  # Take top 5 newest

        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in file_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)

        # Read documents from files
        for file_path in unique_paths:
            try:
                if os.path.exists(file_path):
                    count = 0
                    with open(file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                doc = json.loads(line)
                                # Filter by job_id if available
                                if doc.get("job_id") == job_id or not doc.get("job_id"):
                                    documents.append(doc)
                                    count += 1
                    if count > 0:
                        logger.info(
                            f"read_documents_from_file: path={file_path}, count={count}"
                        )
            except Exception as e:
                logger.warning(f"failed_to_read_file: path={file_path}, error={e}")

        return documents

    async def _read_documents_from_stage1_db(
        self,
        job_id: str,
        batch_id: str
    ) -> List[Dict[str, Any]]:
        """
        Read cleaned documents from Stage 1 PostgreSQL database.

        Fallback when shared files are not available.
        """
        documents = []

        # Stage 1 database connection parameters
        stage1_host = os.getenv("STAGE1_POSTGRES_HOST", "postgres")
        stage1_port = int(os.getenv("STAGE1_POSTGRES_PORT", "5432"))
        stage1_db = os.getenv("STAGE1_POSTGRES_DB", "stage1_cleaning")
        stage1_user = os.getenv("STAGE1_POSTGRES_USER", "stage1_user")
        stage1_password = os.getenv("STAGE1_POSTGRES_PASSWORD", "")

        if not stage1_password:
            logger.warning("stage1_postgres_password_not_configured")
            return documents

        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            conn = psycopg2.connect(
                host=stage1_host,
                port=stage1_port,
                database=stage1_db,
                user=stage1_user,
                password=stage1_password,
                connect_timeout=10
            )

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Query processed articles by job_id or batch_id
                query = """
                    SELECT *
                    FROM processed_articles
                    WHERE job_id = %s OR batch_id = %s
                    ORDER BY processed_at DESC
                    LIMIT 10000
                """
                cur.execute(query, (job_id, batch_id))
                rows = cur.fetchall()

                for row in rows:
                    documents.append(dict(row))

            conn.close()

            logger.info(
                f"read_documents_from_stage1_db: count={len(documents)}, "
                f"job_id={job_id}, batch_id={batch_id}"
            )

        except ImportError:
            logger.warning("psycopg2_not_installed_cannot_read_stage1_db")
        except Exception as e:
            logger.error(f"stage1_db_read_error: {e}")

        return documents

    async def _submit_to_celery(
        self,
        documents: List[Dict[str, Any]],
        job_id: str,
        batch_id: str
    ) -> bool:
        """
        Submit documents to Stage 2 Celery for batch processing.

        Uses the existing process_batch_task from celery_tasks.py.
        """
        try:
            # Import Celery task
            from src.core.celery_tasks import process_batch_task

            # Generate Stage 2 batch ID (linked to Stage 1)
            stage2_batch_id = f"s2_{batch_id}" if batch_id else f"s2_from_{job_id[:8]}"

            # Submit to Celery (async)
            task = process_batch_task.delay(
                documents=documents,
                batch_id=stage2_batch_id,
                options={
                    "stage1_job_id": job_id,
                    "triggered_by": "event_consumer"
                }
            )

            logger.info(
                f"celery_task_submitted: task_id={task.id}, "
                f"batch_id={stage2_batch_id}, documents={len(documents)}"
            )

            return True

        except ImportError:
            logger.error("celery_tasks_not_available")
            return False
        except Exception as e:
            logger.error(f"celery_submission_error: {e}", exc_info=True)
            return False

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
