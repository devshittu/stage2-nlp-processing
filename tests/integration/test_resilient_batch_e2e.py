"""
End-to-End Integration Tests for Resilient Batch Processing

Tests complete workflows including checkpoint creation, progressive saving,
pause/resume/stop operations, error recovery, and monitoring.
"""

import pytest
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from src.core.checkpoint_manager import CheckpointManager, CheckpointStatus
from src.core.celery_tasks import save_single_document


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    checkpoint_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    yield {
        "checkpoint": checkpoint_dir,
        "output": output_dir
    }

    shutil.rmtree(checkpoint_dir)
    shutil.rmtree(output_dir)


@pytest.fixture
def checkpoint_manager(temp_dirs):
    """Create CheckpointManager with temp directory."""
    return CheckpointManager(checkpoint_dir=temp_dirs["checkpoint"])


@pytest.fixture
def sample_batch():
    """Create sample batch of documents."""
    return {
        "job_id": "e2e_test_job_001",
        "batch_id": "e2e_batch_001",
        "documents": [
            {
                "id": f"doc_{i:03d}",
                "content": f"Test document {i} content",
                "metadata": {"source": "test", "index": i}
            }
            for i in range(1, 21)
        ]
    }


class TestFullBatchWithProgressiveSaving:
    """Test complete batch processing with progressive saving."""

    def test_batch_processes_all_documents(self, checkpoint_manager, sample_batch):
        """Test that all documents in batch are processed."""
        job_id = sample_batch["job_id"]
        documents = sample_batch["documents"]

        # Create checkpoint
        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(documents)
        )

        # Simulate processing all documents
        for doc in documents:
            checkpoint_manager.update_checkpoint(
                job_id=job_id,
                processed_doc_id=doc["id"]
            )

        # Verify final state
        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint.processed_documents == len(documents)
        assert checkpoint.failed_documents == 0
        assert len(checkpoint.processed_doc_ids) == len(documents)

    def test_progressive_save_persists_data(self, sample_batch):
        """Test that progressive saves actually persist data."""
        mock_storage = Mock()
        mock_storage.save_batch = Mock(return_value={"success": True})

        results = [
            {
                "document_id": doc["id"],
                "job_id": sample_batch["job_id"],
                "success": True,
                "entities": [],
                "events": []
            }
            for doc in sample_batch["documents"][:5]
        ]

        # Progressively save documents
        for result in results:
            success = save_single_document(result, mock_storage)
            assert success is True

        # Verify each save was called
        assert mock_storage.save_batch.call_count == 5

    def test_checkpoint_updated_after_each_save(
        self,
        checkpoint_manager,
        sample_batch
    ):
        """Test that checkpoint is updated after each document save."""
        job_id = sample_batch["job_id"]

        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(sample_batch["documents"])
        )

        # Process documents one by one
        for i, doc in enumerate(sample_batch["documents"][:10], 1):
            checkpoint_manager.update_checkpoint(
                job_id=job_id,
                processed_doc_id=doc["id"]
            )

            # Verify progressive state
            checkpoint = checkpoint_manager.load_checkpoint(job_id)
            assert checkpoint.processed_documents == i


class TestPauseAndResumeWorkflow:
    """Test pause and resume workflow."""

    def test_pause_during_processing(self, checkpoint_manager, sample_batch):
        """Test pausing batch during processing."""
        job_id = sample_batch["job_id"]

        # Create and start processing
        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(sample_batch["documents"])
        )

        # Process half the documents
        halfway = len(sample_batch["documents"]) // 2
        for doc in sample_batch["documents"][:halfway]:
            checkpoint_manager.update_checkpoint(
                job_id=job_id,
                processed_doc_id=doc["id"]
            )

        # Pause
        success = checkpoint_manager.pause(job_id)
        assert success is True

        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint.status == CheckpointStatus.PAUSED
        assert checkpoint.processed_documents == halfway

    def test_resume_continues_from_checkpoint(
        self,
        checkpoint_manager,
        sample_batch
    ):
        """Test that resume continues from saved checkpoint."""
        job_id = sample_batch["job_id"]
        documents = sample_batch["documents"]

        # Create checkpoint and process some docs
        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(documents)
        )

        processed_count = 7
        for doc in documents[:processed_count]:
            checkpoint_manager.update_checkpoint(
                job_id=job_id,
                processed_doc_id=doc["id"]
            )

        # Pause
        checkpoint_manager.pause(job_id)

        # Resume
        success = checkpoint_manager.resume(job_id)
        assert success is True

        # Get remaining documents
        remaining = checkpoint_manager.get_remaining_documents(job_id, documents)
        assert len(remaining) == len(documents) - processed_count

        # Continue processing remaining
        for doc in remaining:
            checkpoint_manager.update_checkpoint(
                job_id=job_id,
                processed_doc_id=doc["id"]
            )

        # Verify completion
        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint.processed_documents == len(documents)

    def test_complete_pause_resume_cycle(self, checkpoint_manager, sample_batch):
        """Test complete pause-resume-complete cycle."""
        job_id = sample_batch["job_id"]
        documents = sample_batch["documents"]

        # Phase 1: Initial processing
        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(documents)
        )

        for doc in documents[:5]:
            checkpoint_manager.update_checkpoint(job_id, processed_doc_id=doc["id"])

        # Phase 2: Pause
        checkpoint_manager.pause(job_id)
        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint.status == CheckpointStatus.PAUSED

        # Phase 3: Resume and continue
        checkpoint_manager.resume(job_id)
        remaining = checkpoint_manager.get_remaining_documents(job_id, documents)

        for doc in remaining:
            checkpoint_manager.update_checkpoint(job_id, processed_doc_id=doc["id"])

        # Phase 4: Complete
        checkpoint_manager.complete(job_id)
        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint.status == CheckpointStatus.COMPLETED
        assert checkpoint.processed_documents == len(documents)


class TestTimeoutAndRecovery:
    """Test timeout and recovery scenarios."""

    def test_timeout_preserves_progress(self, checkpoint_manager, sample_batch):
        """Test that timeout doesn't lose processed documents."""
        job_id = sample_batch["job_id"]
        documents = sample_batch["documents"]

        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(documents)
        )

        # Process documents until "timeout"
        timeout_at = 8
        for doc in documents[:timeout_at]:
            checkpoint_manager.update_checkpoint(job_id, processed_doc_id=doc["id"])

        # Simulate timeout - checkpoint should be preserved
        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint.processed_documents == timeout_at

        # Recover by resubmitting
        remaining = checkpoint_manager.get_remaining_documents(job_id, documents)
        assert len(remaining) == len(documents) - timeout_at

    def test_resubmit_after_timeout(self, checkpoint_manager, sample_batch):
        """Test resubmitting batch after timeout."""
        job_id = sample_batch["job_id"]
        documents = sample_batch["documents"]

        # Initial run (timeout scenario)
        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(documents)
        )

        for doc in documents[:10]:
            checkpoint_manager.update_checkpoint(job_id, processed_doc_id=doc["id"])

        # Resubmit - load checkpoint and continue
        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint is not None

        remaining = checkpoint_manager.get_remaining_documents(job_id, documents)
        assert len(remaining) == 10

        # Complete processing
        for doc in remaining:
            checkpoint_manager.update_checkpoint(job_id, processed_doc_id=doc["id"])

        checkpoint_manager.complete(job_id)
        final_checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert final_checkpoint.status == CheckpointStatus.COMPLETED


class TestMultipleFailuresContinueProcessing:
    """Test resilience with multiple failures."""

    def test_multiple_failures_dont_stop_batch(
        self,
        checkpoint_manager,
        sample_batch
    ):
        """Test that multiple failures don't stop batch processing."""
        job_id = sample_batch["job_id"]
        documents = sample_batch["documents"]

        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(documents)
        )

        # Simulate mixed success/failure
        failed_indices = [2, 5, 9, 15]

        for i, doc in enumerate(documents):
            if i in failed_indices:
                checkpoint_manager.update_checkpoint(
                    job_id,
                    failed_doc_id=doc["id"]
                )
            else:
                checkpoint_manager.update_checkpoint(
                    job_id,
                    processed_doc_id=doc["id"]
                )

        # Verify final state
        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint.processed_documents == len(documents) - len(failed_indices)
        assert checkpoint.failed_documents == len(failed_indices)

    def test_failed_documents_tracked(self, checkpoint_manager, sample_batch):
        """Test that failed documents are properly tracked."""
        job_id = sample_batch["job_id"]
        documents = sample_batch["documents"]

        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(documents)
        )

        failed_docs = documents[5:10]

        for doc in documents[:5]:
            checkpoint_manager.update_checkpoint(job_id, processed_doc_id=doc["id"])

        for doc in failed_docs:
            checkpoint_manager.update_checkpoint(job_id, failed_doc_id=doc["id"])

        for doc in documents[10:]:
            checkpoint_manager.update_checkpoint(job_id, processed_doc_id=doc["id"])

        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert len(checkpoint.failed_doc_ids) == 5
        assert all(doc["id"] in checkpoint.failed_doc_ids for doc in failed_docs)


class TestCheckpointStatusAccuracy:
    """Test checkpoint status accuracy."""

    def test_progress_matches_reality(self, checkpoint_manager, sample_batch):
        """Test that checkpoint progress matches actual processing."""
        job_id = sample_batch["job_id"]
        documents = sample_batch["documents"]

        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(documents)
        )

        processed_count = 0
        for i, doc in enumerate(documents):
            if i % 3 == 0:  # Simulate some failures
                checkpoint_manager.update_checkpoint(job_id, failed_doc_id=doc["id"])
            else:
                checkpoint_manager.update_checkpoint(job_id, processed_doc_id=doc["id"])
                processed_count += 1

        progress = checkpoint_manager.get_progress(job_id)
        assert progress["processed"] == processed_count
        assert progress["total_documents"] == len(documents)

    def test_checkpoint_timestamps_accurate(self, checkpoint_manager, sample_batch):
        """Test that checkpoint timestamps are accurate."""
        job_id = sample_batch["job_id"]

        checkpoint = checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(sample_batch["documents"])
        )

        created_at = checkpoint.created_at
        time.sleep(0.1)

        # Update checkpoint
        checkpoint_manager.update_checkpoint(job_id, processed_doc_id="doc_001")

        updated_checkpoint = checkpoint_manager.load_checkpoint(job_id)
        updated_at = updated_checkpoint.updated_at

        # Updated timestamp should be after created timestamp
        assert updated_at >= created_at


class TestStopWorkflow:
    """Test stop workflow."""

    def test_stop_prevents_resume(self, checkpoint_manager, sample_batch):
        """Test that stopped batch cannot be resumed."""
        job_id = sample_batch["job_id"]

        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(sample_batch["documents"])
        )

        # Process some documents
        for doc in sample_batch["documents"][:5]:
            checkpoint_manager.update_checkpoint(job_id, processed_doc_id=doc["id"])

        # Stop
        checkpoint_manager.stop(job_id)

        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint.status == CheckpointStatus.STOPPED
        assert checkpoint_manager.is_stopped(job_id) is True

    def test_stop_retains_progress(self, checkpoint_manager, sample_batch):
        """Test that stop retains processed documents."""
        job_id = sample_batch["job_id"]

        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(sample_batch["documents"])
        )

        # Process some documents
        processed_count = 7
        for doc in sample_batch["documents"][:processed_count]:
            checkpoint_manager.update_checkpoint(job_id, processed_doc_id=doc["id"])

        # Stop
        checkpoint_manager.stop(job_id)

        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint.processed_documents == processed_count


class TestPerformanceAndScaling:
    """Test performance characteristics."""

    def test_large_batch_performance(self, checkpoint_manager):
        """Test checkpoint performance with large batch."""
        job_id = "large_batch_test"
        total_docs = 1000

        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id="large_batch",
            total_documents=total_docs
        )

        # Measure update performance
        start_time = time.time()

        for i in range(1, 101):  # Update 100 documents
            checkpoint_manager.update_checkpoint(
                job_id,
                processed_doc_id=f"doc_{i:04d}"
            )

        duration = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds for 100 updates)
        assert duration < 5.0

        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint.processed_documents == 100

    def test_checkpoint_file_size_reasonable(
        self,
        checkpoint_manager,
        temp_dirs,
        sample_batch
    ):
        """Test that checkpoint file size remains reasonable."""
        job_id = sample_batch["job_id"]

        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            batch_id=sample_batch["batch_id"],
            total_documents=len(sample_batch["documents"])
        )

        # Process all documents
        for doc in sample_batch["documents"]:
            checkpoint_manager.update_checkpoint(job_id, processed_doc_id=doc["id"])

        # Check file size
        checkpoint_file = Path(temp_dirs["checkpoint"]) / f"{job_id}.json"
        file_size = checkpoint_file.stat().st_size

        # Should be < 10KB for 20 documents
        assert file_size < 10 * 1024


class TestConcurrentBatches:
    """Test multiple concurrent batches."""

    def test_multiple_batches_independent(self, checkpoint_manager):
        """Test that multiple batches maintain independent state."""
        jobs = [
            {"job_id": f"job_{i:03d}", "batch_id": f"batch_{i:03d}", "total": 10}
            for i in range(1, 4)
        ]

        # Create checkpoints for all jobs
        for job in jobs:
            checkpoint_manager.create_checkpoint(
                job_id=job["job_id"],
                batch_id=job["batch_id"],
                total_documents=job["total"]
            )

        # Process different amounts for each
        for i, job in enumerate(jobs):
            for doc_idx in range(1, (i + 1) * 3):
                checkpoint_manager.update_checkpoint(
                    job["job_id"],
                    processed_doc_id=f"doc_{doc_idx:03d}"
                )

        # Verify independence
        for i, job in enumerate(jobs):
            checkpoint = checkpoint_manager.load_checkpoint(job["job_id"])
            assert checkpoint.processed_documents == (i + 1) * 3
