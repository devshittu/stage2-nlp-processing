"""
checkpoint_manager.py

Manages batch processing checkpoints for pause/resume functionality.
Enables progressive saving and recovery from failures.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import fcntl
from dataclasses import dataclass, asdict
from enum import Enum


class CheckpointStatus(str, Enum):
    """Checkpoint status states."""
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class BatchCheckpoint:
    """Batch processing checkpoint data."""
    job_id: str
    batch_id: str
    status: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    processed_doc_ids: List[str]
    failed_doc_ids: List[str]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchCheckpoint':
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """
    Manages batch processing checkpoints.

    Features:
    - Create/update checkpoints atomically
    - Track processed and failed documents
    - Support pause/resume/stop operations
    - Thread-safe file locking
    - Quick status checks
    """

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files.
                           If None, uses path from settings.yaml.
        """
        if checkpoint_dir is None:
            # Load from settings if available
            try:
                from src.utils.config_manager import get_settings
                settings = get_settings()
                checkpoint_dir = getattr(
                    settings.storage.data_directories,
                    'checkpoints',
                    '/app/data/checkpoints'
                )
            except Exception:
                checkpoint_dir = '/app/data/checkpoints'

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, job_id: str) -> Path:
        """Get checkpoint file path for job."""
        return self.checkpoint_dir / f"{job_id}.json"

    def _get_lock_path(self, job_id: str) -> Path:
        """Get lock file path for job."""
        return self.checkpoint_dir / f"{job_id}.lock"

    def create_checkpoint(
        self,
        job_id: str,
        batch_id: str,
        total_documents: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BatchCheckpoint:
        """
        Create initial checkpoint for batch job.

        Args:
            job_id: Unique job identifier
            batch_id: Batch identifier
            total_documents: Total number of documents in batch
            metadata: Additional metadata

        Returns:
            Created checkpoint
        """
        now = datetime.utcnow().isoformat() + "Z"

        checkpoint = BatchCheckpoint(
            job_id=job_id,
            batch_id=batch_id,
            status=CheckpointStatus.RUNNING,
            total_documents=total_documents,
            processed_documents=0,
            failed_documents=0,
            processed_doc_ids=[],
            failed_doc_ids=[],
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )

        self._save_checkpoint(checkpoint)
        return checkpoint

    def load_checkpoint(self, job_id: str) -> Optional[BatchCheckpoint]:
        """
        Load checkpoint for job.

        Args:
            job_id: Job identifier

        Returns:
            Checkpoint if exists, None otherwise
        """
        checkpoint_path = self._get_checkpoint_path(job_id)

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                return BatchCheckpoint.from_dict(data)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

    def update_checkpoint(
        self,
        job_id: str,
        processed_doc_id: Optional[str] = None,
        failed_doc_id: Optional[str] = None,
        status: Optional[str] = None,
        metadata_update: Optional[Dict[str, Any]] = None
    ) -> Optional[BatchCheckpoint]:
        """
        Update checkpoint with processed/failed document.

        Args:
            job_id: Job identifier
            processed_doc_id: Successfully processed document ID
            failed_doc_id: Failed document ID
            status: New status
            metadata_update: Metadata updates

        Returns:
            Updated checkpoint
        """
        checkpoint = self.load_checkpoint(job_id)
        if not checkpoint:
            return None

        # Update document tracking
        if processed_doc_id:
            if processed_doc_id not in checkpoint.processed_doc_ids:
                checkpoint.processed_doc_ids.append(processed_doc_id)
                checkpoint.processed_documents += 1

        if failed_doc_id:
            if failed_doc_id not in checkpoint.failed_doc_ids:
                checkpoint.failed_doc_ids.append(failed_doc_id)
                checkpoint.failed_documents += 1

        # Update status
        if status:
            checkpoint.status = status

        # Update metadata
        if metadata_update:
            checkpoint.metadata.update(metadata_update)

        # Update timestamp
        checkpoint.updated_at = datetime.utcnow().isoformat() + "Z"

        self._save_checkpoint(checkpoint)
        return checkpoint

    def _save_checkpoint(self, checkpoint: BatchCheckpoint):
        """
        Save checkpoint to disk atomically with file locking.

        Args:
            checkpoint: Checkpoint to save
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint.job_id)
        lock_path = self._get_lock_path(checkpoint.job_id)
        temp_path = checkpoint_path.with_suffix('.tmp')

        try:
            # Acquire lock
            with open(lock_path, 'w') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

                # Write to temp file
                with open(temp_path, 'w') as f:
                    json.dump(checkpoint.to_dict(), f, indent=2)

                # Atomic rename
                temp_path.replace(checkpoint_path)

        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def get_remaining_documents(
        self,
        job_id: str,
        all_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get documents that haven't been processed yet.

        Args:
            job_id: Job identifier
            all_documents: All documents in batch

        Returns:
            List of unprocessed documents
        """
        checkpoint = self.load_checkpoint(job_id)
        if not checkpoint:
            return all_documents

        processed_ids = set(checkpoint.processed_doc_ids)
        failed_ids = set(checkpoint.failed_doc_ids)
        completed_ids = processed_ids | failed_ids

        return [
            doc for doc in all_documents
            if doc.get("document_id") not in completed_ids
        ]

    def is_paused(self, job_id: str) -> bool:
        """Check if job is paused."""
        checkpoint = self.load_checkpoint(job_id)
        return checkpoint and checkpoint.status == CheckpointStatus.PAUSED

    def is_stopped(self, job_id: str) -> bool:
        """Check if job is stopped."""
        checkpoint = self.load_checkpoint(job_id)
        return checkpoint and checkpoint.status == CheckpointStatus.STOPPED

    def pause(self, job_id: str) -> bool:
        """
        Pause batch processing.

        Args:
            job_id: Job identifier

        Returns:
            True if paused successfully
        """
        checkpoint = self.update_checkpoint(
            job_id,
            status=CheckpointStatus.PAUSED,
            metadata_update={"paused_at": datetime.utcnow().isoformat() + "Z"}
        )
        return checkpoint is not None

    def resume(self, job_id: str) -> bool:
        """
        Resume batch processing.

        Args:
            job_id: Job identifier

        Returns:
            True if resumed successfully
        """
        checkpoint = self.update_checkpoint(
            job_id,
            status=CheckpointStatus.RUNNING,
            metadata_update={"resumed_at": datetime.utcnow().isoformat() + "Z"}
        )
        return checkpoint is not None

    def stop(self, job_id: str) -> bool:
        """
        Stop batch processing (cannot be resumed).

        Args:
            job_id: Job identifier

        Returns:
            True if stopped successfully
        """
        checkpoint = self.update_checkpoint(
            job_id,
            status=CheckpointStatus.STOPPED,
            metadata_update={"stopped_at": datetime.utcnow().isoformat() + "Z"}
        )
        return checkpoint is not None

    def complete(self, job_id: str) -> bool:
        """
        Mark batch as completed.

        Args:
            job_id: Job identifier

        Returns:
            True if marked completed successfully
        """
        checkpoint = self.update_checkpoint(
            job_id,
            status=CheckpointStatus.COMPLETED,
            metadata_update={"completed_at": datetime.utcnow().isoformat() + "Z"}
        )
        return checkpoint is not None

    def delete_checkpoint(self, job_id: str):
        """
        Delete checkpoint file.

        Args:
            job_id: Job identifier
        """
        checkpoint_path = self._get_checkpoint_path(job_id)
        lock_path = self._get_lock_path(job_id)

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        if lock_path.exists():
            lock_path.unlink()

    def get_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processing progress.

        Args:
            job_id: Job identifier

        Returns:
            Progress information
        """
        checkpoint = self.load_checkpoint(job_id)
        if not checkpoint:
            return None

        total = checkpoint.processed_documents + checkpoint.failed_documents
        progress_pct = (total / checkpoint.total_documents * 100) if checkpoint.total_documents > 0 else 0

        return {
            "job_id": checkpoint.job_id,
            "batch_id": checkpoint.batch_id,
            "status": checkpoint.status,
            "total_documents": checkpoint.total_documents,
            "processed": checkpoint.processed_documents,
            "failed": checkpoint.failed_documents,
            "remaining": checkpoint.total_documents - total,
            "progress_percentage": round(progress_pct, 2),
            "created_at": checkpoint.created_at,
            "updated_at": checkpoint.updated_at
        }
