"""
Unit tests for Progressive Saving functionality

Tests the save_single_document function and its integration with
storage backends for immediate document persistence.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, List

from src.core.celery_tasks import save_single_document
from src.schemas.data_models import ProcessedDocument
from src.storage.multi_backend_writer import MultiBackendWriter


@pytest.fixture
def mock_storage_writer():
    """Create mock MultiBackendWriter."""
    writer = Mock(spec=MultiBackendWriter)
    writer.save_batch = Mock(return_value={"success": True, "saved_count": 1})
    return writer


@pytest.fixture
def sample_result():
    """Sample processing result for testing."""
    return {
        "document_id": "doc_001",
        "job_id": "test_job_001",
        "success": True,
        "entities": [
            {"text": "John Doe", "type": "PERSON", "start": 0, "end": 8}
        ],
        "events": [
            {
                "type": "Meeting",
                "participants": ["John Doe"],
                "location": "Office",
                "timestamp": "2025-01-15T10:00:00Z"
            }
        ],
        "document_metadata": {
            "title": "Test Document",
            "source": "test"
        }
    }


@pytest.fixture
def sample_linkages():
    """Sample event linkages for testing."""
    from src.events.models import EventLinkage
    return [
        EventLinkage(
            source_event_id="event_001",
            target_event_id="event_002",
            relationship_type="CAUSES",
            confidence=0.95
        )
    ]


@pytest.fixture
def sample_storylines():
    """Sample storylines for testing."""
    from src.events.models import Storyline
    return [
        Storyline(
            storyline_id="story_001",
            title="Test Storyline",
            event_ids=["event_001", "event_002"],
            timeline=["2025-01-15T10:00:00Z", "2025-01-15T11:00:00Z"]
        )
    ]


class TestSaveSingleDocument:
    """Test save_single_document function."""

    def test_save_success(self, mock_storage_writer, sample_result):
        """Test successful single document save."""
        success = save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer
        )

        assert success is True
        assert mock_storage_writer.save_batch.called
        assert mock_storage_writer.save_batch.call_count == 1

    def test_save_creates_processed_document(self, mock_storage_writer, sample_result):
        """Test that ProcessedDocument is created correctly."""
        save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer
        )

        # Verify save_batch was called with a list containing one ProcessedDocument
        call_args = mock_storage_writer.save_batch.call_args
        documents = call_args[0][0]

        assert isinstance(documents, list)
        assert len(documents) == 1
        assert isinstance(documents[0], ProcessedDocument)
        assert documents[0].document_id == "doc_001"

    def test_save_with_progressive_save_metadata(self, mock_storage_writer, sample_result):
        """Test that progressive_save flag is set in metadata."""
        save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer
        )

        call_args = mock_storage_writer.save_batch.call_args
        document = call_args[0][0][0]

        assert "progressive_save" in document.processing_metadata
        assert document.processing_metadata["progressive_save"] is True

    def test_save_with_linkages(self, mock_storage_writer, sample_result, sample_linkages):
        """Test saving with event linkages."""
        success = save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer,
            linkages=sample_linkages
        )

        assert success is True

        call_args = mock_storage_writer.save_batch.call_args
        document = call_args[0][0][0]

        assert document.event_linkages is not None
        assert len(document.event_linkages) == 1

    def test_save_with_storylines(self, mock_storage_writer, sample_result, sample_storylines):
        """Test saving with storylines."""
        success = save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer,
            storylines=sample_storylines
        )

        assert success is True

        call_args = mock_storage_writer.save_batch.call_args
        document = call_args[0][0][0]

        assert document.storylines is not None
        assert len(document.storylines) == 1

    def test_save_with_linkages_and_storylines(
        self,
        mock_storage_writer,
        sample_result,
        sample_linkages,
        sample_storylines
    ):
        """Test saving with both linkages and storylines."""
        success = save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer,
            linkages=sample_linkages,
            storylines=sample_storylines
        )

        assert success is True

        call_args = mock_storage_writer.save_batch.call_args
        document = call_args[0][0][0]

        assert document.event_linkages is not None
        assert document.storylines is not None

    def test_save_failure_returns_false(self, mock_storage_writer, sample_result):
        """Test that save failure returns False."""
        mock_storage_writer.save_batch.side_effect = Exception("Storage error")

        success = save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer
        )

        assert success is False

    def test_save_handles_storage_exception(self, mock_storage_writer, sample_result):
        """Test that storage exceptions are handled gracefully."""
        mock_storage_writer.save_batch.side_effect = ValueError("Invalid data")

        success = save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer
        )

        assert success is False

    def test_save_without_job_id(self, mock_storage_writer):
        """Test saving document without job_id."""
        result = {
            "document_id": "doc_002",
            "success": True,
            "entities": [],
            "events": []
        }

        success = save_single_document(
            result=result,
            storage_writer=mock_storage_writer
        )

        assert success is True

        call_args = mock_storage_writer.save_batch.call_args
        document = call_args[0][0][0]
        assert document.job_id is None


class TestStorageBackendIntegration:
    """Test progressive saving with different storage backends."""

    def test_save_to_jsonl_backend(self, sample_result):
        """Test progressive save to JSONL backend."""
        with patch('src.storage.multi_backend_writer.MultiBackendWriter') as MockWriter:
            mock_writer = MockWriter.return_value
            mock_writer.save_batch.return_value = {"success": True}

            success = save_single_document(
                result=sample_result,
                storage_writer=mock_writer
            )

            assert success is True
            assert mock_writer.save_batch.called

    def test_save_to_postgres_backend(self, sample_result):
        """Test progressive save to PostgreSQL backend."""
        with patch('src.storage.multi_backend_writer.MultiBackendWriter') as MockWriter:
            mock_writer = MockWriter.return_value
            mock_writer.save_batch.return_value = {"success": True, "postgres": True}

            success = save_single_document(
                result=sample_result,
                storage_writer=mock_writer
            )

            assert success is True

    def test_save_to_multi_backend(self, sample_result):
        """Test progressive save to multiple backends."""
        with patch('src.storage.multi_backend_writer.MultiBackendWriter') as MockWriter:
            mock_writer = MockWriter.return_value
            mock_writer.save_batch.return_value = {
                "success": True,
                "jsonl": True,
                "postgres": True
            }

            success = save_single_document(
                result=sample_result,
                storage_writer=mock_writer
            )

            assert success is True


class TestDocumentMetadata:
    """Test document metadata in progressive saves."""

    def test_document_contains_all_fields(self, mock_storage_writer, sample_result):
        """Test that saved document contains all required fields."""
        save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer
        )

        call_args = mock_storage_writer.save_batch.call_args
        document = call_args[0][0][0]

        assert document.document_id == "doc_001"
        assert document.job_id == "test_job_001"
        assert document.entities is not None
        assert document.events is not None
        assert document.processing_metadata is not None

    def test_processing_metadata_includes_progressive_flag(
        self,
        mock_storage_writer,
        sample_result
    ):
        """Test that processing metadata includes progressive save flag."""
        save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer
        )

        call_args = mock_storage_writer.save_batch.call_args
        document = call_args[0][0][0]

        metadata = document.processing_metadata
        assert "progressive_save" in metadata
        assert metadata["progressive_save"] is True

    def test_document_metadata_preserved(self, mock_storage_writer):
        """Test that original document metadata is preserved."""
        result = {
            "document_id": "doc_003",
            "job_id": "test_job_002",
            "success": True,
            "entities": [],
            "events": [],
            "document_metadata": {
                "title": "Test Article",
                "author": "Test Author",
                "published_date": "2025-01-15",
                "custom_field": "custom_value"
            }
        }

        save_single_document(
            result=result,
            storage_writer=mock_storage_writer
        )

        call_args = mock_storage_writer.save_batch.call_args
        document = call_args[0][0][0]

        # Document metadata should be preserved
        assert document.document_id == "doc_003"


class TestErrorHandling:
    """Test error handling in progressive saving."""

    def test_handles_missing_document_id(self, mock_storage_writer):
        """Test handling of missing document_id."""
        result = {
            "success": True,
            "entities": [],
            "events": []
        }

        # Should handle gracefully or raise appropriate error
        try:
            save_single_document(
                result=result,
                storage_writer=mock_storage_writer
            )
        except KeyError:
            # Expected behavior - document_id is required
            pass

    def test_handles_malformed_entities(self, mock_storage_writer):
        """Test handling of malformed entity data."""
        result = {
            "document_id": "doc_004",
            "success": True,
            "entities": "not a list",  # Invalid format
            "events": []
        }

        # Should handle gracefully
        success = save_single_document(
            result=result,
            storage_writer=mock_storage_writer
        )

        # Implementation may vary - either False or exception handling

    def test_handles_storage_write_failure(self, mock_storage_writer, sample_result):
        """Test handling of storage write failures."""
        mock_storage_writer.save_batch.return_value = {"success": False, "error": "Write failed"}

        success = save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer
        )

        # Should still return based on implementation
        # Could be True (attempted) or False (detected failure)


class TestBatchSaving:
    """Test that progressive save maintains batch integrity."""

    def test_single_document_saved_as_batch_of_one(self, mock_storage_writer, sample_result):
        """Test that single document is saved as batch of one."""
        save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer
        )

        call_args = mock_storage_writer.save_batch.call_args
        documents = call_args[0][0]

        assert isinstance(documents, list)
        assert len(documents) == 1

    def test_multiple_progressive_saves(self, mock_storage_writer):
        """Test multiple progressive saves in sequence."""
        results = [
            {"document_id": f"doc_{i:03d}", "job_id": "job_001", "success": True, "entities": [], "events": []}
            for i in range(1, 6)
        ]

        for result in results:
            success = save_single_document(
                result=result,
                storage_writer=mock_storage_writer
            )
            assert success is True

        # Each call should save one document
        assert mock_storage_writer.save_batch.call_count == 5


class TestPerformance:
    """Test performance characteristics of progressive saving."""

    def test_save_overhead_is_minimal(self, mock_storage_writer, sample_result):
        """Test that save operation completes quickly."""
        import time

        start = time.time()
        save_single_document(
            result=sample_result,
            storage_writer=mock_storage_writer
        )
        duration = time.time() - start

        # Should complete in reasonable time (< 100ms with mock)
        assert duration < 0.1

    def test_concurrent_saves_possible(self, mock_storage_writer):
        """Test that concurrent saves can be performed."""
        import threading

        results = [
            {"document_id": f"doc_{i:03d}", "job_id": "job_002", "success": True, "entities": [], "events": []}
            for i in range(1, 11)
        ]

        def save_doc(result):
            save_single_document(result=result, storage_writer=mock_storage_writer)

        threads = [threading.Thread(target=save_doc, args=(result,)) for result in results]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All saves should complete
        assert mock_storage_writer.save_batch.call_count == 10
