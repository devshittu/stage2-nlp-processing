"""
Unit tests for storage backends.
"""
import pytest
from unittest.mock import Mock, patch, mock_open
from datetime import datetime
from pathlib import Path
from src.storage.backends import JSONLBackend
from src.schemas.data_models import ProcessedDocument


class TestJSONLBackend:
    """Test JSONL storage backend."""

    def test_jsonl_backend_initialization(self):
        """Test backend initialization."""
        backend = JSONLBackend()

        assert hasattr(backend, "output_dir")
        assert backend.output_dir is not None

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_save_document(self, mock_mkdir, mock_file):
        """Test saving a single document."""
        backend = JSONLBackend()

        doc = ProcessedDocument(
            document_id="doc_001",
            job_id="job_123",
            processed_at=datetime.now().isoformat(),
            normalized_date="2024-12-13T00:00:00Z",
            original_text="Test text",
            source_document={},
            extracted_entities=[],
            extracted_soa_triplets=[],
            events=[],
            event_linkages=[],
            storylines=[],
            processing_metadata={}
        )

        result = backend.save(doc)

        assert result == True
        mock_file.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_save_batch(self, mock_mkdir, mock_file):
        """Test saving a batch of documents."""
        backend = JSONLBackend()

        docs = [
            ProcessedDocument(
                document_id=f"doc_{i}",
                job_id="job_123",
                processed_at=datetime.now().isoformat(),
                normalized_date="2024-12-13T00:00:00Z",
                original_text="Test text",
                source_document={},
                extracted_entities=[],
                extracted_soa_triplets=[],
                events=[],
                event_linkages=[],
                storylines=[],
                processing_metadata={}
            )
            for i in range(3)
        ]

        result = backend.save_batch(docs)

        # save_batch returns number of successfully saved docs
        assert result == 3

    @patch('builtins.open', side_effect=Exception("Write error"))
    @patch('pathlib.Path.mkdir')
    def test_save_error_handling(self, mock_mkdir, mock_file):
        """Test error handling during save."""
        backend = JSONLBackend()

        doc = ProcessedDocument(
            document_id="doc_001",
            job_id="job_123",
            processed_at=datetime.now().isoformat(),
            normalized_date="2024-12-13T00:00:00Z",
            original_text="Test text",
            source_document={},
            extracted_entities=[],
            extracted_soa_triplets=[],
            events=[],
            event_linkages=[],
            storylines=[],
            processing_metadata={}
        )

        result = backend.save(doc)

        # Should return False on error
        assert result == False
