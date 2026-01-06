"""
backends.py

Storage backend implementations for Stage 2 NLP Processing Service.
Supports JSONL, PostgreSQL, and Elasticsearch with simultaneous multi-backend writes.

Features:
- JSONL: Daily rotating files with optional compression
- PostgreSQL: JSONB columns for complex data structures
- Elasticsearch: Auto-index creation with optimized mappings
- Graceful error handling (failure in one backend doesn't affect others)
- Batch write support for efficiency
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Dict
from abc import ABC, abstractmethod

# PostgreSQL
try:
    import psycopg2
    from psycopg2 import sql, extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Elasticsearch
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

from src.utils.config_manager import get_settings
from src.utils.logger import get_logger
from src.schemas.data_models import ProcessedDocument

logger = get_logger(__name__)


# =============================================================================
# Base Storage Backend
# =============================================================================

class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save(self, document: ProcessedDocument) -> bool:
        """Save single document. Returns True if successful."""
        pass

    @abstractmethod
    def save_batch(self, documents: List[ProcessedDocument]) -> int:
        """Save batch of documents. Returns number successfully saved."""
        pass

    @abstractmethod
    def close(self):
        """Close connections and cleanup resources."""
        pass


# =============================================================================
# JSONL Backend
# =============================================================================

class JSONLBackend(StorageBackend):
    """
    JSONL file storage backend with timestamped output files.

    Creates files like: extracted_events_2025-01-15_14-30-45.jsonl
    Human-readable format: YYYY-MM-DD_HH-MM-SS
    """

    def __init__(self, job_id: Optional[str] = None):
        """
        Initialize JSONL backend.

        Args:
            job_id: Optional job ID to include in filename for batch traceability
        """
        self.settings = get_settings().storage.jsonl
        self.output_dir = Path(self.settings.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = None
        self.current_date = None
        self.job_id = job_id

        # Generate timestamp at initialization for consistent filenames within a session
        self._session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        logger.info(
            "JSONL backend initialized",
            extra={
                "output_dir": str(self.output_dir),
                "session_timestamp": self._session_timestamp
            }
        )

    def _get_file_path(self, use_session_timestamp: bool = True) -> Path:
        """
        Get output file path with human-readable timestamp.

        Args:
            use_session_timestamp: If True, use session timestamp for consistency.
                                   If False, generate new timestamp (for daily rotation).

        Returns:
            Path to output file
        """
        # Determine timestamp format based on settings
        timestamp_format = getattr(self.settings, 'timestamp_format', 'datetime')

        if timestamp_format == 'datetime':
            # Human-readable datetime: YYYY-MM-DD_HH-MM-SS
            if use_session_timestamp:
                timestamp_str = self._session_timestamp
            else:
                timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        elif timestamp_format == 'date' or getattr(self.settings, 'use_daily_files', False):
            # Date only: YYYY-MM-DD (legacy support)
            timestamp_str = datetime.now().strftime("%Y-%m-%d")
        else:
            # No timestamp
            timestamp_str = None

        # Build filename
        if timestamp_str:
            if self.job_id:
                filename = f"{self.settings.file_prefix}_{timestamp_str}_{self.job_id[:8]}.jsonl"
            else:
                filename = f"{self.settings.file_prefix}_{timestamp_str}.jsonl"
        else:
            filename = f"{self.settings.file_prefix}.jsonl"

        if self.settings.compression == "gzip":
            filename += ".gz"
        elif self.settings.compression == "bz2":
            filename += ".bz2"

        return self.output_dir / filename

    def get_output_file_path(self) -> str:
        """Get the current output file path as string (for external reference)."""
        return str(self._get_file_path())

    def save(self, document: ProcessedDocument) -> bool:
        """Save single document to JSONL file."""
        try:
            file_path = self._get_file_path()

            # Convert to dict and serialize
            doc_dict = document.model_dump(mode='json')
            json_line = json.dumps(doc_dict, ensure_ascii=False)

            # Write to file
            mode = 'a' if self.settings.append_mode else 'w'

            if self.settings.compression == "gzip":
                import gzip
                with gzip.open(file_path, mode + 't', encoding='utf-8') as f:
                    f.write(json_line + '\n')
            elif self.settings.compression == "bz2":
                import bz2
                with bz2.open(file_path, mode + 't', encoding='utf-8') as f:
                    f.write(json_line + '\n')
            else:
                with open(file_path, mode, encoding='utf-8') as f:
                    f.write(json_line + '\n')

            logger.debug(
                f"Saved document to JSONL",
                extra={"document_id": document.document_id, "file": str(file_path)}
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to save document to JSONL",
                exc_info=True,
                extra={"document_id": document.document_id, "error": str(e)}
            )
            return False

    def save_batch(self, documents: List[ProcessedDocument]) -> int:
        """
        Save batch of documents efficiently (buffered write).

        This is much faster than calling save() for each document
        as it opens the file once and writes all documents.
        """
        if not documents:
            return 0

        try:
            file_path = self._get_file_path()
            mode = 'a' if self.settings.append_mode else 'w'

            # Prepare all lines first
            lines = []
            for doc in documents:
                try:
                    doc_dict = doc.model_dump(mode='json')
                    json_line = json.dumps(doc_dict, ensure_ascii=False)
                    lines.append(json_line)
                except Exception as e:
                    logger.warning(
                        f"Failed to serialize document {doc.document_id}: {e}",
                        extra={"document_id": doc.document_id}
                    )

            # Write all lines in one operation (buffered write)
            if self.settings.compression == "gzip":
                import gzip
                with gzip.open(file_path, mode + 't', encoding='utf-8') as f:
                    f.write('\n'.join(lines) + '\n')
            elif self.settings.compression == "bz2":
                import bz2
                with bz2.open(file_path, mode + 't', encoding='utf-8') as f:
                    f.write('\n'.join(lines) + '\n')
            else:
                with open(file_path, mode, encoding='utf-8') as f:
                    f.write('\n'.join(lines) + '\n')

            logger.info(
                f"Batch saved to JSONL: {len(lines)} documents",
                extra={"file": str(file_path), "count": len(lines)}
            )
            return len(lines)

        except Exception as e:
            logger.error(
                f"Failed to save batch to JSONL: {e}",
                exc_info=True
            )
            return 0

    def close(self):
        """Close JSONL backend (no cleanup needed)."""
        pass


# =============================================================================
# PostgreSQL Backend
# =============================================================================

class PostgreSQLBackend(StorageBackend):
    """
    PostgreSQL storage backend with JSONB columns.

    Stores complex structures (events, entities) as JSONB for flexible querying.
    """

    def __init__(self):
        """Initialize PostgreSQL backend."""
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")

        self.settings = get_settings().storage.postgresql
        self.conn = None

        try:
            self._connect()
            self._create_table()

            logger.info(
                "PostgreSQL backend initialized",
                extra={"host": self.settings.host, "database": self.settings.database}
            )
        except Exception as e:
            # Ensure connection is closed on initialization failure
            if self.conn:
                try:
                    self.conn.close()
                except:
                    pass
            logger.error(f"Failed to initialize PostgreSQL backend: {e}", exc_info=True)
            raise

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(
                host=self.settings.host,
                port=self.settings.port,
                database=self.settings.database,
                user=self.settings.user,
                password=self.settings.password
            )
            logger.debug("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}", exc_info=True)
            raise

    def _create_table(self):
        """Create table if not exists."""
        if not self.settings.create_table_if_not_exists:
            return

        try:
            with self.conn.cursor() as cur:
                if self.settings.use_jsonb_columns:
                    # JSONB schema for flexible storage
                    create_query = sql.SQL("""
                        CREATE TABLE IF NOT EXISTS {table} (
                            document_id VARCHAR(255) PRIMARY KEY,
                            job_id VARCHAR(255),
                            processed_at TIMESTAMP,
                            normalized_date VARCHAR(100),
                            original_text TEXT,
                            extracted_entities JSONB,
                            extracted_soa_triplets JSONB,
                            events JSONB,
                            event_linkages JSONB,
                            storylines JSONB,
                            source_document JSONB,
                            processing_metadata JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );

                        CREATE INDEX IF NOT EXISTS idx_{table}_processed_at
                            ON {table} (processed_at);
                        CREATE INDEX IF NOT EXISTS idx_{table}_job_id
                            ON {table} (job_id);
                        CREATE INDEX IF NOT EXISTS idx_{table}_events
                            ON {table} USING GIN (events);
                    """).format(table=sql.Identifier(self.settings.table_name))
                else:
                    # Simple text schema
                    create_query = sql.SQL("""
                        CREATE TABLE IF NOT EXISTS {table} (
                            document_id VARCHAR(255) PRIMARY KEY,
                            job_id VARCHAR(255),
                            processed_at TIMESTAMP,
                            data JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """).format(table=sql.Identifier(self.settings.table_name))

                cur.execute(create_query)
                self.conn.commit()
                logger.debug(f"Table {self.settings.table_name} ready")

        except Exception as e:
            logger.error(f"Failed to create table: {e}", exc_info=True)
            self.conn.rollback()

    def save(self, document: ProcessedDocument, auto_commit: bool = True) -> bool:
        """
        Save single document to PostgreSQL.

        Args:
            document: Document to save
            auto_commit: If True, commits immediately. If False, caller must commit.

        Returns:
            True if successful, False otherwise

        Note:
            For batch operations, use save_batch() which is much more efficient.
        """
        try:
            doc_dict = document.model_dump(mode='json')

            with self.conn.cursor() as cur:
                if self.settings.use_jsonb_columns:
                    # Insert with JSONB columns
                    insert_query = sql.SQL("""
                        INSERT INTO {table} (
                            document_id, job_id, processed_at, normalized_date,
                            original_text, extracted_entities, extracted_soa_triplets,
                            events, event_linkages, storylines, source_document,
                            processing_metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (document_id) DO UPDATE SET
                            job_id = EXCLUDED.job_id,
                            processed_at = EXCLUDED.processed_at,
                            events = EXCLUDED.events;
                    """).format(table=sql.Identifier(self.settings.table_name))

                    cur.execute(insert_query, (
                        doc_dict['document_id'],
                        doc_dict.get('job_id'),
                        doc_dict['processed_at'],
                        doc_dict.get('normalized_date'),
                        doc_dict['original_text'],
                        json.dumps(doc_dict['extracted_entities']),
                        json.dumps(doc_dict['extracted_soa_triplets']),
                        json.dumps(doc_dict['events']),
                        json.dumps(doc_dict.get('event_linkages')),
                        json.dumps(doc_dict.get('storylines')),
                        json.dumps(doc_dict['source_document']),
                        json.dumps(doc_dict['processing_metadata'])
                    ))
                else:
                    # Insert as single JSONB column
                    insert_query = sql.SQL("""
                        INSERT INTO {table} (document_id, job_id, processed_at, data)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (document_id) DO UPDATE SET
                            data = EXCLUDED.data;
                    """).format(table=sql.Identifier(self.settings.table_name))

                    cur.execute(insert_query, (
                        doc_dict['document_id'],
                        doc_dict.get('job_id'),
                        doc_dict['processed_at'],
                        json.dumps(doc_dict)
                    ))

                # Only commit if auto_commit is True (for single document saves)
                if auto_commit:
                    self.conn.commit()

            logger.debug(
                f"Saved document to PostgreSQL",
                extra={"document_id": document.document_id, "auto_commit": auto_commit}
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to save document to PostgreSQL",
                exc_info=True,
                extra={"document_id": document.document_id, "error": str(e)}
            )
            if auto_commit:
                self.conn.rollback()
            return False

    def save_batch(self, documents: List[ProcessedDocument]) -> int:
        """
        Save batch using execute_batch for efficiency.

        This is much faster than calling save() for each document because:
        1. Uses psycopg2's execute_batch for batched inserts
        2. Commits only once instead of N times
        3. Reduces network round-trips

        Args:
            documents: List of documents to save

        Returns:
            Number of documents successfully saved
        """
        if not documents:
            return 0

        try:
            doc_dicts = [doc.model_dump(mode='json') for doc in documents]

            with self.conn.cursor() as cur:
                if self.settings.use_jsonb_columns:
                    insert_query = sql.SQL("""
                        INSERT INTO {table} (
                            document_id, job_id, processed_at, normalized_date,
                            original_text, extracted_entities, extracted_soa_triplets,
                            events, event_linkages, storylines, source_document,
                            processing_metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (document_id) DO UPDATE SET
                            events = EXCLUDED.events;
                    """).format(table=sql.Identifier(self.settings.table_name))

                    data = [
                        (
                            d['document_id'], d.get('job_id'), d['processed_at'],
                            d.get('normalized_date'), d['original_text'],
                            json.dumps(d['extracted_entities']),
                            json.dumps(d['extracted_soa_triplets']),
                            json.dumps(d['events']),
                            json.dumps(d.get('event_linkages')),
                            json.dumps(d.get('storylines')),
                            json.dumps(d['source_document']),
                            json.dumps(d['processing_metadata'])
                        )
                        for d in doc_dicts
                    ]
                else:
                    insert_query = sql.SQL("""
                        INSERT INTO {table} (document_id, job_id, processed_at, data)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (document_id) DO UPDATE SET data = EXCLUDED.data;
                    """).format(table=sql.Identifier(self.settings.table_name))

                    data = [
                        (d['document_id'], d.get('job_id'), d['processed_at'], json.dumps(d))
                        for d in doc_dicts
                    ]

                # Batch insert with single commit
                extras.execute_batch(cur, insert_query, data, page_size=100)
                self.conn.commit()

            logger.info(
                f"Batch saved to PostgreSQL: {len(documents)} documents (single transaction)",
                extra={"count": len(documents)}
            )
            return len(documents)

        except Exception as e:
            logger.error(f"Failed to save batch to PostgreSQL: {e}", exc_info=True)
            self.conn.rollback()
            return 0

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.debug("PostgreSQL connection closed")


# =============================================================================
# Elasticsearch Backend
# =============================================================================

class ElasticsearchBackend(StorageBackend):
    """
    Elasticsearch storage backend with auto-index creation.

    Optimized for event search and aggregation queries.
    """

    def __init__(self):
        """Initialize Elasticsearch backend."""
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError("elasticsearch not installed. Install with: pip install elasticsearch")

        self.settings = get_settings().storage.elasticsearch
        self.es = None
        self._connect()
        self._create_index()

        logger.info(
            "Elasticsearch backend initialized",
            extra={"host": self.settings.host, "index": self.settings.index_name}
        )

    def _connect(self):
        """Connect to Elasticsearch."""
        try:
            es_config = {
                "hosts": [f"{'https' if self.settings.use_ssl else 'http'}://{self.settings.host}:{self.settings.port}"],
                "verify_certs": self.settings.verify_certs
            }

            if self.settings.api_key:
                es_config["api_key"] = self.settings.api_key

            self.es = Elasticsearch(**es_config)

            # Test connection
            if not self.es.ping():
                raise ConnectionError("Failed to ping Elasticsearch")

            logger.debug("Connected to Elasticsearch")

        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}", exc_info=True)
            raise

    def _create_index(self):
        """Create index with optimized mappings."""
        if not self.settings.create_index_if_not_exists:
            return

        if self.es.indices.exists(index=self.settings.index_name):
            logger.debug(f"Index {self.settings.index_name} already exists")
            return

        try:
            mappings = {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "job_id": {"type": "keyword"},
                    "processed_at": {"type": "date"},
                    "normalized_date": {"type": "date"},
                    "original_text": {"type": "text"},
                    "extracted_entities": {"type": "object"},
                    "extracted_soa_triplets": {"type": "object"},
                    "events": {
                        "type": "nested",
                        "properties": {
                            "event_id": {"type": "keyword"},
                            "event_type": {"type": "keyword"},
                            "domain": {"type": "keyword"},
                            "trigger": {"type": "object"},
                            "arguments": {"type": "nested"}
                        }
                    },
                    "storylines": {"type": "nested"},
                    "source_document": {"type": "object"},
                    "processing_metadata": {"type": "object"}
                }
            }

            # Using new Elasticsearch 8.x API (no 'body' parameter)
            self.es.indices.create(
                index=self.settings.index_name,
                settings={
                    "number_of_shards": self.settings.number_of_shards,
                    "number_of_replicas": self.settings.number_of_replicas,
                    "refresh_interval": self.settings.refresh_interval
                },
                mappings=mappings
            )

            logger.info(f"Created Elasticsearch index: {self.settings.index_name}")

        except Exception as e:
            logger.error(f"Failed to create index: {e}", exc_info=True)

    def save(self, document: ProcessedDocument) -> bool:
        """Save single document to Elasticsearch."""
        try:
            doc_dict = document.model_dump(mode='json')

            self.es.index(
                index=self.settings.index_name,
                id=document.document_id,
                document=doc_dict
            )

            logger.debug(
                f"Saved document to Elasticsearch",
                extra={"document_id": document.document_id}
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to save document to Elasticsearch",
                exc_info=True,
                extra={"document_id": document.document_id, "error": str(e)}
            )
            return False

    def save_batch(self, documents: List[ProcessedDocument]) -> int:
        """Save batch using bulk API."""
        try:
            actions = [
                {
                    "_index": self.settings.index_name,
                    "_id": doc.document_id,
                    "_source": doc.model_dump(mode='json')
                }
                for doc in documents
            ]

            success, failed = bulk(self.es, actions, raise_on_error=False)

            logger.info(
                f"Bulk saved to Elasticsearch: {success} success, {len(failed)} failed",
                extra={"success_count": success, "failed_count": len(failed)}
            )

            return success

        except Exception as e:
            logger.error(f"Failed to bulk save to Elasticsearch: {e}", exc_info=True)
            return 0

    def close(self):
        """Close Elasticsearch connection."""
        if self.es:
            self.es.close()
            logger.debug("Elasticsearch connection closed")


# =============================================================================
# Storage Backend Factory
# =============================================================================

class StorageBackendFactory:
    """Factory for creating storage backend instances."""

    @staticmethod
    def create_backend(backend_name: str, job_id: Optional[str] = None) -> Optional[StorageBackend]:
        """
        Create storage backend by name.

        Args:
            backend_name: Backend name ("jsonl", "postgresql", "elasticsearch")
            job_id: Optional job ID for traceability in output filenames

        Returns:
            Backend instance or None if creation failed
        """
        try:
            if backend_name == "jsonl":
                return JSONLBackend(job_id=job_id)
            elif backend_name == "postgresql":
                return PostgreSQLBackend()
            elif backend_name == "elasticsearch":
                return ElasticsearchBackend()
            else:
                logger.error(f"Unknown backend: {backend_name}")
                return None

        except Exception as e:
            logger.error(f"Failed to create {backend_name} backend: {e}", exc_info=True)
            return None

    @staticmethod
    def create_enabled_backends(job_id: Optional[str] = None) -> List[StorageBackend]:
        """
        Create all enabled backends from configuration.

        Args:
            job_id: Optional job ID for traceability in output filenames

        Returns:
            List of backend instances
        """
        settings = get_settings().storage
        backends = []

        for backend_name in settings.enabled_backends:
            backend = StorageBackendFactory.create_backend(backend_name, job_id=job_id)
            if backend:
                backends.append(backend)

        logger.info(f"Initialized {len(backends)} storage backends")
        return backends


# =============================================================================
# Multi-Backend Writer
# =============================================================================

class MultiBackendWriter:
    """
    Writes to multiple storage backends simultaneously.

    Failure in one backend does not affect others.
    """

    def __init__(self, backends: Optional[List[StorageBackend]] = None, job_id: Optional[str] = None):
        """
        Initialize with backends.

        Args:
            backends: Optional list of pre-configured backends
            job_id: Optional job ID for traceability in output filenames
        """
        self.backends = backends or StorageBackendFactory.create_enabled_backends(job_id=job_id)
        self.job_id = job_id

        logger.info(f"MultiBackendWriter initialized with {len(self.backends)} backends")

    def save(self, document: ProcessedDocument) -> Dict[str, bool]:
        """
        Save document to all backends.

        Returns:
            Dict mapping backend class name to success status
        """
        results = {}

        for backend in self.backends:
            backend_name = backend.__class__.__name__
            success = backend.save(document)
            results[backend_name] = success

        return results

    def save_batch(self, documents: List[ProcessedDocument]) -> Dict[str, int]:
        """
        Save batch to all backends.

        Returns:
            Dict mapping backend class name to number of documents saved
        """
        results = {}

        for backend in self.backends:
            backend_name = backend.__class__.__name__
            count = backend.save_batch(documents)
            results[backend_name] = count

        return results

    def close(self):
        """Close all backends."""
        for backend in self.backends:
            try:
                backend.close()
            except Exception as e:
                logger.error(f"Error closing {backend.__class__.__name__}: {e}")


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("Testing storage backends...")

    # Test JSONL backend
    print("\n1. Testing JSONL backend...")
    try:
        jsonl = JSONLBackend()
        print(f"   ✓ JSONL backend initialized")
        print(f"   Output directory: {jsonl.output_dir}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    print("\nStorage backends test completed!")
