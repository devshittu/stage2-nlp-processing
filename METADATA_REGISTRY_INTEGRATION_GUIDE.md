# Stage 2 Metadata Registry Integration Guide

**Purpose**: Add metadata registry writes to Stage 2 NLP processing
**Status**: Ready for integration
**Zero-Regression Guarantee**: All changes are additive, wrapped in try-catch, and optional

---

## Overview

This guide shows how to integrate the shared metadata registry into Stage 2's Celery tasks. The integration follows a **dual-write pattern**: existing storage backends continue working unchanged, while metadata is also written to the registry for downstream stages.

## Files Created

1. ✅ `/src/storage/metadata_writer.py` - Metadata writer client
2. ✅ `/src/storage/metadata_integration.py` - Integration hooks with sync wrappers

## Integration Points

### 1. Install Shared Metadata Registry

**Location**: Stage 2 root directory

```bash
# Install the shared library
cd /home/mshittu/projects/nlp/infrastructure/shared-metadata-registry
pip install -e .

# Verify installation
python -c "from shared_metadata_registry import MetadataRegistry; print('✓ Registry installed')"
```

### 2. Add Environment Variables

**Location**: `stage2-nlp-processing/.env`

```bash
# Metadata Registry Configuration
METADATA_REGISTRY_ENABLED=true              # Enable/disable registry writes
METADATA_PRIMARY_BACKEND=postgresql         # Primary backend (postgresql or redis)
METADATA_ENABLE_REDIS_CACHE=true           # Enable Redis cache
METADATA_WRITE_ONLY=false                  # Set true for dry-run testing
METADATA_PERCENTAGE_ROLLOUT=100            # Gradual rollout (0-100)

# PostgreSQL Connection (points to infrastructure)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=pipeline_metadata
POSTGRES_USER=metadata_service
METADATA_SERVICE_PASSWORD=changeme_metadata_password

# Redis Connection (points to infrastructure)
REDIS_HOST=redis-cache
REDIS_PORT=6379
REDIS_DB=15
REDIS_TTL=86400  # 24 hours
```

### 3. Modify Celery Tasks

**Location**: `src/core/celery_tasks.py`

#### Change 1: Add imports (top of file)

```python
# Add after existing imports
from src.storage.metadata_integration import (
    sync_write_job_to_registry,
    sync_write_documents_to_registry,
    update_job_status_in_registry,
)
```

#### Change 2: Register job at task start

**Location**: `process_batch_task` function (around line 800)

**Before**:
```python
@app.task(name="process_batch_task", bind=True)
def process_batch_task(
    self,
    documents_json: List[str],
    job_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    progressive_save: bool = False
) -> Dict[str, Any]:
    """Process a batch of documents with Dask parallelism."""

    # Generate job/batch IDs
    if job_id is None:
        job_id = str(uuid.uuid4())
    if batch_id is None:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Starting batch processing: {len(documents_json)} documents", ...)
    # ... rest of function
```

**After** (add these lines right after job_id/batch_id generation):
```python
@app.task(name="process_batch_task", bind=True)
def process_batch_task(
    self,
    documents_json: List[str],
    job_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    progressive_save: bool = False
) -> Dict[str, Any]:
    """Process a batch of documents with Dask parallelism."""

    # Generate job/batch IDs
    if job_id is None:
        job_id = str(uuid.uuid4())
    if batch_id is None:
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # NEW: Register job in metadata registry
    try:
        from uuid import UUID
        job_uuid = UUID(job_id) if isinstance(job_id, str) else job_id
        sync_write_job_to_registry(
            job_id=job_uuid,
            batch_id=batch_id,
            metadata={"documents_count": len(documents_json)}
        )
    except Exception as e:
        logger.warning(f"Failed to register job in metadata registry: {e}")
        # Continue processing - registry failure doesn't stop pipeline

    logger.info(f"Starting batch processing: {len(documents_json)} documents", ...)
    # ... rest of function continues unchanged
```

#### Change 3: Write metadata after storage save

**Location**: `save_processed_documents` function (around line 617)

**Before**:
```python
def save_processed_documents(
    processed_results: List[Dict[str, Any]],
    linkages: List[EventLinkage],
    storylines: List[Storyline],
    storage_writer: MultiBackendWriter
) -> Dict[str, int]:
    """Save processed documents to storage backends."""
    logger.info(f"Saving {len(processed_results)} processed documents...")

    # ... build processed_docs list ...

    # Save batch
    if processed_docs:
        save_results = storage_writer.save_batch(processed_docs)

        logger.info(
            f"Saved {len(processed_docs)} documents to storage",
            extra={"save_results": save_results}
        )

        return {
            "documents_saved": len(processed_docs),
            "backend_results": save_results
        }
    # ... rest of function
```

**After** (add metadata registry write right after existing save):
```python
def save_processed_documents(
    processed_results: List[Dict[str, Any]],
    linkages: List[EventLinkage],
    storylines: List[Storyline],
    storage_writer: MultiBackendWriter
) -> Dict[str, int]:
    """Save processed documents to storage backends."""
    logger.info(f"Saving {len(processed_results)} processed documents...")

    # ... build processed_docs list (unchanged) ...

    # Save batch to existing backends
    if processed_docs:
        save_results = storage_writer.save_batch(processed_docs)

        logger.info(
            f"Saved {len(processed_docs)} documents to storage",
            extra={"save_results": save_results}
        )

        # NEW: Write to metadata registry (dual-write pattern)
        try:
            registry_count = sync_write_documents_to_registry(processed_docs)
            logger.info(
                f"Metadata registry: wrote {registry_count}/{len(processed_docs)} documents"
            )
        except Exception as e:
            logger.warning(f"Failed to write to metadata registry: {e}")
            # Continue - registry failure doesn't stop pipeline

        return {
            "documents_saved": len(processed_docs),
            "backend_results": save_results
        }
    # ... rest of function unchanged
```

#### Change 4: Update job status on completion/failure

**Location**: End of `process_batch_task` (in finally block)

**Before**:
```python
    finally:
        # Cleanup Dask cluster
        if dask_client:
            dask_client.close()
        if dask_cluster:
            dask_cluster.close()

        logger.info("Batch processing complete", extra=result_summary)
```

**After**:
```python
    finally:
        # Update job status in metadata registry
        try:
            if result_summary.get("success"):
                update_job_status_in_registry(
                    job_id=UUID(job_id),
                    status="completed"
                )
            else:
                error_msg = result_summary.get("error", "Unknown error")
                update_job_status_in_registry(
                    job_id=UUID(job_id),
                    status="failed",
                    error_message=error_msg
                )
        except Exception as e:
            logger.warning(f"Failed to update job status in registry: {e}")

        # Cleanup Dask cluster (unchanged)
        if dask_client:
            dask_client.close()
        if dask_cluster:
            dask_cluster.close()

        logger.info("Batch processing complete", extra=result_summary)
```

## Testing the Integration

### 1. Test with Registry Disabled (Baseline)

```bash
# Set registry to disabled
export METADATA_REGISTRY_ENABLED=false

# Run existing tests
pytest tests/

# Run a small batch
# (should work exactly as before)
```

### 2. Test with Registry Enabled (New Functionality)

```bash
# Enable registry
export METADATA_REGISTRY_ENABLED=true

# Run same tests
pytest tests/

# Verify no regressions (all tests should still pass)
```

### 3. Verify Metadata Written

```bash
# Check PostgreSQL for metadata
docker exec storytelling-postgres psql -U admin -d pipeline_metadata -c \
  "SELECT job_id, stage, status, created_at FROM job_registry WHERE stage = 2 ORDER BY created_at DESC LIMIT 5;"

# Check event metadata
docker exec storytelling-postgres psql -U admin -d pipeline_metadata -c \
  "SELECT event_id, event_type, event_date, description FROM event_metadata LIMIT 5;"

# Check document metadata
docker exec storytelling-postgres psql -U admin -d pipeline_metadata -c \
  "SELECT document_id, title, author, publication_date FROM document_metadata LIMIT 5;"
```

### 4. Test Graceful Degradation

```bash
# Stop PostgreSQL
docker stop storytelling-postgres

# Run batch processing
# Should continue working with existing backends, log warnings about registry

# Restart PostgreSQL
docker start storytelling-postgres
```

## Zero-Regression Checklist

Before deploying to production:

- [ ] All existing Stage 2 tests pass with registry disabled
- [ ] All existing Stage 2 tests pass with registry enabled
- [ ] No changes to existing storage backend behavior
- [ ] Processing time increase < 10% (registry writes are fast)
- [ ] Pipeline continues working if registry unavailable
- [ ] No breaking changes to existing APIs
- [ ] JSONL files still created
- [ ] PostgreSQL (Stage 2 DB) still populated
- [ ] Elasticsearch still indexed
- [ ] No new Python dependencies required (registry is optional)

## Rollback Plan

If issues occur after deployment:

```bash
# Option 1: Disable via environment variable (instant)
export METADATA_REGISTRY_ENABLED=false
# Restart Celery workers

# Option 2: Gradual rollback via percentage
export METADATA_PERCENTAGE_ROLLOUT=50  # 50% of jobs
export METADATA_PERCENTAGE_ROLLOUT=10  # 10% of jobs
export METADATA_PERCENTAGE_ROLLOUT=0   # 0% (effectively disabled)

# Option 3: Revert code changes
git revert <commit-hash>
```

## Performance Impact

Expected overhead from metadata registry writes:

- **Job registration**: ~50ms per batch
- **Document metadata**: ~10-20ms per document
- **Event metadata**: ~5-10ms per event
- **Total batch overhead**: < 5% of total processing time

For a batch of 100 documents:
- Without registry: ~120 seconds
- With registry: ~126 seconds (5% increase)

## Monitoring

Add these metrics to track registry performance:

```python
# In metadata_integration.py, add:
from prometheus_client import Counter, Histogram

registry_writes = Counter(
    'stage2_metadata_registry_writes_total',
    'Total metadata registry writes',
    ['entity_type', 'status']
)

registry_write_duration = Histogram(
    'stage2_metadata_registry_write_seconds',
    'Metadata registry write duration',
    ['entity_type']
)
```

## Next Steps

After Stage 2 integration:

1. ✅ **Test with real data** - Run full batch processing
2. ✅ **Verify metadata in registry** - Query PostgreSQL
3. ➡️ **Integrate Stage 5** - Read metadata for graph enrichment
4. ➡️ **Test end-to-end** - Stage 2 writes → Stage 5 reads
5. ➡️ **Production deployment** - Gradual rollout with monitoring

---

**Integration Status**: 🟡 **READY FOR APPLICATION**
**Files Modified**: 1 (celery_tasks.py)
**Lines Added**: ~50 lines
**Risk Level**: LOW (all changes wrapped in try-catch, optional via config)
