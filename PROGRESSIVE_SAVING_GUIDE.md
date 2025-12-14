# Progressive Saving & Checkpoint Guide

Comprehensive guide for progressive saving, checkpointing, and pause/resume functionality in batch processing.

## 🎯 Key Features

### ✅ **Progressive Saving**
- Documents saved immediately after processing (not at batch end)
- No data loss on timeout, crash, or interruption
- Already processed documents persist to storage
- Partial batch results always available

### ✅ **Checkpointing**
- Automatic checkpoint creation for all batch jobs
- Tracks processed and failed documents
- Enables pause/resume functionality
- Survives worker restarts and crashes

### ✅ **Pause/Resume/Stop**
- **Pause**: Temporarily halt processing, can resume later
- **Resume**: Continue from where you left off
- **Stop**: Permanently halt (cannot resume)
- Graceful completion of current document before pausing/stopping

### ✅ **Error Resilience**
- Failed documents don't stop the entire batch
- Each failure tracked and logged separately
- Successful documents continue processing
- Comprehensive error reporting per document

---

## 📊 How Progressive Saving Works

### Traditional Batch Processing (Old Behavior)
```
Process all docs → Link events → Save everything → Return
                                    ↑
                            If fails here, ALL work lost!
```

### Progressive Saving (New Behavior)
```
For each document:
  1. Process (NER → DP → Event LLM)
  2. ✅ Save immediately to storage
  3. ✅ Update checkpoint
  4. Continue to next document

Result: No data loss, even on interruption!
```

---

## 🗂️ Checkpoint System

### Checkpoint Location
```
/app/data/checkpoints/{job_id}.json
```

### Checkpoint Contents
```json
{
  "job_id": "f90aa645-4061-4e91-8199-43504ee4b770",
  "batch_id": "batch_20251214_204300",
  "status": "RUNNING",
  "total_documents": 253,
  "processed_documents": 136,
  "failed_documents": 12,
  "processed_doc_ids": ["doc_101", "doc_102", ...],
  "failed_doc_ids": ["doc_105", "doc_132"],
  "created_at": "2025-12-14T20:43:00Z",
  "updated_at": "2025-12-14T21:45:30Z",
  "metadata": {
    "started_at": "2025-12-14T20:43:00Z",
    "progressive_save_enabled": true
  }
}
```

### Checkpoint Status Values
- **RUNNING**: Currently processing
- **PAUSED**: Temporarily paused, can resume
- **STOPPED**: Permanently stopped
- **COMPLETED**: Successfully finished
- **FAILED**: Failed with error

---

## 🎮 Pause/Resume/Stop Operations

### Pause a Batch
```bash
# Pause currently running batch
./scripts/pause_batch.sh <job_id>

# Or pause last submitted batch
./scripts/pause_batch.sh
```

**What happens:**
1. Pause signal written to checkpoint
2. Current document finishes processing
3. Batch stops after current document
4. Checkpoint marked as PAUSED
5. All processed documents already saved

### Resume a Paused Batch
```bash
# Resume paused batch
./scripts/resume_batch.sh <job_id>

# Then resubmit the original batch
# System automatically detects checkpoint and resumes
./scripts/submit_batch.sh data/your_file.jsonl original_batch_id
```

**What happens:**
1. Checkpoint loaded
2. Already processed documents skipped
3. Processing continues from next unprocessed document
4. No duplicate processing
5. Seamless continuation

### Stop a Batch (Permanent)
```bash
# Stop batch permanently
./scripts/stop_batch.sh <job_id>
```

**What happens:**
1. Stop signal written to checkpoint
2. Current document finishes processing
3. Batch stops permanently
4. Checkpoint marked as STOPPED
5. Cannot be resumed (use pause for temporary stops)

### Check Checkpoint Status
```bash
# View detailed checkpoint information
./scripts/checkpoint_status.sh <job_id>
```

**Output example:**
```
======================================================================
CHECKPOINT STATUS
======================================================================
Job ID: f90aa645-4061-4e91-8199-43504ee4b770

Status:              PAUSED
Batch ID:            batch_20251214_204300
Total Documents:     253
Processed:           136 (53.8%)
Failed:              12
Remaining:           105
Created:             2025-12-14T20:43:00Z
Updated:             2025-12-14T21:45:30Z

Recent Processed Documents (last 10):
  ✓ doc_227
  ✓ doc_228
  ✓ doc_229
  ...

Failed Documents (12):
  ✗ doc_105
  ✗ doc_132
  ...
```

---

## 💾 Data Persistence Guarantees

### What's Saved Progressively
- ✅ Processed documents (entities, events, SOA triplets)
- ✅ Document metadata
- ✅ Processing timestamps
- ✅ Error information for failed documents

### What's Saved at End
- Event linkages across batch (requires all documents)
- Storylines (requires event linking)
- Batch-level statistics

### Storage Backends
All progressive saves go to configured backends:
- JSONL files (`/app/data/extracted_events_*.jsonl`)
- PostgreSQL (if enabled)
- Elasticsearch (if enabled)

---

## 🔄 Resume Behavior

### Automatic Resume on Resubmission
```bash
# Original submission
./scripts/submit_batch.sh data/my_batch.jsonl batch_001
# Processes 136/253 docs, then pauses

# Resume by resubmitting with SAME batch_id
./scripts/submit_batch.sh data/my_batch.jsonl batch_001
# Automatically:
#   - Loads checkpoint
#   - Skips 136 processed docs
#   - Starts from doc 137
#   - No duplicate processing!
```

### Manual Resume Steps
1. Check checkpoint status
2. Update status to RUNNING (if paused)
3. Resubmit original batch with same batch_id
4. System loads checkpoint and resumes

---

## 📈 Progress Monitoring

### Real-time Progress with Checkpoints
```bash
./scripts/monitor_batch.sh <job_id>
```

**Enhanced Output:**
```
[1] 20:43:25 | Status: PROGRESS | Progress: 136/253 (53.8%) | Failed: 12 | Rate: 2.5 docs/min
[2] 20:43:55 | Status: PROGRESS | Progress: 138/253 (54.5%) | Failed: 12 | Rate: 2.0 docs/min
...
```

Shows:
- Total progress (not just logs)
- Failed document count
- Percentage complete
- Processing rate

---

## 🛡️ Error Handling

### Individual Document Failures
**Behavior:**
- Document fails → Logged and tracked
- Checkpoint updated with failed doc ID
- Batch continues processing next document
- No cascade failure

**Example:**
```
Processing 253 documents...
✓ doc_101 processed successfully
✓ doc_102 processed successfully
✗ doc_103 processing failed: Timeout
✓ doc_104 processed successfully  ← Continues despite failure
...

Final: 241 successful, 12 failed
```

### Failed Document Tracking
```bash
# View all failed documents
./scripts/checkpoint_status.sh <job_id>

# Failed documents section shows:
Failed Documents (12):
  ✗ doc_103 (Timeout)
  ✗ doc_105 (Invalid format)
  ✗ doc_132 (Network error)
  ...
```

---

## 🎯 Use Cases

### Scenario 1: Long-Running Batch Interrupted
```
Problem: Processing 1000 docs, crashes after 500
Old behavior: Lose all 500 processed docs, start over
New behavior: 500 docs already saved, resume from doc 501
```

### Scenario 2: Need to Pause for Maintenance
```
1. ./scripts/pause_batch.sh <job_id>
2. Perform maintenance
3. ./scripts/resume_batch.sh <job_id>
4. Resubmit batch
5. Continues seamlessly
```

### Scenario 3: Debugging Failed Documents
```
1. Batch completes with some failures
2. ./scripts/checkpoint_status.sh <job_id>
3. See exactly which docs failed
4. Fix issues in those specific docs
5. Reprocess only failed docs
```

### Scenario 4: Resource Management
```
Problem: Batch using too much GPU, need to free resources
Solution:
  1. Pause batch
  2. Resources freed after current doc
  3. Resume when resources available
  4. No data loss, picks up where left off
```

---

## 📊 Performance Impact

### Progressive Saving Overhead
- **Per Document:** ~10-50ms additional time
- **Total Batch:** Negligible (<2% overhead)
- **Benefit:** Complete resilience and recoverability

### Checkpoint Update Performance
- **File I/O:** Atomic writes with locking
- **Update Time:** <5ms per checkpoint update
- **Storage:** ~1-5KB per checkpoint file

### Trade-offs
**Pros:**
- ✅ No data loss on interruption
- ✅ Pause/resume capability
- ✅ Failed docs don't stop batch
- ✅ Progress always visible

**Cons:**
- Small performance overhead (~2%)
- Additional disk I/O
- Checkpoint files need management

---

## 🔧 Advanced Configuration

### Checkpoint Directory
Default: `/app/data/checkpoints/`

To change, modify `CheckpointManager` initialization in `celery_tasks.py`.

### Cleanup Old Checkpoints
```bash
# Remove checkpoints older than 7 days
find /app/data/checkpoints -name "*.json" -mtime +7 -delete
```

### Disable Progressive Saving (Not Recommended)
Progressive saving is always enabled and cannot be disabled without code changes. This ensures data safety.

---

## 🐛 Troubleshooting

### Checkpoint Not Found
**Problem:** `./scripts/resume_batch.sh` says "No checkpoint found"

**Solutions:**
- Verify job_id is correct
- Check `/app/data/checkpoints/` exists
- Ensure job was submitted (not just planned)

### Resume Doesn't Skip Processed Docs
**Problem:** Resubmission reprocesses all documents

**Cause:** Different batch_id used

**Solution:** Use exact same batch_id as original submission

### Checkpoint Shows Wrong Progress
**Problem:** Monitor shows different count than checkpoint

**Cause:** Log-based counting vs checkpoint-based

**Solution:** Trust checkpoint (more accurate), logs may lag

---

## 📚 Related Documentation

- **Basic Usage:** `scripts/README.md`
- **Docker Commands:** `BATCH_PROCESSING_GUIDE.md`
- **API Reference:** Check orchestrator service docs

---

## ✅ Summary

**Progressive saving + checkpointing provides:**
1. ✅ Complete data safety
2. ✅ Pause/resume capability
3. ✅ Error resilience (failed docs don't stop batch)
4. ✅ No reprocessing on resume
5. ✅ Always-visible progress
6. ✅ Graceful interruption handling

**Your data is safe with progressive saving enabled!**
