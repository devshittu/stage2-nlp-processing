#!/usr/bin/env python3
"""
Test script to submit batch to Stage 2 and verify metadata registry writes.
"""
import json
import requests
import time
import sys

# Configuration
ORCHESTRATOR_URL = "http://localhost:9080/api/v1/nlp"
TEST_FILE = "/home/mshittu/projects/nlp/stage2-nlp-processing/data/test_batch_metadata_registry.jsonl"

def load_test_documents():
    """Load test documents from JSONL file."""
    documents = []
    with open(TEST_FILE, 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            documents.append(doc)
    return documents

def submit_batch(documents):
    """Submit batch to Stage 2 for processing."""
    url = f"{ORCHESTRATOR_URL}/api/v1/documents/batch"
    payload = {
        "documents": documents,
        "batch_id": f"test_metadata_registry_{int(time.time())}"
    }

    print(f"Submitting batch of {len(documents)} documents...")
    response = requests.post(url, json=payload, timeout=30)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Batch submitted successfully!")
        print(f"  Job ID: {result.get('job_id')}")
        print(f"  Batch ID: {result.get('batch_id')}")
        print(f"  Status: {result.get('status')}")
        return result
    else:
        print(f"✗ Failed to submit batch: {response.status_code}")
        print(f"  Response: {response.text}")
        return None

def check_job_status(job_id):
    """Check job status."""
    url = f"{ORCHESTRATOR_URL}/api/v1/jobs/{job_id}/status"
    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        return response.json()
    else:
        return None

def main():
    print("=" * 60)
    print("Stage 2 Metadata Registry Integration Test")
    print("=" * 60)
    print()

    # Load test documents
    print("Loading test documents...")
    documents = load_test_documents()
    print(f"✓ Loaded {len(documents)} documents")
    print()

    # Submit batch
    result = submit_batch(documents)
    if not result:
        sys.exit(1)

    job_id = result.get('job_id')
    print()

    # Monitor job
    print("Monitoring job progress...")
    while True:
        time.sleep(5)
        status = check_job_status(job_id)

        if status:
            state = status.get('status', 'unknown')
            progress = status.get('progress', {})
            processed = progress.get('documents_processed', 0)
            total = progress.get('documents_total', 0)

            print(f"  Status: {state} - {processed}/{total} documents")

            if state in ['completed', 'failed']:
                break
        else:
            print("  Could not fetch status")
            break

    print()
    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Check celery worker logs for metadata registry writes")
    print("2. Verify PostgreSQL pipeline_metadata database")
    print("3. Verify Redis cache (DB 15)")

if __name__ == "__main__":
    main()
