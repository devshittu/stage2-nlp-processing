#!/usr/bin/env python3
"""
System Performance Test Script

Tests the NLP processing pipeline with both short and long articles,
collecting metrics on:
- Processing time
- Event extraction quality
- Entity deduplication effectiveness
"""

import json
import time
import httpx
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict


class PerformanceTester:
    """Test runner for NLP processing system."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.client = httpx.Client(timeout=600.0)
        self.results = {
            "short_articles": {},
            "long_articles": {},
            "summary": {}
        }

    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load documents from JSONL file."""
        documents = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))
        return documents

    def process_batch(self, documents: List[Dict], batch_name: str) -> Tuple[str, float]:
        """Process a batch of documents and return job_id and submission time."""
        print(f"\n{'='*70}")
        print(f"Processing batch: {batch_name}")
        print(f"Documents: {len(documents)}")
        print(f"{'='*70}\n")

        payload = {
            "documents": documents,
            "batch_id": f"perf_test_{batch_name}_{int(time.time())}"
        }

        # Submit batch and measure time
        start_time = time.time()
        response = self.client.post(f"{self.api_url}/api/v1/documents/batch", json=payload)
        submission_time = time.time() - start_time

        response.raise_for_status()
        result = response.json()

        if not result.get("success"):
            raise Exception(f"Batch submission failed: {result.get('error')}")

        job_id = result["job_id"]
        print(f"✓ Batch submitted successfully")
        print(f"  Job ID: {job_id}")
        print(f"  Submission time: {submission_time:.2f}s\n")

        return job_id, submission_time

    def wait_for_completion(self, job_id: str, batch_name: str) -> Tuple[Dict, float]:
        """Wait for job completion and return results with total processing time."""
        print(f"Waiting for job {job_id} to complete...")

        start_time = time.time()
        last_status = None

        while True:
            response = self.client.get(f"{self.api_url}/api/v1/jobs/{job_id}")
            response.raise_for_status()
            status_data = response.json()

            status = status_data.get("status")
            progress = status_data.get("progress") or 0
            docs_processed = status_data.get("documents_processed") or 0
            docs_total = status_data.get("documents_total") or 0

            if status != last_status:
                print(f"  Status: {status} - Progress: {progress:.1f}% ({docs_processed}/{docs_total})")
                last_status = status

            if status == "SUCCESS":
                processing_time = time.time() - start_time
                print(f"✓ Job completed successfully in {processing_time:.2f}s\n")
                return status_data.get("result", {}), processing_time

            elif status == "FAILURE":
                error = status_data.get("error", "Unknown error")
                raise Exception(f"Job failed: {error}")

            time.sleep(2)  # Poll every 2 seconds

    def load_processed_documents(self, job_id: str) -> List[Dict]:
        """Load processed documents from JSONL file by job_id."""
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = f"/app/data/extracted_events_{today}.jsonl"

        documents = []
        if not Path(output_file).exists():
            print(f"Warning: Output file {output_file} does not exist")
            return documents

        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    if doc.get("job_id") == job_id:
                        documents.append(doc)

        print(f"✓ Loaded {len(documents)} processed documents for job {job_id}\n")
        return documents

    def analyze_results(self, results: List[Dict], article_type: str) -> Dict[str, Any]:
        """Analyze processing results for quality metrics."""
        print(f"\n{'='*70}")
        print(f"Analyzing results for {article_type}")
        print(f"{'='*70}\n")

        metrics = {
            "total_documents": len(results),
            "total_entities": 0,
            "total_events": 0,
            "total_triplets": 0,
            "entity_types": Counter(),
            "event_types": Counter(),
            "duplicate_entities": 0,
            "entity_dedup_rate": 0.0,
            "avg_entities_per_doc": 0.0,
            "avg_events_per_doc": 0.0,
            "processing_times": [],
            "entity_examples": [],
            "event_examples": [],
            "deduplication_analysis": {}
        }

        # Track entities for deduplication analysis
        entity_text_counts = Counter()
        entity_normalized_counts = Counter()
        all_entities = []

        for doc in results:
            # Count entities
            entities = doc.get("extracted_entities", [])
            metrics["total_entities"] += len(entities)
            all_entities.extend(entities)

            for ent in entities:
                entity_type = ent.get("type", "UNKNOWN")
                metrics["entity_types"][entity_type] += 1

                # Track for deduplication
                text = ent.get("text", "")
                normalized = ent.get("normalized_form", text)
                entity_text_counts[text] += 1
                entity_normalized_counts[normalized] += 1

            # Count events
            events = doc.get("events", [])
            metrics["total_events"] += len(events)

            for evt in events:
                event_type = evt.get("event_type", "UNKNOWN")
                metrics["event_types"][event_type] += 1

            # Count triplets
            triplets = doc.get("extracted_soa_triplets", [])
            metrics["total_triplets"] += len(triplets)

        # Calculate deduplication metrics
        raw_entity_count = sum(entity_text_counts.values())
        unique_text_count = len(entity_text_counts)
        unique_normalized_count = len(entity_normalized_counts)

        metrics["deduplication_analysis"] = {
            "raw_entity_mentions": raw_entity_count,
            "unique_text_forms": unique_text_count,
            "unique_normalized_forms": unique_normalized_count,
            "text_dedup_rate": (1 - unique_text_count / raw_entity_count) * 100 if raw_entity_count > 0 else 0,
            "normalized_dedup_rate": (1 - unique_normalized_count / raw_entity_count) * 100 if raw_entity_count > 0 else 0
        }

        # Find duplicate entities (mentioned multiple times)
        duplicates = {text: count for text, count in entity_text_counts.items() if count > 1}
        metrics["duplicate_entities"] = len(duplicates)
        metrics["entity_dedup_rate"] = (len(duplicates) / unique_text_count * 100) if unique_text_count > 0 else 0

        # Average metrics
        if metrics["total_documents"] > 0:
            metrics["avg_entities_per_doc"] = metrics["total_entities"] / metrics["total_documents"]
            metrics["avg_events_per_doc"] = metrics["total_events"] / metrics["total_documents"]

        # Collect examples (first 5 entities and events)
        if all_entities:
            metrics["entity_examples"] = [
                {
                    "text": e.get("text"),
                    "type": e.get("type"),
                    "confidence": e.get("confidence"),
                    "normalized": e.get("normalized_form")
                }
                for e in all_entities[:5]
            ]

        if results and results[0].get("events"):
            metrics["event_examples"] = [
                {
                    "type": e.get("event_type"),
                    "trigger": e.get("trigger", {}).get("text"),
                    "arguments": len(e.get("arguments", []))
                }
                for e in results[0]["events"][:5]
            ]

        return metrics

    def validate_event_extraction(self, results: List[Dict]) -> Dict[str, Any]:
        """Validate quality of event extraction."""
        validation = {
            "has_events": 0,
            "has_triggers": 0,
            "has_arguments": 0,
            "avg_arguments_per_event": 0.0,
            "event_coverage": 0.0,
            "quality_score": 0.0
        }

        total_events = 0
        total_arguments = 0

        for doc in results:
            events = doc.get("events", [])
            if events:
                validation["has_events"] += 1

                for evt in events:
                    total_events += 1

                    # Check for trigger
                    if evt.get("trigger"):
                        validation["has_triggers"] += 1

                    # Check for arguments
                    args = evt.get("arguments", [])
                    if args:
                        validation["has_arguments"] += 1
                        total_arguments += len(args)

        # Calculate metrics
        if total_events > 0:
            validation["avg_arguments_per_event"] = total_arguments / total_events
            validation["trigger_coverage"] = (validation["has_triggers"] / total_events) * 100
            validation["argument_coverage"] = (validation["has_arguments"] / total_events) * 100

        if len(results) > 0:
            validation["event_coverage"] = (validation["has_events"] / len(results)) * 100

        # Quality score: weighted average of coverage metrics
        validation["quality_score"] = (
            validation.get("event_coverage", 0) * 0.4 +
            validation.get("trigger_coverage", 0) * 0.3 +
            validation.get("argument_coverage", 0) * 0.3
        )

        return validation

    def validate_entity_deduplication(self, results: List[Dict]) -> Dict[str, Any]:
        """Validate entity deduplication effectiveness."""
        validation = {
            "total_entities": 0,
            "entities_with_normalized_form": 0,
            "potential_duplicates_found": [],
            "dedup_effectiveness": 0.0,
            "normalization_rate": 0.0
        }

        # Group entities by normalized form
        entity_groups = defaultdict(list)

        for doc in results:
            entities = doc.get("extracted_entities", [])
            validation["total_entities"] += len(entities)

            for ent in entities:
                text = ent.get("text", "")
                normalized = ent.get("normalized_form")

                if normalized:
                    validation["entities_with_normalized_form"] += 1
                    entity_groups[normalized].append(text)
                else:
                    entity_groups[text].append(text)

        # Find groups with multiple variants
        for normalized, variants in entity_groups.items():
            unique_variants = set(variants)
            if len(unique_variants) > 1:
                validation["potential_duplicates_found"].append({
                    "normalized": normalized,
                    "variants": list(unique_variants),
                    "count": len(variants)
                })

        # Calculate metrics
        if validation["total_entities"] > 0:
            validation["normalization_rate"] = (
                validation["entities_with_normalized_form"] / validation["total_entities"]
            ) * 100

        # Deduplication effectiveness: how many entity groups have duplicates resolved
        if entity_groups:
            groups_with_duplicates = len([g for g in entity_groups.values() if len(set(g)) > 1])
            validation["dedup_effectiveness"] = (
                (1 - groups_with_duplicates / len(entity_groups)) * 100
            )

        return validation

    def run_test(self, file_path: str, test_name: str) -> Dict[str, Any]:
        """Run complete test for a dataset."""
        # Load documents
        documents = self.load_jsonl(file_path)

        # Process batch
        job_id, submission_time = self.process_batch(documents, test_name)

        # Wait for completion
        summary_stats, processing_time = self.wait_for_completion(job_id, test_name)

        # Load processed documents from JSONL file
        results = self.load_processed_documents(job_id)

        # Analyze results
        metrics = self.analyze_results(results, test_name)

        # Validate event extraction
        event_validation = self.validate_event_extraction(results)

        # Validate entity deduplication
        dedup_validation = self.validate_entity_deduplication(results)

        return {
            "test_name": test_name,
            "file_path": file_path,
            "document_count": len(documents),
            "timing": {
                "submission_time": submission_time,
                "total_processing_time": processing_time,
                "avg_time_per_doc": processing_time / len(documents) if documents else 0
            },
            "metrics": metrics,
            "event_validation": event_validation,
            "entity_deduplication": dedup_validation,
            "results": results
        }

    def print_report(self, test_results: Dict[str, Any]):
        """Print formatted test results."""
        print(f"\n{'='*70}")
        print(f"TEST RESULTS: {test_results['test_name']}")
        print(f"{'='*70}\n")

        # Timing
        print("TIMING METRICS:")
        print(f"  Submission time: {test_results['timing']['submission_time']:.2f}s")
        print(f"  Total processing time: {test_results['timing']['total_processing_time']:.2f}s")
        print(f"  Avg time per document: {test_results['timing']['avg_time_per_doc']:.2f}s")

        # Extraction metrics
        metrics = test_results['metrics']
        print(f"\nEXTRACTION METRICS:")
        print(f"  Total documents: {metrics['total_documents']}")
        print(f"  Total entities: {metrics['total_entities']}")
        print(f"  Total events: {metrics['total_events']}")
        print(f"  Total triplets: {metrics['total_triplets']}")
        print(f"  Avg entities per doc: {metrics['avg_entities_per_doc']:.2f}")
        print(f"  Avg events per doc: {metrics['avg_events_per_doc']:.2f}")

        # Entity types
        print(f"\n  Entity types distribution:")
        for ent_type, count in metrics['entity_types'].most_common():
            print(f"    {ent_type}: {count}")

        # Event types
        if metrics['event_types']:
            print(f"\n  Event types distribution:")
            for evt_type, count in metrics['event_types'].most_common():
                print(f"    {evt_type}: {count}")

        # Event validation
        ev = test_results['event_validation']
        print(f"\nEVENT EXTRACTION QUALITY:")
        print(f"  Documents with events: {ev['has_events']}/{metrics['total_documents']} ({ev['event_coverage']:.1f}%)")
        print(f"  Events with triggers: {ev.get('trigger_coverage', 0):.1f}%")
        print(f"  Events with arguments: {ev.get('argument_coverage', 0):.1f}%")
        print(f"  Avg arguments per event: {ev['avg_arguments_per_event']:.2f}")
        print(f"  Overall quality score: {ev['quality_score']:.1f}/100")

        # Deduplication
        dedup = metrics['deduplication_analysis']
        print(f"\nENTITY DEDUPLICATION ANALYSIS:")
        print(f"  Raw entity mentions: {dedup['raw_entity_mentions']}")
        print(f"  Unique text forms: {dedup['unique_text_forms']}")
        print(f"  Unique normalized forms: {dedup['unique_normalized_forms']}")
        print(f"  Text-level dedup rate: {dedup['text_dedup_rate']:.1f}%")
        print(f"  Normalized dedup rate: {dedup['normalized_dedup_rate']:.1f}%")

        # Deduplication validation
        dv = test_results['entity_deduplication']
        print(f"\nDEDUPLICATION EFFECTIVENESS:")
        print(f"  Total entities: {dv['total_entities']}")
        print(f"  Entities normalized: {dv['entities_with_normalized_form']} ({dv['normalization_rate']:.1f}%)")
        print(f"  Dedup effectiveness: {dv['dedup_effectiveness']:.1f}%")

        if dv['potential_duplicates_found'][:3]:
            print(f"\n  Sample duplicate groups found:")
            for dup in dv['potential_duplicates_found'][:3]:
                print(f"    '{dup['normalized']}' → {dup['variants']}")

        print(f"\n{'='*70}\n")

    def generate_summary(self, short_results: Dict, long_results: Dict) -> Dict:
        """Generate comparison summary."""
        summary = {
            "test_date": datetime.now().isoformat(),
            "comparison": {
                "short_vs_long_processing_time": {
                    "short_avg": short_results['timing']['avg_time_per_doc'],
                    "long_avg": long_results['timing']['avg_time_per_doc'],
                    "ratio": long_results['timing']['avg_time_per_doc'] / short_results['timing']['avg_time_per_doc'] if short_results['timing']['avg_time_per_doc'] > 0 else 0
                },
                "short_vs_long_extraction": {
                    "short_entities_per_doc": short_results['metrics']['avg_entities_per_doc'],
                    "long_entities_per_doc": long_results['metrics']['avg_entities_per_doc'],
                    "short_events_per_doc": short_results['metrics']['avg_events_per_doc'],
                    "long_events_per_doc": long_results['metrics']['avg_events_per_doc']
                },
                "quality_comparison": {
                    "short_event_quality": short_results['event_validation']['quality_score'],
                    "long_event_quality": long_results['event_validation']['quality_score'],
                    "short_dedup_effectiveness": short_results['entity_deduplication']['dedup_effectiveness'],
                    "long_dedup_effectiveness": long_results['entity_deduplication']['dedup_effectiveness']
                }
            }
        }

        return summary

    def save_results(self, output_file: str):
        """Save all results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Full results saved to: {output_file}")


def main():
    """Main test execution."""
    print("\n" + "="*70)
    print("NLP PROCESSING SYSTEM - PERFORMANCE TEST")
    print("="*70)

    tester = PerformanceTester()

    # Test short articles
    short_results = tester.run_test(
        "/app/data/test_short_articles.jsonl",
        "short_articles"
    )
    tester.print_report(short_results)
    tester.results["short_articles"] = short_results

    # Test long articles
    long_results = tester.run_test(
        "/app/data/test_long_articles.jsonl",
        "long_articles"
    )
    tester.print_report(long_results)
    tester.results["long_articles"] = long_results

    # Generate summary
    summary = tester.generate_summary(short_results, long_results)
    tester.results["summary"] = summary

    # Print final summary
    print("="*70)
    print("COMPARATIVE SUMMARY")
    print("="*70)
    print(f"\nProcessing Time Ratio (Long/Short): {summary['comparison']['short_vs_long_processing_time']['ratio']:.2f}x")
    print(f"\nExtraction Rates:")
    print(f"  Short articles - Entities: {summary['comparison']['short_vs_long_extraction']['short_entities_per_doc']:.2f}/doc, Events: {summary['comparison']['short_vs_long_extraction']['short_events_per_doc']:.2f}/doc")
    print(f"  Long articles  - Entities: {summary['comparison']['short_vs_long_extraction']['long_entities_per_doc']:.2f}/doc, Events: {summary['comparison']['short_vs_long_extraction']['long_events_per_doc']:.2f}/doc")
    print(f"\nQuality Scores:")
    print(f"  Short articles - Event quality: {summary['comparison']['quality_comparison']['short_event_quality']:.1f}/100, Dedup: {summary['comparison']['quality_comparison']['short_dedup_effectiveness']:.1f}%")
    print(f"  Long articles  - Event quality: {summary['comparison']['quality_comparison']['long_event_quality']:.1f}/100, Dedup: {summary['comparison']['quality_comparison']['long_dedup_effectiveness']:.1f}%")
    print("="*70)

    # Save results
    output_file = f"/app/data/performance_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    tester.save_results(output_file)

    print("\n✓ All tests completed successfully!\n")


if __name__ == "__main__":
    main()
