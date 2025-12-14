
# CLAUDE.md

This file provides contextual information for Claude Code (claude.ai/code) when considering the development of code in a new repository for Stage 2 of the broader project.

## Project Overview

The **Event & Entity Extraction (EEE) Pipeline** is intended as a production-ready microservices system that extracts events, entities, and relationships from unstructured text using state-of-the-art NLP models available as of December 2025 (such as Mistral-7B, spaCy, HuggingFace transformers, or equivalents).

## Broader Project Context: Sequential Storytelling Pipeline

This potential repository would represent **Stage 2 (NLP Processing Service)** of an 8-stage sequential storytelling pipeline that transforms raw news articles into coherent, temporal narratives by connecting the dots across stories.

### The Complete Pipeline

**Stage 1: Cleaning Ingestion Service** (Upstream)
- Preprocesses raw news articles
- Normalizes data
- Stores in PostgreSQL with in-memory cache layer
- **Outputs**: Cleaned documents with structured fields (cleaned_text, cleaned_title, cleaned_author, etc.)

**Stage 2: NLP Processing Service** ← **POTENTIAL FOCUS HERE (New Repository)**
- Separate entity/event extraction for better control and debugging
- Utilizes advanced models with GPU acceleration where applicable
- **Inputs**: Cleaned documents from Stage 1 (via configurable field mapping)
- **Outputs**: Extracted events, entities, SOA triplets, sentiment, causality → JSONL/PostgreSQL/Elasticsearch

**Stage 3: Embedding Generation Service** (Downstream)
- Generates high-dimensional embeddings (768D)
- Stores in vector database (Pinecone/FAISS)
- Leverages 160GB RAM for large-scale processing

**Stage 4: Clustering Service**
- Clusters articles using FAISS-GPU and HDBSCAN
- Temporal sorting optimized for 48-core Threadripper

**Stage 5: Graph Construction Service**
- Builds neo4j knowledge graph with batch processing
- Connects entities and events across stories

**Stage 6: Timeline Generation Service**
- Summarizes timelines using DeepSeek R1
- Caches in Redis for performance

**Stage 7: API Service**
- FastAPI backend serving timelines
- Real-time query optimization

**Stage 8: Frontend Service**
- React/TypeScript UI for interactive display

### Design Philosophy: Independent Black Box Services

Each stage lives in its own repository and treats others as black boxes:
- **Clear Contracts**: Well-defined input/output schemas
- **Independent Deployment**: Each service can be developed, tested, deployed separately
- **Data Transformation**: Each stage transforms and moves data down the pipeline
- **Loose Coupling**: Changes in one stage don't break others (as long as contracts are maintained)

### Stage 2 Contract

**Expected Inputs (from Stage 1):**
```json
{
  "document_id": "unique-id",
  "cleaned_text": "The main article text...",
  "cleaned_title": "Article Title",
  "cleaned_author": "Author Name",
  "cleaned_publication_date": "2024-01-15T10:30:00Z",
  "cleaned_source_url": "https://example.com/article",
  "cleaned_excerpt": "Brief summary...",
  "cleaned_categories": ["politics", "technology"],
  "cleaned_tags": ["AI", "regulation"],
  "cleaned_word_count": 850
}
```

**Guaranteed Outputs (to Stage 3):**
```json
{
  "events": [
    {
      "event_type": "policy_change",
      "trigger": {"text": "announced", "start_char": 10, "end_char": 19},
      "arguments": [
        {"argument_role": "agent", "entity": {"text": "UK Government", "type": "ORG"}},
        {"argument_role": "time", "entity": {"text": "yesterday", "type": "DATE"}}
      ],
      "metadata": {"sentiment": "neutral", "causality": "Policy follows pilot program"}
    }
  ],
  "extracted_entities": [
    {"text": "UK Government", "type": "ORG", "start_char": 0, "end_char": 13}
  ],
  "extracted_soa_triplets": [
    {"subject": {"text": "UK Government"}, "action": {"text": "announced"}, "object": {"text": "policy"}}
  ],
  "document_id": "unique-id",
  "normalized_date": "2024-01-15T10:30:00Z",
  "original_text": "The main article text...",
  "job_id": "processing-job-id",
  "source_document": {/* original input fields preserved */}
}
```

**Configuration for Stage 1 Integration:**

In a potential `config/settings.yaml`, the `document_field_mapping` section could define how to extract fields from Stage 1's output:

```yaml
document_field_mapping:
  text_field: "cleaned_text"  # Primary text field to process
  text_field_fallbacks: ["original_text", "text", "content"]  # Fallbacks if primary missing
  context_fields: ["cleaned_title", "cleaned_excerpt", "cleaned_author", "cleaned_publication_date"]
  preserve_in_output: ["document_id", "cleaned_publication_date", "cleaned_source_url"]
```

### Hardware Context

The system is optimized for:
- **CPU**: 48-core AMD Threadripper (24 physical, 48 threads)
- **RAM**: 160GB
- **GPU**: NVIDIA RTX A4000 (16GB VRAM)
- **OS**: Linux

Potential configuration could leverage this hardware via:
- Parallelism options: Up to 22 workers (half of 48 cores)
- GPU acceleration for NER, DP, and LLM inference
- Large memory limits: 140GB for cluster operations

## Development Considerations

### Architecture Insights

A microservices approach could involve multiple Docker services communicating via HTTP and Redis, focusing on:
- Named Entity Recognition (NER)
- Dependency Parsing (DP)
- Event extraction with advanced models
- Orchestration for coordination
- Background batch processing
- Broker for job status

Processing flows could emphasize asynchronous batch handling for bulk inputs from Stage 1.

### Data Flow Patterns

- Handling of long documents: Splitting into chunks with overlap to manage token limits.
- Event linking: Post-processing to connect co-referent events across batches using embeddings.
- Flexible input schema: Support for arbitrary document fields via configurable mapping.
- Multi-backend storage: Simultaneous writes to JSONL, PostgreSQL, and/or Elasticsearch.

## Key Configuration Insights

Potential settings in `config/settings.yaml` could include:
- GPU usage
- Model selection with chunk size and token limits
- Batch processing chunk size
- Parallelism settings (workers, threads, memory)
- Document field mapping for enriched input
- Storage backends

### Environment Variables

Potential `.env.dev` could include:
```bash
HUGGINGFACE_TOKEN=hf_YOUR_TOKEN_HERE
```

## Important Patterns & Constraints

### Model Memory Requirements

- NER: ~2GB VRAM per inference
- DP: ~2GB VRAM per inference
- Event extraction: ~4GB VRAM (quantized), ~16GB unquantized
- Total per document: ~6GB VRAM for concurrent processing

### Batch Processing Strategy

- Chunk size: Controls docs per task
- Workers: ~half of CPU cores
- Memory limit: Leave buffer for OS/Docker

For hardware (48-core CPU, 160GB RAM, RTX A4000):
- Workers: 22
- Memory limit: "140GB"
- Batch chunk size: 200

### Chunking Behavior

Documents exceeding token limits split into overlapping chunks to prevent context loss.

### Event Linking Algorithm

Post-processing: Extract descriptions, generate embeddings, compute similarities, link above threshold, merge unique info.

### Storage Backend Behavior

Enabled backends write simultaneously; failure in one does not affect others.

## Performance Characteristics

### Batch Processing

- Throughput: 100-300 documents/hour (with 22 workers)
- Scaling: Linear with workers up to GPU saturation
- Memory: ~6GB per concurrent document
- Optimal batch size: 100-200 documents

### Hardware Recommendations

- Minimum: 8GB RAM, 4-core CPU, 8GB VRAM
- Recommended: 32GB RAM, 16-core CPU, 16GB VRAM
- Optimal: 160GB RAM, 48-core CPU, 16GB VRAM (RTX A4000)

## Notes on Code Style

- Structured logging (JSON format)
- Pydantic models for API contracts
- Async/await for HTTP calls
- Type hints throughout
- Configuration via YAML
- Health checks
- Graceful error handling