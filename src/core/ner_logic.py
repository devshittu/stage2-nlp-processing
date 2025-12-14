"""
ner_logic.py

Named Entity Recognition (NER) core logic for Stage 2 NLP Processing Service.
Extracts entities from text using transformer-based models with GPU acceleration.

Features:
- GPU-accelerated inference with fallback to CPU
- Batch processing for efficiency
- Redis-based entity caching for duplicate detection
- Context extraction for entity disambiguation
- Confidence scoring and filtering
- Support for multiple entity types (PER, ORG, LOC, GPE, DATE, TIME, MONEY, MISC, EVENT)
- Comprehensive error handling and logging
- Structured logging with performance metrics

Model:
- Primary: Babelscape/wikineural-multilingual-ner
- Supports multilingual entity recognition
- Fine-tuned on WikiANN dataset
"""

import torch
import redis
import hashlib
import json
import traceback
from typing import List, Optional, Dict, Any, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)
from datetime import datetime

from src.schemas.data_models import Entity
from src.utils.config_manager import get_settings, get_device
from src.utils.logger import get_logger, PerformanceLogger, log_exception
from src.core.entity_resolution import resolve_entities


# =============================================================================
# Module Logger
# =============================================================================

logger = get_logger(__name__, service="ner_service")


# =============================================================================
# NER Model Class
# =============================================================================

class NERModel:
    """
    Named Entity Recognition model with GPU acceleration and caching.

    Features:
    - Automatic GPU/CPU device selection
    - Batch processing for efficiency
    - Redis caching for duplicate text detection
    - Context extraction around entities
    - Confidence-based filtering
    - Comprehensive error handling

    Example:
        ner_model = NERModel()
        entities = ner_model.extract_entities("Donald Trump met with Angela Merkel in Berlin.")
    """

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize NER model with configuration.

        Args:
            config_override: Optional configuration overrides for testing

        Raises:
            RuntimeError: If model loading fails
        """
        # Load configuration
        self.settings = get_settings()
        self.config = self.settings.ner_service
        self.cache_config = self.settings.caching

        # Apply overrides if provided
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)

        # Device setup
        self.device = get_device()
        logger.info(
            f"Initializing NER model on device: {self.device}",
            extra={
                "model_name": self.config.model_name,
                "device": self.device,
                "gpu_enabled": self.settings.general.gpu_enabled
            }
        )

        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None

        # Redis cache (optional)
        self.cache_client = None
        if self.cache_config.enabled and self.config.enable_cache:
            self._initialize_cache()

        # Load model
        self._load_model()

        logger.info(
            "NER model initialized successfully",
            extra={
                "model_name": self.config.model_name,
                "cache_enabled": self.cache_client is not None,
                "entity_types": self.config.entity_types
            }
        )

    def _initialize_cache(self) -> None:
        """
        Initialize Redis cache connection.

        Logs warning if connection fails but doesn't raise exception.
        """
        try:
            # Parse Redis URL
            redis_url = self.cache_config.redis_url
            self.cache_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # Test connection
            self.cache_client.ping()
            logger.info(
                "Redis cache connected successfully",
                extra={"redis_url": redis_url.split('@')[-1]}  # Hide credentials
            )
        except Exception as e:
            logger.warning(
                f"Failed to connect to Redis cache, continuing without caching: {e}",
                extra={"error": str(e)}
            )
            self.cache_client = None

    def _load_model(self) -> None:
        """
        Load NER model and tokenizer from HuggingFace.

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            with PerformanceLogger("model_loading", logger.logger):
                # Load tokenizer
                logger.info(f"Loading tokenizer: {self.config.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    use_fast=True
                )

                # Load model
                logger.info(f"Loading NER model: {self.config.model_name}")
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.config.model_name
                )

                # Move model to device
                self.model.to(self.device)

                # Create pipeline for easier inference
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    aggregation_strategy="simple"  # Merge sub-word tokens
                )

                logger.info(
                    "NER model loaded successfully",
                    extra={
                        "model_name": self.config.model_name,
                        "device": self.device,
                        "model_parameters": sum(p.numel() for p in self.model.parameters())
                    }
                )

        except Exception as e:
            log_exception(
                logger.logger,
                f"Failed to load NER model: {self.config.model_name}",
                model_name=self.config.model_name,
                error=str(e)
            )
            raise RuntimeError(f"NER model loading failed: {e}")

    def extract_entities(
        self,
        text: str,
        document_id: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Input text to process
            document_id: Optional document identifier for logging
            use_cache: Whether to use cache for this request

        Returns:
            List of Entity objects with type, position, confidence, and context

        Example:
            entities = ner_model.extract_entities(
                "Donald Trump met Angela Merkel in Berlin on Monday.",
                document_id="doc123"
            )
            # Returns: [
            #   Entity(text="Donald Trump", type="PER", start_char=0, end_char=12, ...),
            #   Entity(text="Angela Merkel", type="PER", start_char=17, end_char=30, ...),
            #   Entity(text="Berlin", type="LOC", start_char=34, end_char=40, ...),
            #   Entity(text="Monday", type="DATE", start_char=44, end_char=50, ...)
            # ]
        """
        if not text or not text.strip():
            logger.warning(
                "Empty text provided for NER extraction",
                extra={"document_id": document_id}
            )
            return []

        # Check cache first
        if use_cache and self.cache_client:
            cached_entities = self._get_from_cache(text)
            if cached_entities is not None:
                logger.debug(
                    "NER entities retrieved from cache",
                    extra={"document_id": document_id, "entity_count": len(cached_entities)}
                )
                return cached_entities

        # Extract entities
        try:
            with PerformanceLogger(
                "ner_extraction",
                logger.logger,
                document_id=document_id,
                text_length=len(text)
            ):
                entities = self._extract_entities_internal(text, document_id)

                # Apply entity resolution (deduplicate similar entities)
                # keep_all_mentions=True ensures all entities have normalized_form populated
                original_count = len(entities)
                entities = resolve_entities(
                    entities,
                    substring_match=True,
                    token_overlap_threshold=0.5,
                    fuzzy_similarity_threshold=0.85,
                    keep_all_mentions=True  # Keep all mentions but populate normalized_form for deduplication
                )

                # Cache results (cache the resolved entities)
                if use_cache and self.cache_client and entities:
                    self._save_to_cache(text, entities)

                logger.info(
                    "NER extraction completed",
                    extra={
                        "document_id": document_id,
                        "entity_count": len(entities),
                        "original_entity_count": original_count,
                        "deduplication_ratio": 1 - (len(entities) / original_count) if original_count > 0 else 0,
                        "text_length": len(text)
                    }
                )

                return entities

        except Exception as e:
            log_exception(
                logger.logger,
                f"NER extraction failed for document: {document_id}",
                document_id=document_id,
                text_length=len(text)
            )
            # Return empty list instead of raising to allow pipeline to continue
            return []

    def extract_entities_batch(
        self,
        texts: List[str],
        document_ids: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[List[Entity]]:
        """
        Extract entities from multiple texts in batch.

        Batch processing is more efficient for GPU inference.

        Args:
            texts: List of input texts
            document_ids: Optional list of document identifiers
            use_cache: Whether to use cache for this batch

        Returns:
            List of lists of Entity objects, one list per input text

        Example:
            texts = [
                "Donald Trump announced new policy.",
                "Angela Merkel visited Paris."
            ]
            batch_results = ner_model.extract_entities_batch(texts)
            # Returns: [[Entity(...)], [Entity(...)]]
        """
        if not texts:
            logger.warning("Empty text list provided for batch NER extraction")
            return []

        if document_ids is None:
            document_ids = [f"batch_doc_{i}" for i in range(len(texts))]

        logger.info(
            "Starting batch NER extraction",
            extra={"batch_size": len(texts)}
        )

        results = []

        try:
            with PerformanceLogger(
                "ner_batch_extraction",
                logger.logger,
                batch_size=len(texts)
            ):
                # Process texts individually (cache lookup per text)
                # TODO: Optimize by batching uncached texts together
                for i, text in enumerate(texts):
                    doc_id = document_ids[i] if i < len(document_ids) else f"batch_doc_{i}"
                    entities = self.extract_entities(text, document_id=doc_id, use_cache=use_cache)
                    results.append(entities)

                logger.info(
                    "Batch NER extraction completed",
                    extra={
                        "batch_size": len(texts),
                        "total_entities": sum(len(r) for r in results)
                    }
                )

                return results

        except Exception as e:
            log_exception(
                logger.logger,
                "Batch NER extraction failed",
                batch_size=len(texts)
            )
            # Return empty lists for all texts
            return [[] for _ in texts]

    def _extract_entities_internal(
        self,
        text: str,
        document_id: Optional[str] = None
    ) -> List[Entity]:
        """
        Internal method to extract entities using the NER pipeline.

        Args:
            text: Input text
            document_id: Optional document ID for logging

        Returns:
            List of Entity objects
        """
        # Truncate text if too long
        max_length = self.settings.general.max_text_length
        if len(text) > max_length:
            logger.warning(
                f"Text exceeds max length ({max_length}), truncating",
                extra={"document_id": document_id, "original_length": len(text)}
            )
            text = text[:max_length]

        # Run NER pipeline
        raw_entities = self.ner_pipeline(text)

        # Convert to Entity objects
        entities = []
        for i, raw_entity in enumerate(raw_entities):
            # Filter by confidence threshold
            if raw_entity['score'] < self.config.confidence_threshold:
                continue

            # Map entity type (handle variations)
            entity_type = self._normalize_entity_type(raw_entity['entity_group'])

            # Filter by allowed entity types
            if entity_type not in self.config.entity_types:
                continue

            # Extract context around entity
            context = self._extract_context(
                text,
                raw_entity['start'],
                raw_entity['end'],
                window_size=50
            )

            # Create Entity object
            entity = Entity(
                text=raw_entity['word'].strip(),
                type=entity_type,
                start_char=raw_entity['start'],
                end_char=raw_entity['end'],
                confidence=round(raw_entity['score'], 4),
                context=context,
                entity_id=f"{document_id}_entity_{i}" if document_id else f"entity_{i}"
            )

            entities.append(entity)

        return entities

    def _normalize_entity_type(self, entity_type: str) -> str:
        """
        Normalize entity type to standard format.

        Different models use different naming conventions.

        Args:
            entity_type: Raw entity type from model

        Returns:
            Normalized entity type
        """
        # Remove B- and I- prefixes (BIO tagging)
        entity_type = entity_type.replace("B-", "").replace("I-", "")

        # Map common variations
        type_mapping = {
            "PERSON": "PER",
            "ORGANIZATION": "ORG",
            "LOCATION": "LOC",
            "GEOPOLITICAL_ENTITY": "GPE",
            "GEOPOLITICAL": "GPE",
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "MONEY",
            "PERCENT": "MONEY",
            "CARDINAL": "MISC",
            "ORDINAL": "MISC",
            "QUANTITY": "MISC",
            "MISC": "MISC",
            "MISCELLANEOUS": "MISC",
            "EVENT": "EVENT"
        }

        return type_mapping.get(entity_type.upper(), entity_type.upper())

    def _extract_context(
        self,
        text: str,
        start_char: int,
        end_char: int,
        window_size: int = 50
    ) -> str:
        """
        Extract context around an entity for disambiguation.

        Args:
            text: Full text
            start_char: Entity start position
            end_char: Entity end position
            window_size: Number of characters to include on each side

        Returns:
            Context string with entity marked

        Example:
            text = "Donald Trump met Angela Merkel in Berlin on Monday."
            context = _extract_context(text, 0, 12, window_size=20)
            # Returns: "[Donald Trump] met Angela Merkel"
        """
        # Calculate context window
        context_start = max(0, start_char - window_size)
        context_end = min(len(text), end_char + window_size)

        # Extract context
        before = text[context_start:start_char]
        entity = text[start_char:end_char]
        after = text[end_char:context_end]

        # Combine with entity marker
        context = f"{before}[{entity}]{after}"

        # Clean up whitespace
        context = " ".join(context.split())

        return context

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.

        Uses MD5 hash of text for efficient lookup.

        Args:
            text: Input text

        Returns:
            Cache key string
        """
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"ner:entities:{text_hash}"

    def _get_from_cache(self, text: str) -> Optional[List[Entity]]:
        """
        Retrieve entities from cache.

        Args:
            text: Input text

        Returns:
            List of Entity objects if found, None otherwise
        """
        if not self.cache_client:
            return None

        try:
            cache_key = self._get_cache_key(text)
            cached_data = self.cache_client.get(cache_key)

            if cached_data:
                # Deserialize entities
                entities_dict = json.loads(cached_data)
                entities = [Entity(**entity_dict) for entity_dict in entities_dict]
                return entities

            return None

        except Exception as e:
            logger.warning(
                f"Cache retrieval failed: {e}",
                extra={"error": str(e)}
            )
            return None

    def _save_to_cache(self, text: str, entities: List[Entity]) -> None:
        """
        Save entities to cache.

        Args:
            text: Input text
            entities: List of Entity objects to cache
        """
        if not self.cache_client:
            return

        try:
            cache_key = self._get_cache_key(text)

            # Serialize entities
            entities_dict = [entity.model_dump() for entity in entities]
            cached_data = json.dumps(entities_dict, ensure_ascii=False)

            # Save to cache with TTL
            self.cache_client.setex(
                cache_key,
                self.cache_config.entity_cache_ttl,
                cached_data
            )

            logger.debug(
                "Entities cached successfully",
                extra={
                    "cache_key": cache_key,
                    "entity_count": len(entities),
                    "ttl_seconds": self.cache_config.entity_cache_ttl
                }
            )

        except Exception as e:
            logger.warning(
                f"Cache save failed: {e}",
                extra={"error": str(e)}
            )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.

        Returns:
            Dictionary with model information

        Example:
            info = ner_model.get_model_info()
            # Returns: {
            #   "model_name": "Babelscape/wikineural-multilingual-ner",
            #   "device": "cuda",
            #   "entity_types": ["PER", "ORG", ...],
            #   "confidence_threshold": 0.75,
            #   "cache_enabled": true
            # }
        """
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "entity_types": self.config.entity_types,
            "confidence_threshold": self.config.confidence_threshold,
            "cache_enabled": self.cache_client is not None,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }

    def clear_cache(self) -> bool:
        """
        Clear all cached entities.

        Returns:
            True if successful, False otherwise
        """
        if not self.cache_client:
            logger.warning("Cache not enabled, cannot clear")
            return False

        try:
            # Get all NER cache keys
            keys = self.cache_client.keys("ner:entities:*")

            if keys:
                self.cache_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cached NER entries")
                return True
            else:
                logger.info("No cached entries to clear")
                return True

        except Exception as e:
            log_exception(
                logger.logger,
                "Failed to clear cache",
                error=str(e)
            )
            return False


# =============================================================================
# Convenience Functions
# =============================================================================

# Global NER model instance (singleton pattern)
_ner_model_instance: Optional[NERModel] = None


def get_ner_model(config_override: Optional[Dict[str, Any]] = None) -> NERModel:
    """
    Get or create global NER model instance.

    Uses singleton pattern to avoid reloading model.

    Args:
        config_override: Optional configuration overrides

    Returns:
        NERModel instance

    Example:
        ner_model = get_ner_model()
        entities = ner_model.extract_entities("Some text...")
    """
    global _ner_model_instance

    if _ner_model_instance is None:
        _ner_model_instance = NERModel(config_override=config_override)

    return _ner_model_instance


def extract_entities(
    text: str,
    document_id: Optional[str] = None,
    use_cache: bool = True
) -> List[Entity]:
    """
    Convenience function to extract entities.

    Args:
        text: Input text
        document_id: Optional document ID
        use_cache: Whether to use cache

    Returns:
        List of Entity objects

    Example:
        entities = extract_entities("Donald Trump met Angela Merkel in Berlin.")
    """
    ner_model = get_ner_model()
    return ner_model.extract_entities(text, document_id=document_id, use_cache=use_cache)


def extract_entities_batch(
    texts: List[str],
    document_ids: Optional[List[str]] = None,
    use_cache: bool = True
) -> List[List[Entity]]:
    """
    Convenience function to extract entities from batch.

    Args:
        texts: List of input texts
        document_ids: Optional list of document IDs
        use_cache: Whether to use cache

    Returns:
        List of lists of Entity objects

    Example:
        batch_results = extract_entities_batch([
            "Text 1...",
            "Text 2..."
        ])
    """
    ner_model = get_ner_model()
    return ner_model.extract_entities_batch(texts, document_ids=document_ids, use_cache=use_cache)


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    """
    Test NER model with sample text.

    Run with: python -m src.core.ner_logic
    """
    import sys
    from src.utils.logger import setup_logging

    # Setup logging
    setup_logging(log_level="INFO", log_format="json")

    logger.info("Starting NER model test")

    # Sample text
    test_text = """
    Donald Trump, the former President of the United States, met with
    Angela Merkel, the Chancellor of Germany, in Berlin on Monday, March 15, 2021.
    They discussed trade relations between the European Union and the United States,
    as well as climate policy initiatives. The meeting took place at the
    Chancellery and lasted approximately two hours. Trump mentioned that he invested
    $50 million in renewable energy projects last year.
    """

    try:
        # Initialize model
        logger.info("Initializing NER model...")
        ner_model = get_ner_model()

        # Get model info
        model_info = ner_model.get_model_info()
        logger.info("Model information:", extra={"model_info": model_info})

        # Extract entities
        logger.info("Extracting entities from sample text...")
        entities = ner_model.extract_entities(test_text, document_id="test_doc_001")

        # Display results
        print("\n" + "=" * 80)
        print("EXTRACTED ENTITIES")
        print("=" * 80)

        for i, entity in enumerate(entities, 1):
            print(f"\n{i}. {entity.text}")
            print(f"   Type: {entity.type}")
            print(f"   Position: {entity.start_char}-{entity.end_char}")
            print(f"   Confidence: {entity.confidence:.2%}")
            print(f"   Context: {entity.context}")

        print("\n" + "=" * 80)
        print(f"Total entities extracted: {len(entities)}")
        print("=" * 80)

        # Test batch processing
        logger.info("Testing batch processing...")
        batch_texts = [
            "Apple Inc. announced new products in Cupertino.",
            "The United Nations held a meeting in New York on Friday."
        ]

        batch_results = ner_model.extract_entities_batch(
            batch_texts,
            document_ids=["batch_test_1", "batch_test_2"]
        )

        print("\n" + "=" * 80)
        print("BATCH PROCESSING RESULTS")
        print("=" * 80)

        for i, (text, entities) in enumerate(zip(batch_texts, batch_results), 1):
            print(f"\nDocument {i}: {text[:50]}...")
            print(f"Entities found: {len(entities)}")
            for entity in entities:
                print(f"  - {entity.text} ({entity.type}, confidence: {entity.confidence:.2%})")

        logger.info("NER model test completed successfully")

    except Exception as e:
        log_exception(logger.logger, "NER model test failed")
        sys.exit(1)
