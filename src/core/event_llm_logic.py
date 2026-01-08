"""
event_llm_logic.py

Event extraction using vLLM-optimized LLM inference.
Implements hierarchical event extraction with domain classification for storyline distinction.

Features:
- vLLM integration for 15-25x speedup vs standard inference
- Hierarchical extraction (summary pass + detailed pass)
- Domain-aware prompts for better storyline distinction
- Chunking strategy for long documents
- Batch inference for efficiency
- GPU acceleration with memory optimization
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import torch

# vLLM imports with proper fallback
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None  # Set to None to avoid NameError
    SamplingParams = None  # Set to None to avoid NameError
    logging.warning("vLLM not available. Falling back to HuggingFace transformers.")

# HuggingFace transformers fallback
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.config_manager import get_settings
from src.utils.logger import get_logger, PerformanceLogger
from src.schemas.data_models import (
    Event, EventTrigger, EventArgument, EventMetadata,
    Entity, create_event_id
)

logger = get_logger(__name__)


# =============================================================================
# Event LLM Model
# =============================================================================

class EventLLMModel:
    """
    Event extraction model using vLLM or HuggingFace transformers.

    Optimized for storyline distinction through domain-aware processing.
    """

    def __init__(self):
        """Initialize Event LLM model with vLLM or fallback."""
        self.settings = get_settings().event_llm_service
        self.general_settings = get_settings().general

        self.model = None
        self.tokenizer = None
        self.use_vllm = self.settings.use_vllm and VLLM_AVAILABLE

        # Load model
        self._load_model()

        logger.info(
            "Event LLM model initialized",
            extra={
                "model_name": self.settings.model_name,
                "use_vllm": self.use_vllm,
                "device": self.general_settings.device
            }
        )

    def _load_model(self):
        """Load model with vLLM or HuggingFace."""
        if self.use_vllm:
            self._load_vllm_model()
        else:
            self._load_hf_model()

    def _log_gpu_state(self, phase: str):
        """Log GPU memory state for debugging vLLM initialization."""
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                reserved = torch.cuda.memory_reserved(device) / (1024**3)
                free = total - reserved
                logger.info(
                    f"[GPU STATE - {phase}] Total: {total:.2f}GB, Allocated: {allocated:.2f}GB, "
                    f"Reserved: {reserved:.2f}GB, Free: {free:.2f}GB",
                    extra={
                        "phase": phase,
                        "gpu_total_gb": round(total, 2),
                        "gpu_allocated_gb": round(allocated, 2),
                        "gpu_reserved_gb": round(reserved, 2),
                        "gpu_free_gb": round(free, 2)
                    }
                )
            else:
                logger.warning(f"[GPU STATE - {phase}] CUDA not available")
        except Exception as e:
            logger.error(f"[GPU STATE - {phase}] Failed to get GPU state: {e}")

    def _load_vllm_model(self):
        """Load model with vLLM for optimized inference."""
        import os
        import time

        logger.info("=" * 60)
        logger.info("STARTING vLLM MODEL INITIALIZATION")
        logger.info("=" * 60)

        # Log environment
        logger.info(
            "Environment check",
            extra={
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
                "VLLM_LOGGING_LEVEL": os.environ.get("VLLM_LOGGING_LEVEL", "not set"),
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "not set"),
            }
        )

        self._log_gpu_state("PRE_INIT")

        try:
            # Phase 1: Load tokenizer
            phase_start = time.time()
            logger.info("[PHASE 1/4] Loading and configuring tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.model_name,
                    use_fast=True
                )
            except Exception as tokenizer_error:
                logger.warning(f"Fast tokenizer failed ({tokenizer_error}), trying slow tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.model_name,
                    use_fast=False
                )

            # Configure tokenizer to avoid vLLM generation hang
            # Mistral models don't have a default pad_token, which causes vLLM to hang
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    logger.info(f"Set pad_token to eos_token: {self.tokenizer.eos_token}")
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logger.info("Added new pad_token: [PAD]")

            logger.info(
                f"[PHASE 1/4] Tokenizer loaded in {time.time() - phase_start:.2f}s",
                extra={
                    "vocab_size": len(self.tokenizer),
                    "pad_token": self.tokenizer.pad_token,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token": self.tokenizer.eos_token,
                    "eos_token_id": self.tokenizer.eos_token_id
                }
            )
            self._log_gpu_state("POST_TOKENIZER")

            # Phase 2: Build vLLM arguments
            phase_start = time.time()
            logger.info("[PHASE 2/4] Building vLLM engine arguments...")

            vllm_kwargs = {
                "model": self.settings.model_name,
                "tensor_parallel_size": self.settings.tensor_parallel_size,
                "gpu_memory_utilization": self.settings.gpu_memory_utilization,
                "max_model_len": self.settings.max_model_len,
                "quantization": self.settings.quantization,
                "dtype": self.settings.dtype,
                "swap_space": self.settings.swap_space_gb,
                "trust_remote_code": True,
                "disable_log_stats": False,
                "enforce_eager": True,  # Disable CUDA graphs to prevent initialization hang
            }

            # Explicitly disable prefix caching to avoid conflicts with resource lifecycle
            try:
                import inspect
                from vllm.entrypoints.llm import EngineArgs
                if 'enable_prefix_caching' in inspect.signature(EngineArgs.__init__).parameters:
                    vllm_kwargs["enable_prefix_caching"] = False
                    logger.info("Prefix caching explicitly disabled")
            except Exception as param_check_error:
                logger.debug(f"Could not check for enable_prefix_caching support: {param_check_error}")

            logger.info(
                f"[PHASE 2/4] vLLM kwargs prepared",
                extra={
                    "model": self.settings.model_name,
                    "tensor_parallel_size": vllm_kwargs["tensor_parallel_size"],
                    "gpu_memory_utilization": vllm_kwargs["gpu_memory_utilization"],
                    "max_model_len": vllm_kwargs["max_model_len"],
                    "enforce_eager": vllm_kwargs["enforce_eager"],
                    "quantization": vllm_kwargs["quantization"],
                }
            )

            # Phase 3: Initialize vLLM engine (this is where hang typically occurs)
            phase_start = time.time()
            logger.info("[PHASE 3/4] Initializing vLLM engine (this may take 30-120 seconds)...")
            logger.info(">>> If hang occurs here, check: shared memory, GPU memory, CUDA graphs")

            self.model = LLM(**vllm_kwargs)

            logger.info(f"[PHASE 3/4] vLLM engine initialized in {time.time() - phase_start:.2f}s")
            self._log_gpu_state("POST_ENGINE_INIT")

            # Phase 4: Verify model is ready with a warmup generation
            phase_start = time.time()
            logger.info("[PHASE 4/4] Verifying model with warmup generation...")

            warmup_prompt = "Hello, world!"
            warmup_params = SamplingParams(max_tokens=5, temperature=0.1)
            warmup_output = self.model.generate([warmup_prompt], warmup_params)

            if warmup_output and warmup_output[0].outputs:
                logger.info(
                    f"[PHASE 4/4] Warmup completed in {time.time() - phase_start:.2f}s",
                    extra={"warmup_output_len": len(warmup_output[0].outputs[0].text)}
                )
            else:
                logger.warning("[PHASE 4/4] Warmup completed but no output generated")

            self._log_gpu_state("POST_WARMUP")

            logger.info("=" * 60)
            logger.info("vLLM MODEL INITIALIZATION COMPLETE - SUCCESS")
            logger.info("=" * 60)

        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"vLLM INITIALIZATION FAILED: {e}")
            logger.error("=" * 60, exc_info=True)
            self._log_gpu_state("FAILURE")
            logger.info("Falling back to HuggingFace transformers")
            self.use_vllm = False
            self._load_hf_model()

    def _load_hf_model(self):
        """Load model with HuggingFace transformers (fallback)."""
        logger.info("Loading model with HuggingFace transformers...")

        device = self.general_settings.device if self.general_settings.gpu_enabled else "cpu"

        try:
            # Load tokenizer with fallback to slow tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.model_name,
                    use_fast=True
                )
                logger.debug("Fast tokenizer loaded successfully")
            except Exception as tokenizer_error:
                logger.warning(f"Fast tokenizer failed ({tokenizer_error}), trying slow tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.model_name,
                    use_fast=False
                )
                logger.info("Slow tokenizer loaded successfully")

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )

            if device == "cpu":
                self.model = self.model.to(device)

            self.model.eval()

            logger.info(f"HuggingFace model loaded successfully on {device}")

        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}", exc_info=True)
            raise

    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Chunk long text into overlapping segments.

        Args:
            text: Input text

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_size = self.settings.chunk_size_tokens
        overlap = self.settings.chunk_overlap_tokens

        if len(tokens) <= chunk_size:
            return [(text, 0, len(text))]

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode chunk
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Calculate character positions (approximate)
            char_start = len(self.tokenizer.decode(tokens[:start_idx], skip_special_tokens=True))
            char_end = char_start + len(chunk_text)

            chunks.append((chunk_text, char_start, char_end))

            # Move to next chunk with overlap
            if end_idx >= len(tokens):
                break
            start_idx = end_idx - overlap

        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def extract_events(
        self,
        text: str,
        document_id: str,
        context: Optional[Dict[str, Any]] = None,
        domain_hint: Optional[str] = None
    ) -> List[Event]:
        """
        Extract events from text using LLM.

        Args:
            text: Input text
            document_id: Document identifier
            context: Additional context (title, author, etc.)
            domain_hint: Optional domain hint for focused extraction

        Returns:
            List of extracted events
        """
        logger.info(f"Extracting events from document: {document_id}")

        with PerformanceLogger("event_extraction", logger.logger, document_id=document_id):
            # Import prompts (avoid circular import)
            from src.core.llm_prompts import build_prompt, parse_llm_output

            # Chunk text if needed
            chunks = self.chunk_text(text)

            all_events = []

            for chunk_idx, (chunk_text, char_start, char_end) in enumerate(chunks):
                logger.debug(
                    f"Processing chunk {chunk_idx + 1}/{len(chunks)}",
                    extra={"document_id": document_id, "chunk_size": len(chunk_text)}
                )

                # Build prompt
                prompt = build_prompt(chunk_text, context or {}, domain_hint)

                # Generate
                try:
                    output = self._generate(prompt)
                    events = parse_llm_output(output, document_id, chunk_text, domain_hint)

                    # Adjust character positions for chunk offset
                    for event in events:
                        event.trigger.start_char += char_start
                        event.trigger.end_char += char_start
                        for arg in event.arguments:
                            arg.entity.start_char += char_start
                            arg.entity.end_char += char_start

                        # Store chunk index
                        event.metadata.source_chunk_index = chunk_idx

                    all_events.extend(events)

                except Exception as e:
                    logger.error(
                        f"Failed to extract events from chunk {chunk_idx}",
                        exc_info=True,
                        extra={"document_id": document_id, "chunk_idx": chunk_idx}
                    )

            # Merge overlapping events from chunks
            merged_events = self._merge_overlapping_events(all_events)

            logger.info(
                f"Extracted {len(merged_events)} events from {len(chunks)} chunks",
                extra={"document_id": document_id}
            )

            return merged_events

    def _generate(self, prompt: str) -> str:
        """
        Generate text using vLLM or HuggingFace.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        if self.use_vllm:
            return self._generate_vllm(prompt)
        else:
            return self._generate_hf(prompt)

    def _generate_vllm(self, prompt: str) -> str:
        """Generate with vLLM."""
        # Configure sampling parameters with safeguards against hangs
        sampling_params = SamplingParams(
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            top_k=self.settings.top_k,
            max_tokens=self.settings.max_new_tokens,
            stop=None,  # Disable stop tokens to prevent hang issues
            skip_special_tokens=True,  # Skip special tokens in output
            logprobs=None,  # Disable logprobs for faster generation
        )

        logger.debug(f"Generating with vLLM (max_tokens={self.settings.max_new_tokens})")
        outputs = self.model.generate([prompt], sampling_params)

        generated_text = outputs[0].outputs[0].text
        logger.debug(f"Generated {len(generated_text)} characters")

        return generated_text

    def _generate_hf(self, prompt: str) -> str:
        """Generate with HuggingFace transformers."""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move inputs to the same device as the model
        # Using next(model.parameters()).device ensures compatibility with device_map="auto"
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                top_k=self.settings.top_k,
                do_sample=True if self.settings.temperature > 0 else False
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new generation (remove prompt)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]

        return generated_text

    def _merge_overlapping_events(self, events: List[Event]) -> List[Event]:
        """
        Merge duplicate/overlapping events from different chunks.

        Args:
            events: List of events (may contain duplicates)

        Returns:
            Merged list of events
        """
        if len(events) <= 1:
            return events

        # Group events by approximate position and type
        from collections import defaultdict
        groups = defaultdict(list)

        for event in events:
            key = (
                event.event_type,
                event.trigger.text.lower(),
                event.trigger.start_char // 100  # Group by ~100 char buckets
            )
            groups[key].append(event)

        # Merge each group
        merged = []
        for group_events in groups.values():
            if len(group_events) == 1:
                merged.append(group_events[0])
            else:
                # Take the event with highest confidence
                best_event = max(group_events, key=lambda e: e.metadata.confidence)
                merged.append(best_event)

        return merged

    def extract_events_batch(
        self,
        texts: List[str],
        document_ids: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
        domain_hints: Optional[List[Optional[str]]] = None
    ) -> List[List[Event]]:
        """
        Extract events from multiple texts in batch.

        Args:
            texts: List of input texts
            document_ids: List of document identifiers
            contexts: Optional list of context dictionaries
            domain_hints: Optional list of domain hints

        Returns:
            List of event lists (one per document)
        """
        logger.info(f"Batch extracting events from {len(texts)} documents")

        if contexts is None:
            contexts = [None] * len(texts)
        if domain_hints is None:
            domain_hints = [None] * len(texts)

        results = []

        # Process in smaller batches for vLLM
        batch_size = self.settings.max_batch_size if self.use_vllm else 1

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = document_ids[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            batch_domains = domain_hints[i:i + batch_size]

            if self.use_vllm and len(batch_texts) > 1:
                # True batch processing with vLLM
                batch_results = self._extract_events_batch_vllm(
                    batch_texts, batch_ids, batch_contexts, batch_domains
                )
                results.extend(batch_results)
            else:
                # Sequential processing
                for text, doc_id, ctx, domain in zip(batch_texts, batch_ids, batch_contexts, batch_domains):
                    events = self.extract_events(text, doc_id, ctx, domain)
                    results.append(events)

        return results

    def _extract_events_batch_vllm(
        self,
        texts: List[str],
        document_ids: List[str],
        contexts: List[Optional[Dict[str, Any]]],
        domain_hints: List[Optional[str]]
    ) -> List[List[Event]]:
        """Extract events using vLLM batch inference."""
        from src.core.llm_prompts import build_prompt, parse_llm_output

        # Build prompts
        prompts = [
            build_prompt(text, ctx or {}, domain)
            for text, ctx, domain in zip(texts, contexts, domain_hints)
        ]

        # Generate in batch with safeguards against hangs
        sampling_params = SamplingParams(
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            top_k=self.settings.top_k,
            max_tokens=self.settings.max_new_tokens,
            stop=None,  # Disable stop tokens to prevent hang issues
            skip_special_tokens=True,  # Skip special tokens in output
            logprobs=None,  # Disable logprobs for faster generation
        )

        logger.debug(f"Batch generating for {len(prompts)} documents")
        outputs = self.model.generate(prompts, sampling_params)

        # Parse outputs
        results = []
        for output, doc_id, text, domain in zip(outputs, document_ids, texts, domain_hints):
            try:
                events = parse_llm_output(output.outputs[0].text, doc_id, text, domain)
                results.append(events)
            except Exception as e:
                logger.error(f"Failed to parse output for {doc_id}", exc_info=True)
                results.append([])

        return results


# =============================================================================
# Singleton Instance
# =============================================================================

_model_instance: Optional[EventLLMModel] = None


def get_event_llm_model() -> EventLLMModel:
    """
    Get singleton Event LLM model instance.

    Returns:
        EventLLMModel instance
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = EventLLMModel()
    return _model_instance


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Test model loading
    try:
        model = get_event_llm_model()
        print(f"Model loaded successfully. Using vLLM: {model.use_vllm}")

        # Test text chunking
        test_text = "This is a test. " * 200  # Create long text
        chunks = model.chunk_text(test_text)
        print(f"Chunked text into {len(chunks)} chunks")

    except Exception as e:
        print(f"Failed to test model: {e}")
        sys.exit(1)
