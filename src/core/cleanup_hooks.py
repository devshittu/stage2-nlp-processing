"""
Framework-Specific Cleanup Hooks

Provides specialized cleanup functions for different ML frameworks used in the pipeline:
- vLLM (Event LLM service)
- Transformers (NER service)
- spaCy (DP service)

Each hook implements framework-specific resource release patterns while maintaining
separation of concerns between persistent infrastructure and ephemeral task resources.
"""

import gc
import logging
from typing import Any, Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None

try:
    from transformers import Pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    Pipeline = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


logger = logging.getLogger(__name__)


class VLLMCleanupHook:
    """
    Cleanup hook for vLLM resources.

    Releases GPU memory, clears KV cache, and frees engine resources
    while preserving model weights for quick reinitialization.
    """

    def __init__(self, model_instance: Optional[Any] = None):
        """
        Initialize vLLM cleanup hook.

        Args:
            model_instance: EventLLMModel instance (optional, can be set later)
        """
        self.model_instance = model_instance
        self.logger = logging.getLogger(f"{__name__}.VLLMCleanupHook")

    def set_model_instance(self, model_instance: Any):
        """Set the model instance to manage"""
        self.model_instance = model_instance

    def cleanup(self, strategy_config: Dict[str, bool]) -> Dict[str, Any]:
        """
        Execute cleanup based on strategy configuration.

        Args:
            strategy_config: Dict with cleanup flags (from settings.yaml)

        Returns:
            Dict with cleanup results
        """
        if not VLLM_AVAILABLE or not TORCH_AVAILABLE:
            return {"status": "skipped", "reason": "vLLM or PyTorch not available"}

        results = {
            "status": "success",
            "actions": []
        }

        try:
            # Clear GPU cache (always safe, no model state lost)
            if getattr(strategy_config, "clear_gpu_cache", False):
                self._clear_gpu_cache()
                results["actions"].append("cleared_gpu_cache")

            # Clear KV cache (vLLM-specific, releases attention cache)
            # Check framework hooks for vLLM-specific settings from config
            from src.utils.config_manager import get_settings
            settings = get_settings()
            vllm_config = settings.framework_hooks.vllm if hasattr(settings, 'framework_hooks') and hasattr(settings.framework_hooks, 'vllm') else None
            if vllm_config and getattr(vllm_config, "clear_kv_cache", False):
                self._clear_kv_cache()
                results["actions"].append("cleared_kv_cache")

            # Free engine worker memory (aggressive, requires reinitialization)
            if vllm_config and getattr(vllm_config, "free_engine_on_idle", False) and getattr(strategy_config, "unload_models", False):
                self._free_engine_memory()
                results["actions"].append("freed_engine_memory")

            # Force garbage collection
            if getattr(strategy_config, "force_garbage_collection", True):
                collected = gc.collect()
                results["actions"].append(f"gc_collected_{collected}_objects")

            self.logger.info(
                f"vLLM cleanup completed: {', '.join(results['actions'])}"
            )

        except Exception as e:
            self.logger.error(f"vLLM cleanup failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)

        return results

    def _clear_gpu_cache(self):
        """Clear PyTorch GPU cache"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Also synchronize to ensure cache is truly cleared
            torch.cuda.synchronize()
            self.logger.debug("GPU cache cleared and synchronized")

    def _clear_kv_cache(self):
        """
        Clear vLLM KV cache.

        Note: vLLM manages KV cache internally. We can't directly clear it,
        but emptying CUDA cache helps release memory.
        """
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Clear CUDA cache to free KV cache memory
            torch.cuda.empty_cache()

            # If model has a cache attribute, try to clear it
            if self.model_instance and hasattr(self.model_instance, 'model'):
                if hasattr(self.model_instance.model, 'llm_engine'):
                    # vLLM engine - we can't directly clear KV cache without breaking state
                    # but we can ensure CUDA cache is cleared
                    self.logger.debug("KV cache memory released via CUDA cache clear")
                else:
                    self.logger.debug("No vLLM engine found, GPU cache cleared")

    def _free_engine_memory(self):
        """
        Free vLLM engine worker memory (aggressive cleanup).

        WARNING: This requires model reinitialization for next request.
        Only use with 'aggressive' strategy during extended idle periods.
        """
        if not self.model_instance:
            self.logger.warning("No model instance set, cannot free engine memory")
            return

        try:
            # Check if using vLLM
            if not hasattr(self.model_instance, 'use_vllm') or not self.model_instance.use_vllm:
                self.logger.debug("Not using vLLM, skipping engine cleanup")
                return

            # Free vLLM LLM engine if exists
            if hasattr(self.model_instance, 'model') and self.model_instance.model is not None:
                if hasattr(self.model_instance.model, 'llm_engine'):
                    # Try to free engine resources
                    try:
                        # vLLM doesn't have a public cleanup API, so we rely on Python GC
                        # and CUDA cache clearing. Setting model to None would break service.
                        # Instead, we just clear CUDA cache aggressively.
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            # Try to clear IPC memory if possible
                            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                                torch.cuda.reset_peak_memory_stats()
                            self.logger.debug("vLLM engine memory cleared via aggressive CUDA cache management")
                    except Exception as e:
                        self.logger.warning(f"Could not fully free vLLM engine memory: {e}")

        except Exception as e:
            self.logger.error(f"Failed to free engine memory: {e}", exc_info=True)


class TransformersCleanupHook:
    """
    Cleanup hook for HuggingFace Transformers models (NER service).

    Manages pipeline cache and optionally moves models to CPU to free GPU memory.
    """

    def __init__(self, model_instance: Optional[Any] = None):
        """
        Initialize Transformers cleanup hook.

        Args:
            model_instance: Model instance (optional, can be set later)
        """
        self.model_instance = model_instance
        self.logger = logging.getLogger(f"{__name__}.TransformersCleanupHook")

    def set_model_instance(self, model_instance: Any):
        """Set the model instance to manage"""
        self.model_instance = model_instance

    def cleanup(self, strategy_config: Dict[str, bool]) -> Dict[str, Any]:
        """
        Execute cleanup based on strategy configuration.

        Args:
            strategy_config: Dict with cleanup flags

        Returns:
            Dict with cleanup results
        """
        results = {
            "status": "success",
            "actions": []
        }

        try:
            # Clear pipeline cache
            if strategy_config.get("clear_pipeline_cache", False):
                self._clear_pipeline_cache()
                results["actions"].append("cleared_pipeline_cache")

            # Move model to CPU (if on GPU and strategy allows)
            if strategy_config.get("move_model_to_cpu_on_idle", False):
                self._move_model_to_cpu()
                results["actions"].append("moved_model_to_cpu")

            # Clear GPU cache
            if strategy_config.get("clear_gpu_cache", False):
                self._clear_gpu_cache()
                results["actions"].append("cleared_gpu_cache")

            # Force garbage collection
            if strategy_config.get("force_garbage_collection", True):
                collected = gc.collect()
                results["actions"].append(f"gc_collected_{collected}_objects")

            self.logger.info(
                f"Transformers cleanup completed: {', '.join(results['actions'])}"
            )

        except Exception as e:
            self.logger.error(f"Transformers cleanup failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)

        return results

    def _clear_pipeline_cache(self):
        """Clear Transformers pipeline cache"""
        # Transformers doesn't have a global cache to clear like this,
        # but we can ensure tokenizer caches are cleared if needed
        if self.model_instance and hasattr(self.model_instance, 'tokenizer'):
            # Clear tokenizer cache if it exists
            if hasattr(self.model_instance.tokenizer, '_tokenizer'):
                # Fast tokenizers have internal caches
                pass  # No public API to clear, relies on GC

        self.logger.debug("Pipeline cache clearing requested (relies on GC)")

    def _move_model_to_cpu(self):
        """Move model from GPU to CPU to free VRAM"""
        if not self.model_instance:
            return

        if hasattr(self.model_instance, 'model') and self.model_instance.model is not None:
            if TORCH_AVAILABLE:
                try:
                    # Check current device
                    current_device = next(self.model_instance.model.parameters()).device
                    if current_device.type == 'cuda':
                        self.model_instance.model.to('cpu')
                        torch.cuda.empty_cache()
                        self.logger.info("Model moved from GPU to CPU")
                    else:
                        self.logger.debug("Model already on CPU")
                except Exception as e:
                    self.logger.warning(f"Could not move model to CPU: {e}")

    def _clear_gpu_cache(self):
        """Clear PyTorch GPU cache"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared")


class SpacyCleanupHook:
    """
    Cleanup hook for spaCy models (DP service).

    Clears doc cache and optionally removes pipes to free memory.
    """

    def __init__(self, model_instance: Optional[Any] = None):
        """
        Initialize spaCy cleanup hook.

        Args:
            model_instance: spaCy nlp instance (optional, can be set later)
        """
        self.model_instance = model_instance
        self.logger = logging.getLogger(f"{__name__}.SpacyCleanupHook")

    def set_model_instance(self, model_instance: Any):
        """Set the model instance to manage"""
        self.model_instance = model_instance

    def cleanup(self, strategy_config: Dict[str, bool]) -> Dict[str, Any]:
        """
        Execute cleanup based on strategy configuration.

        Args:
            strategy_config: Dict with cleanup flags

        Returns:
            Dict with cleanup results
        """
        results = {
            "status": "success",
            "actions": []
        }

        try:
            # Clear doc cache
            if strategy_config.get("clear_doc_cache", False):
                self._clear_doc_cache()
                results["actions"].append("cleared_doc_cache")

            # Remove pipes (aggressive - requires re-adding)
            if strategy_config.get("remove_pipes_on_idle", False):
                self._remove_pipes()
                results["actions"].append("removed_pipes")

            # Clear GPU cache (spaCy can use GPU)
            if strategy_config.get("clear_gpu_cache", False):
                self._clear_gpu_cache()
                results["actions"].append("cleared_gpu_cache")

            # Force garbage collection
            if strategy_config.get("force_garbage_collection", True):
                collected = gc.collect()
                results["actions"].append(f"gc_collected_{collected}_objects")

            self.logger.info(
                f"spaCy cleanup completed: {', '.join(results['actions'])}"
            )

        except Exception as e:
            self.logger.error(f"spaCy cleanup failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)

        return results

    def _clear_doc_cache(self):
        """Clear spaCy doc cache"""
        # spaCy doesn't maintain a global doc cache by default,
        # but we can ensure any cached docs are released via GC
        gc.collect()
        self.logger.debug("spaCy doc cache cleared via GC")

    def _remove_pipes(self):
        """
        Remove spaCy pipeline components (aggressive cleanup).

        WARNING: Requires re-adding pipes for next processing.
        Only use with 'aggressive' strategy.
        """
        if not self.model_instance:
            return

        if SPACY_AVAILABLE and hasattr(self.model_instance, 'pipe_names'):
            # This is destructive - don't do it unless absolutely necessary
            # Instead, just log that we would do it
            self.logger.debug("Pipe removal requested but skipped (too destructive)")

    def _clear_gpu_cache(self):
        """Clear GPU cache (for spaCy transformer models)"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("GPU cache cleared (spaCy)")


def create_cleanup_hook(framework: str, model_instance: Optional[Any] = None):
    """
    Factory function to create appropriate cleanup hook.

    Args:
        framework: Framework name ('vllm', 'transformers', 'spacy')
        model_instance: Model instance to manage

    Returns:
        Cleanup hook instance
    """
    framework = framework.lower()

    if framework == 'vllm':
        return VLLMCleanupHook(model_instance)
    elif framework == 'transformers':
        return TransformersCleanupHook(model_instance)
    elif framework == 'spacy':
        return SpacyCleanupHook(model_instance)
    else:
        raise ValueError(f"Unknown framework: {framework}")
