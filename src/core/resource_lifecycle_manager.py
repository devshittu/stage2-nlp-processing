"""
Resource Lifecycle Management System

Intelligent resource allocation and cleanup for NLP pipeline services.
Monitors GPU VRAM, system RAM, and manages lifecycle of model instances,
cached tensors, and ephemeral processing artifacts.

Ensures efficient resource utilization in shared hardware environments
while maintaining service availability without container restarts.
"""

import gc
import logging
import os
import threading
import time
import weakref
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    GPUtil = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from src.utils.config_manager import get_settings


class CleanupStrategy(str, Enum):
    """Resource cleanup strategies"""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


class ResourceType(str, Enum):
    """Types of resources being managed"""
    GPU_MEMORY = "gpu_memory"
    SYSTEM_MEMORY = "system_memory"
    MODEL_INSTANCE = "model_instance"
    TOKENIZER = "tokenizer"
    CACHE = "cache"
    FILE_HANDLE = "file_handle"
    TENSOR = "tensor"


class ServiceState:
    """Track service activity and idle state"""

    def __init__(self, service_name: str, idle_timeout: int):
        self.service_name = service_name
        self.idle_timeout = idle_timeout
        self.last_activity = datetime.now()
        self.is_idle = False
        self.task_count = 0
        self._lock = threading.Lock()

    def mark_activity(self):
        """Record service activity"""
        with self._lock:
            self.last_activity = datetime.now()
            self.is_idle = False
            self.task_count += 1

    def check_idle(self) -> bool:
        """Check if service has been idle beyond timeout"""
        with self._lock:
            elapsed = (datetime.now() - self.last_activity).total_seconds()
            self.is_idle = elapsed > self.idle_timeout
            return self.is_idle

    def get_idle_duration(self) -> float:
        """Get duration of idle time in seconds"""
        with self._lock:
            return (datetime.now() - self.last_activity).total_seconds()


class ResourceTracker:
    """Track allocated resources for cleanup"""

    def __init__(self):
        self._resources: Dict[str, Set[weakref.ref]] = {
            ResourceType.MODEL_INSTANCE: set(),
            ResourceType.TOKENIZER: set(),
            ResourceType.CACHE: set(),
            ResourceType.FILE_HANDLE: set(),
            ResourceType.TENSOR: set(),
        }
        self._lock = threading.Lock()

    def register(self, resource_type: ResourceType, resource: Any) -> None:
        """Register a resource for tracking"""
        with self._lock:
            try:
                ref = weakref.ref(resource)
                self._resources[resource_type].add(ref)
            except TypeError:
                # Object doesn't support weak references
                pass

    def get_active_count(self, resource_type: ResourceType) -> int:
        """Get count of active resources of given type"""
        with self._lock:
            # Clean up dead references
            self._resources[resource_type] = {
                ref for ref in self._resources[resource_type]
                if ref() is not None
            }
            return len(self._resources[resource_type])

    def clear(self, resource_type: ResourceType) -> None:
        """Clear tracked resources of given type"""
        with self._lock:
            self._resources[resource_type].clear()


class MemoryMonitor:
    """Monitor GPU VRAM and system RAM usage"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._gpu_initialized = False
        self._init_gpu_monitoring()

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring library"""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._gpu_initialized = True
                self.logger.info("GPU monitoring initialized with pynvml")
            except Exception as e:
                self.logger.warning(f"Failed to initialize pynvml: {e}")
                self._gpu_initialized = False

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """
        Get GPU memory usage statistics

        Returns:
            Dict with keys: used_mb, total_mb, used_percent
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"used_mb": 0.0, "total_mb": 0.0, "used_percent": 0.0}

        try:
            if self._gpu_initialized and PYNVML_AVAILABLE:
                # Use pynvml for accurate stats
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mb = info.used / (1024 ** 2)
                total_mb = info.total / (1024 ** 2)
                used_percent = (info.used / info.total) * 100
            elif GPUTIL_AVAILABLE:
                # Fallback to GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    used_mb = gpu.memoryUsed
                    total_mb = gpu.memoryTotal
                    used_percent = gpu.memoryUtil * 100
                else:
                    return {"used_mb": 0.0, "total_mb": 0.0, "used_percent": 0.0}
            else:
                # Fallback to PyTorch stats
                used_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
                total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                used_percent = (used_mb / total_mb) * 100

            return {
                "used_mb": used_mb,
                "total_mb": total_mb,
                "used_percent": used_percent
            }
        except Exception as e:
            self.logger.error(f"Error getting GPU memory usage: {e}")
            return {"used_mb": 0.0, "total_mb": 0.0, "used_percent": 0.0}

    def get_system_memory_usage(self) -> Dict[str, float]:
        """
        Get system RAM usage statistics

        Returns:
            Dict with keys: used_mb, total_mb, used_percent
        """
        if psutil is None:
            self.logger.warning("psutil not available, cannot monitor system memory")
            return {"used_mb": 0.0, "total_mb": 0.0, "used_percent": 0.0}

        try:
            mem = psutil.virtual_memory()
            return {
                "used_mb": mem.used / (1024 ** 2),
                "total_mb": mem.total / (1024 ** 2),
                "used_percent": mem.percent
            }
        except Exception as e:
            self.logger.error(f"Error getting system memory usage: {e}")
            return {"used_mb": 0.0, "total_mb": 0.0, "used_percent": 0.0}

    def check_memory_pressure(
        self,
        gpu_threshold: float = 85.0,
        system_threshold: float = 80.0
    ) -> Dict[str, bool]:
        """
        Check if memory usage exceeds thresholds

        Args:
            gpu_threshold: GPU memory threshold percentage
            system_threshold: System memory threshold percentage

        Returns:
            Dict with keys: gpu_pressure, system_pressure
        """
        gpu_stats = self.get_gpu_memory_usage()
        sys_stats = self.get_system_memory_usage()

        return {
            "gpu_pressure": gpu_stats["used_percent"] >= gpu_threshold,
            "system_pressure": sys_stats["used_percent"] >= system_threshold,
            "gpu_usage": gpu_stats["used_percent"],
            "system_usage": sys_stats["used_percent"]
        }

    def __del__(self):
        """Cleanup GPU monitoring on deletion"""
        if self._gpu_initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class ResourceLifecycleManager:
    """
    Singleton manager for intelligent resource lifecycle management.

    Monitors memory usage, tracks idle time, and coordinates cleanup
    of ephemeral resources while preserving persistent infrastructure.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self.settings = get_settings()
        self.config = self.settings.resource_lifecycle
        self.logger = logging.getLogger(__name__)

        # Core components
        self.memory_monitor = MemoryMonitor()
        self.resource_tracker = ResourceTracker()

        # Service states
        self.service_states: Dict[str, ServiceState] = {}
        self._init_service_states()

        # Monitoring thread
        self._monitoring_active = False
        self._monitoring_thread = None

        # Cleanup callbacks
        self._cleanup_callbacks: Dict[str, List[Callable]] = {}

        # Metrics
        self.metrics = {
            "total_cleanups": 0,
            "gpu_cleanups": 0,
            "memory_pressure_events": 0,
            "idle_cleanups": 0
        }

        self._initialized = True

        if self.config.enabled:
            self.start_monitoring()
            self.logger.info("ResourceLifecycleManager initialized and monitoring started")

    def _init_service_states(self):
        """Initialize service state trackers"""
        for service_name, config in self.config.idle_timeouts.items():
            if config.enabled:
                self.service_states[service_name] = ServiceState(
                    service_name=service_name,
                    idle_timeout=config.timeout_seconds
                )

    def mark_service_activity(self, service_name: str):
        """Record activity for a service"""
        if service_name in self.service_states:
            self.service_states[service_name].mark_activity()

            if self.config.logging.log_allocation_events:
                self.logger.debug(f"Service activity recorded: {service_name}")

    def register_cleanup_callback(
        self,
        service_name: str,
        callback: Callable,
        priority: int = 0
    ):
        """
        Register a cleanup callback for a service

        Args:
            service_name: Name of the service
            callback: Cleanup function to call
            priority: Priority (higher = earlier execution)
        """
        if service_name not in self._cleanup_callbacks:
            self._cleanup_callbacks[service_name] = []

        self._cleanup_callbacks[service_name].append((priority, callback))
        self._cleanup_callbacks[service_name].sort(key=lambda x: x[0], reverse=True)

        self.logger.debug(f"Registered cleanup callback for {service_name}")

    def cleanup_service(
        self,
        service_name: str,
        strategy: Optional[CleanupStrategy] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Execute cleanup for a service

        Args:
            service_name: Name of service to cleanup
            strategy: Cleanup strategy (overrides config)
            force: Force cleanup regardless of idle state

        Returns:
            Dict with cleanup results
        """
        if not self.config.enabled:
            return {"status": "disabled"}

        # Get strategy from config or parameter
        if strategy is None:
            service_config = self.config.idle_timeouts.get(service_name)
            if service_config:
                strategy_name = service_config.cleanup_strategy
            else:
                strategy_name = "balanced"
            strategy = CleanupStrategy(strategy_name)

        # Check if cleanup needed
        state = self.service_states.get(service_name)
        if not force and state and not state.check_idle():
            return {
                "status": "skipped",
                "reason": "service not idle",
                "idle_duration": state.get_idle_duration()
            }

        self.logger.info(
            f"Starting {strategy.value} cleanup for {service_name}"
        )

        start_time = time.time()
        results = {
            "status": "success",
            "service": service_name,
            "strategy": strategy.value,
            "actions_performed": []
        }

        try:
            # Get strategy config
            strategy_config = self.config.cleanup_strategies[strategy.value]

            # Execute cleanup callbacks
            if service_name in self._cleanup_callbacks:
                for priority, callback in self._cleanup_callbacks[service_name]:
                    try:
                        callback(strategy_config)
                        results["actions_performed"].append(
                            f"callback_{callback.__name__}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Cleanup callback error for {service_name}: {e}",
                            exc_info=True
                        )

            # GPU cache cleanup
            if strategy_config.clear_gpu_cache:
                self._clear_gpu_cache()
                results["actions_performed"].append("clear_gpu_cache")

            # Force garbage collection
            if strategy_config.force_garbage_collection:
                collected = gc.collect()
                results["actions_performed"].append(
                    f"garbage_collection_{collected}_objects"
                )

            # Update metrics
            self.metrics["total_cleanups"] += 1
            if "clear_gpu_cache" in results["actions_performed"]:
                self.metrics["gpu_cleanups"] += 1

            duration = time.time() - start_time
            results["duration_seconds"] = duration

            if self.config.logging.log_release_events:
                self.logger.info(
                    f"Cleanup completed for {service_name} in {duration:.2f}s: "
                    f"{', '.join(results['actions_performed'])}"
                )

        except Exception as e:
            self.logger.error(
                f"Cleanup failed for {service_name}: {e}",
                exc_info=True
            )
            results["status"] = "failed"
            results["error"] = str(e)

        return results

    def _clear_gpu_cache(self):
        """Clear GPU cache (PyTorch)"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.config.logging.log_release_events:
                self.logger.debug("GPU cache cleared")

    def check_and_cleanup_on_pressure(self) -> bool:
        """
        Check memory pressure and trigger cleanup if needed

        Returns:
            True if cleanup was performed
        """
        if not self.config.memory_monitoring.enabled:
            return False

        pressure = self.memory_monitor.check_memory_pressure(
            gpu_threshold=self.config.memory_monitoring.gpu_memory_threshold_percent,
            system_threshold=self.config.memory_monitoring.system_memory_threshold_percent
        )

        if pressure["gpu_pressure"] or pressure["system_pressure"]:
            if self.config.logging.log_memory_pressure:
                self.logger.warning(
                    f"Memory pressure detected - GPU: {pressure['gpu_usage']:.1f}%, "
                    f"System: {pressure['system_usage']:.1f}%"
                )

            self.metrics["memory_pressure_events"] += 1

            # Trigger aggressive cleanup on all services
            for service_name in self.service_states.keys():
                self.cleanup_service(
                    service_name,
                    strategy=CleanupStrategy.AGGRESSIVE,
                    force=True
                )

            return True

        return False

    def _monitoring_loop(self):
        """Background monitoring loop"""
        check_interval = self.config.memory_monitoring.check_interval_seconds

        while self._monitoring_active:
            try:
                # Check memory pressure
                self.check_and_cleanup_on_pressure()

                # Check idle services
                for service_name, state in self.service_states.items():
                    if state.check_idle():
                        self.cleanup_service(service_name)
                        self.metrics["idle_cleanups"] += 1

                # Log peak usage if enabled
                if self.config.logging.log_peak_usage:
                    gpu_stats = self.memory_monitor.get_gpu_memory_usage()
                    if gpu_stats["used_percent"] > 0:
                        self.logger.debug(
                            f"GPU VRAM: {gpu_stats['used_mb']:.0f}MB / "
                            f"{gpu_stats['total_mb']:.0f}MB "
                            f"({gpu_stats['used_percent']:.1f}%)"
                        )

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)

            time.sleep(check_interval)

    def start_monitoring(self):
        """Start background monitoring thread"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceMonitoring"
        )
        self._monitoring_thread.start()
        self.logger.info("Resource monitoring thread started")

    def stop_monitoring(self):
        """Stop background monitoring thread"""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("Resource monitoring thread stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        gpu_stats = self.memory_monitor.get_gpu_memory_usage()
        sys_stats = self.memory_monitor.get_system_memory_usage()

        return {
            **self.metrics,
            "gpu_memory_mb": gpu_stats["used_mb"],
            "gpu_memory_percent": gpu_stats["used_percent"],
            "system_memory_mb": sys_stats["used_mb"],
            "system_memory_percent": sys_stats["used_percent"],
            "service_states": {
                name: {
                    "idle": state.is_idle,
                    "idle_duration": state.get_idle_duration(),
                    "task_count": state.task_count
                }
                for name, state in self.service_states.items()
            }
        }

    @contextmanager
    def track_task(self, service_name: str):
        """
        Context manager to track task execution and mark activity

        Usage:
            with resource_manager.track_task("event_llm_service"):
                # Process task
                result = process_document(doc)
        """
        self.mark_service_activity(service_name)

        try:
            yield
        finally:
            # Mark activity on completion (successful or not)
            self.mark_service_activity(service_name)


# Singleton instance getter
_manager_instance = None


def get_resource_manager() -> ResourceLifecycleManager:
    """Get singleton ResourceLifecycleManager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ResourceLifecycleManager()
    return _manager_instance
