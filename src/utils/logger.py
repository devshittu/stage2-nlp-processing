"""
logger.py

Structured logging utility for Stage 2 NLP Processing Service.
Provides JSON-formatted logging with context enrichment and proper error handling.

Features:
- JSON-formatted logs for easy parsing and analysis
- Context enrichment (request IDs, document IDs, timestamps)
- Automatic rotation and compression
- Performance metrics logging
- Error tracking with stack traces
- Integration with monitoring systems
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import traceback


# =============================================================================
# Custom JSON Formatter
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format.

    Each log record includes:
    - timestamp: ISO 8601 formatted timestamp
    - level: Log level (INFO, WARNING, ERROR, etc.)
    - logger: Logger name
    - message: Log message
    - extra: Any additional context fields
    - exc_info: Exception information if present
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add process and thread info
        log_entry["process_id"] = record.process
        log_entry["thread_id"] = record.thread

        # Add any extra fields from the record
        # These are passed via logger.info(..., extra={...})
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                # Skip built-in attributes
                if key not in [
                    'name', 'msg', 'args', 'created', 'msecs', 'levelname',
                    'levelno', 'pathname', 'filename', 'module', 'exc_info',
                    'exc_text', 'stack_info', 'lineno', 'funcName', 'process',
                    'processName', 'thread', 'threadName', 'getMessage',
                    'message', 'asctime', 'relativeCreated'
                ]:
                    try:
                        # Attempt to serialize the value
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        # If not serializable, convert to string
                        log_entry[key] = str(value)

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }

        # Add stack info if present
        if record.stack_info:
            log_entry["stack_info"] = record.stack_info

        return json.dumps(log_entry, ensure_ascii=False)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "json",
    enable_console: bool = True,
    enable_file: bool = True,
    rotation_size: str = "100MB",
    backup_count: int = 10
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, uses default from config.
        log_format: Format type ("json" or "text")
        enable_console: Enable console logging
        enable_file: Enable file logging
        rotation_size: Log file rotation size (e.g., "100MB")
        backup_count: Number of backup files to keep

    Example:
        setup_logging(log_level="INFO", log_file="/app/logs/nlp_processing.log")
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Choose formatter
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        # Standard text format
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if enable_file and log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse rotation size (e.g., "100MB" -> 100 * 1024 * 1024 bytes)
        max_bytes = _parse_size(rotation_size)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log initialization
    root_logger.info(
        "Logging initialized",
        extra={
            "log_level": log_level,
            "log_format": log_format,
            "console_enabled": enable_console,
            "file_enabled": enable_file,
            "log_file": log_file
        }
    )


def _parse_size(size_str: str) -> int:
    """
    Parse size string to bytes.

    Args:
        size_str: Size string (e.g., "100MB", "1GB", "500KB")

    Returns:
        Size in bytes

    Example:
        _parse_size("100MB") -> 104857600
    """
    size_str = size_str.upper().strip()

    # Extract number and unit
    import re
    match = re.match(r'(\d+)\s*(KB|MB|GB)?', size_str)
    if not match:
        return 100 * 1024 * 1024  # Default 100MB

    number = int(match.group(1))
    unit = match.group(2)

    if unit == 'KB':
        return number * 1024
    elif unit == 'MB':
        return number * 1024 * 1024
    elif unit == 'GB':
        return number * 1024 * 1024 * 1024
    else:
        return number  # Assume bytes


# =============================================================================
# Logger Factory
# =============================================================================

def get_logger(name: str, **extra_context: Any) -> logging.LoggerAdapter:
    """
    Get a logger with optional context enrichment.

    Args:
        name: Logger name (usually __name__)
        **extra_context: Additional context to include in all log messages

    Returns:
        LoggerAdapter with context

    Example:
        logger = get_logger(__name__, service="ner_service", version="1.0")
        logger.info("Processing document", extra={"document_id": "doc123"})
    """
    logger = logging.getLogger(name)

    if extra_context:
        return logging.LoggerAdapter(logger, extra_context)
    return logging.LoggerAdapter(logger, {})


# =============================================================================
# Performance Logging
# =============================================================================

class PerformanceLogger:
    """
    Context manager for logging performance metrics.

    Automatically logs duration and optional metrics on exit.

    Example:
        with PerformanceLogger("document_processing", logger, document_id="doc123"):
            # ... processing code ...
            pass
        # Logs: {"operation": "document_processing", "duration_ms": 1234.56, "document_id": "doc123"}
    """

    def __init__(
        self,
        operation: str,
        logger: logging.Logger,
        log_level: int = logging.INFO,
        **context: Any
    ):
        """
        Initialize performance logger.

        Args:
            operation: Operation name
            logger: Logger instance
            log_level: Log level for performance message
            **context: Additional context fields
        """
        self.operation = operation
        self.logger = logger
        self.log_level = log_level
        self.context = context
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.utcnow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log performance."""
        self.end_time = datetime.utcnow()
        duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        log_data = {
            "operation": self.operation,
            "duration_ms": round(duration_ms, 2),
            "start_time": self.start_time.isoformat() + "Z",
            "end_time": self.end_time.isoformat() + "Z",
        }
        log_data.update(self.context)

        if exc_type is not None:
            # Operation failed
            log_data["success"] = False
            log_data["error_type"] = exc_type.__name__
            log_data["error_message"] = str(exc_val)
            self.logger.error(
                f"Operation '{self.operation}' failed after {duration_ms:.2f}ms",
                extra=log_data
            )
        else:
            # Operation succeeded
            log_data["success"] = True
            self.logger.log(
                self.log_level,
                f"Operation '{self.operation}' completed in {duration_ms:.2f}ms",
                extra=log_data
            )

        return False  # Don't suppress exceptions


# =============================================================================
# Error Logging Utilities
# =============================================================================

def log_exception(
    logger: logging.Logger,
    message: str,
    exc_info: bool = True,
    **context: Any
) -> None:
    """
    Log an exception with full context and stack trace.

    Args:
        logger: Logger instance
        message: Error message
        exc_info: Include exception info
        **context: Additional context fields

    Example:
        try:
            # ... code ...
        except Exception as e:
            log_exception(logger, "Failed to process document", document_id="doc123")
    """
    logger.error(message, exc_info=exc_info, extra=context)


def log_warning_with_context(
    logger: logging.Logger,
    message: str,
    **context: Any
) -> None:
    """
    Log a warning with context.

    Args:
        logger: Logger instance
        message: Warning message
        **context: Additional context fields
    """
    logger.warning(message, extra=context)


def log_info_with_metrics(
    logger: logging.Logger,
    message: str,
    metrics: Dict[str, Any],
    **context: Any
) -> None:
    """
    Log informational message with metrics.

    Args:
        logger: Logger instance
        message: Info message
        metrics: Metrics dictionary
        **context: Additional context fields

    Example:
        log_info_with_metrics(
            logger,
            "Batch processing complete",
            metrics={"documents_processed": 100, "errors": 2, "duration_seconds": 45.3},
            batch_id="batch123"
        )
    """
    log_data = {"metrics": metrics}
    log_data.update(context)
    logger.info(message, extra=log_data)


# =============================================================================
# Initialization for Configuration Integration
# =============================================================================

def initialize_logging_from_config(config: Optional[Any] = None) -> None:
    """
    Initialize logging from configuration object.

    Args:
        config: Configuration object (Settings from config_manager)
               If None, attempts to load from ConfigManager

    Example:
        from src.utils.config_manager import get_settings
        settings = get_settings()
        initialize_logging_from_config(settings)
    """
    if config is None:
        try:
            from src.utils.config_manager import get_settings
            config = get_settings()
        except Exception as e:
            # Fallback to default logging if config not available
            setup_logging()
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not load config for logging, using defaults: {e}")
            return

    # Extract logging settings
    log_level = config.general.log_level
    log_format = config.monitoring.log_format
    log_file = config.monitoring.log_file
    log_rotation = config.monitoring.log_rotation

    # Setup logging
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        log_format=log_format,
        enable_console=True,
        enable_file=True,
        rotation_size=log_rotation
    )


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    # Test logging setup
    setup_logging(log_level="DEBUG", log_format="json")

    logger = get_logger(__name__, service="test", version="1.0")

    # Test various log levels
    logger.debug("This is a debug message", extra={"test_id": "debug_001"})
    logger.info("This is an info message", extra={"test_id": "info_001"})
    logger.warning("This is a warning message", extra={"test_id": "warn_001"})

    # Test exception logging
    try:
        raise ValueError("This is a test exception")
    except Exception:
        log_exception(
            logger.logger,  # Get underlying logger from adapter
            "Exception occurred during testing",
            document_id="doc123"
        )

    # Test performance logging
    import time
    with PerformanceLogger("test_operation", logger.logger, document_id="doc456"):
        time.sleep(0.1)  # Simulate work

    # Test metrics logging
    log_info_with_metrics(
        logger.logger,
        "Test metrics logged",
        metrics={"items_processed": 100, "errors": 0, "duration_s": 1.5},
        batch_id="batch123"
    )

    print("\nLogging test completed. Check console output above.")
