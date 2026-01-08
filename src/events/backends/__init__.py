"""
Event backend implementations for publishing events.

Available backends:
- RedisStreamsBackend: Redis Streams (default, always available)
- WebhookBackend: HTTP webhook callbacks (always available)
- KafkaBackend: Apache Kafka (requires kafka-python)
- NATSBackend: NATS messaging (requires nats-py)
- RabbitMQBackend: RabbitMQ (requires pika)
- MultiBackend: Publish to multiple backends simultaneously
"""
from .base import EventBackend, NullBackend
from .redis_streams import RedisStreamsBackend
from .webhook import WebhookBackend
from .multi import MultiBackend

# Conditional imports for optional backends
try:
    from .kafka import KafkaBackend, KAFKA_AVAILABLE
except ImportError:
    KafkaBackend = None
    KAFKA_AVAILABLE = False

try:
    from .nats import NATSBackend, NATS_AVAILABLE
except ImportError:
    NATSBackend = None
    NATS_AVAILABLE = False

try:
    from .rabbitmq import RabbitMQBackend, RABBITMQ_AVAILABLE
except ImportError:
    RabbitMQBackend = None
    RABBITMQ_AVAILABLE = False

__all__ = [
    # Base classes
    "EventBackend",
    "NullBackend",
    # Core backends (always available)
    "RedisStreamsBackend",
    "WebhookBackend",
    # Optional backends
    "KafkaBackend",
    "NATSBackend",
    "RabbitMQBackend",
    # Multi-backend
    "MultiBackend",
    # Availability flags
    "KAFKA_AVAILABLE",
    "NATS_AVAILABLE",
    "RABBITMQ_AVAILABLE",
]


def get_available_backends() -> dict:
    """
    Get dictionary of available backends and their status.

    Returns:
        Dictionary mapping backend name to availability boolean
    """
    return {
        "redis_streams": True,  # Always available
        "webhook": True,  # Always available
        "kafka": KAFKA_AVAILABLE,
        "nats": NATS_AVAILABLE,
        "rabbitmq": RABBITMQ_AVAILABLE,
    }
