"""
Event consumer backends for Stage 2 NLP Processing Service.

Consumes events from Stage 1 (Cleaning Service) via multiple backends:
- Redis Streams (default, always available)
- Webhook (HTTP push)
- Kafka (optional, requires kafka-python)
- NATS (optional, requires nats-py)
- RabbitMQ (optional, requires pika)
"""
from .base import ConsumerBackend, NullConsumer
from .redis_streams import RedisStreamsConsumer
from .webhook import WebhookConsumer, create_webhook_handler
from .multi import MultiConsumer, create_multi_consumer

# Optional backend availability flags
KAFKA_AVAILABLE = False
NATS_AVAILABLE = False
RABBITMQ_AVAILABLE = False

try:
    from .kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KafkaConsumer = None

try:
    from .nats import NATSConsumer
    NATS_AVAILABLE = True
except ImportError:
    NATSConsumer = None

try:
    from .rabbitmq import RabbitMQConsumer
    RABBITMQ_AVAILABLE = True
except ImportError:
    RabbitMQConsumer = None


def get_available_consumers() -> dict:
    """
    Get dictionary of available consumer backends.

    Returns:
        Dict mapping backend name to availability status
    """
    return {
        "redis_streams": True,  # Always available
        "webhook": True,  # Always available
        "kafka": KAFKA_AVAILABLE,
        "nats": NATS_AVAILABLE,
        "rabbitmq": RABBITMQ_AVAILABLE,
    }


__all__ = [
    # Base classes
    "ConsumerBackend",
    "NullConsumer",
    # Core backends (always available)
    "RedisStreamsConsumer",
    "WebhookConsumer",
    "create_webhook_handler",
    # Optional backends
    "KafkaConsumer",
    "NATSConsumer",
    "RabbitMQConsumer",
    # Multi-backend
    "MultiConsumer",
    "create_multi_consumer",
    # Utility
    "get_available_consumers",
    "KAFKA_AVAILABLE",
    "NATS_AVAILABLE",
    "RABBITMQ_AVAILABLE",
]
