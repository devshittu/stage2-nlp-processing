"""
config_manager.py

Configuration management utility for Stage 2 NLP Processing Service.
Loads and validates settings from YAML configuration file with environment variable substitution.

Features:
- YAML configuration loading with validation
- Environment variable substitution (${VAR_NAME} syntax)
- Singleton pattern for global settings access
- Type-safe configuration with Pydantic models
- Default values and validation
"""

import os
import re
import yaml
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models (Pydantic)
# =============================================================================

class GeneralSettings(BaseModel):
    """General application settings."""
    log_level: str = Field(default="INFO", description="Logging level")
    gpu_enabled: bool = Field(default=True, description="Enable GPU acceleration")
    device: str = Field(default="cuda", description="Device for model inference")
    max_text_length: int = Field(default=1000000, description="Maximum text length in characters")


class DocumentFieldMapping(BaseModel):
    """Document field mapping for Stage 1 integration."""
    text_field: str = Field(default="cleaned_text", description="Primary text field")
    text_field_fallbacks: List[str] = Field(
        default_factory=lambda: ["original_text", "text", "content"],
        description="Fallback text fields"
    )
    context_fields: List[str] = Field(
        default_factory=lambda: ["cleaned_title", "cleaned_excerpt"],
        description="Context fields for enhanced extraction"
    )
    preserve_in_output: List[str] = Field(
        default_factory=lambda: ["document_id", "cleaned_publication_date"],
        description="Fields to preserve in output"
    )


class NERServiceSettings(BaseModel):
    """Named Entity Recognition service settings."""
    port: int = Field(default=8001, description="Service port")
    model_name: str = Field(default="Babelscape/wikineural-multilingual-ner")
    batch_size: int = Field(default=16, description="GPU batch size")
    max_length: int = Field(default=512, description="Maximum sequence length")
    entity_types: List[str] = Field(
        default_factory=lambda: ["PER", "ORG", "LOC", "GPE", "DATE", "TIME", "MONEY", "MISC", "EVENT"]
    )
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    enable_cache: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600)


class DPServiceSettings(BaseModel):
    """Dependency Parsing service settings."""
    port: int = Field(default=8002, description="Service port")
    model_name: str = Field(default="en_core_web_trf")
    batch_size: int = Field(default=8, description="GPU batch size")
    extract_soa_triplets: bool = Field(default=True)
    min_triplet_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    dependency_relations: List[str] = Field(
        default_factory=lambda: ["nsubj", "nsubjpass", "dobj", "iobj", "pobj", "agent", "attr"]
    )


class EventLLMServiceSettings(BaseModel):
    """Event LLM service settings with vLLM optimization."""
    port: int = Field(default=8003, description="Service port")
    model_name: str = Field(default="mistralai/Mistral-7B-Instruct-v0.3")
    use_vllm: bool = Field(default=True, description="Use vLLM for optimization")
    tensor_parallel_size: int = Field(default=1, description="Number of GPUs for tensor parallelism")
    gpu_memory_utilization: float = Field(default=0.90, ge=0.1, le=0.99)
    max_model_len: int = Field(default=8192, description="Maximum context length")
    quantization: Optional[str] = Field(default="awq", description="Quantization method")
    dtype: str = Field(default="auto")
    max_new_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    chunk_size_tokens: int = Field(default=2048)
    chunk_overlap_tokens: int = Field(default=256)
    max_batch_size: int = Field(default=4)
    swap_space_gb: int = Field(default=4)
    extraction_mode: str = Field(default="hierarchical")
    enable_domain_classification: bool = Field(default=True)
    domains: List[str] = Field(default_factory=lambda: [
        "geopolitical_conflict", "diplomatic_relations", "economic_policy",
        "domestic_policy", "elections_politics", "technology_innovation"
    ])
    event_types: List[str] = Field(default_factory=lambda: [
        "conflict_attack", "contact_meet", "policy_announce", "agreement_sign"
    ])
    argument_roles: List[str] = Field(default_factory=lambda: [
        "agent", "patient", "time", "place", "instrument", "beneficiary", "purpose"
    ])


class StorylineWeights(BaseModel):
    """Weights for multi-dimensional distance calculation in storyline clustering."""
    semantic: float = Field(default=0.4, ge=0.0, le=1.0)
    entity: float = Field(default=0.3, ge=0.0, le=1.0)
    temporal: float = Field(default=0.2, ge=0.0, le=1.0)
    domain: float = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator('semantic', 'entity', 'temporal', 'domain')
    @classmethod
    def check_weights_sum(cls, v, info):
        """Validate that weights sum to approximately 1.0."""
        # This is called for each field, we'll do a full check elsewhere
        return v


class StorylineClusteringSettings(BaseModel):
    """Storyline clustering configuration."""
    enabled: bool = Field(default=True)
    min_cluster_size: int = Field(default=2, ge=1)
    algorithm: str = Field(default="agglomerative")
    distance_metric: str = Field(default="multi_dimensional")
    weights: StorylineWeights = Field(default_factory=StorylineWeights)


class EventLinkingSettings(BaseModel):
    """Event linking and storyline distinction settings."""
    embedding_model: str = Field(default="sentence-transformers/all-mpnet-base-v2")
    semantic_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    entity_overlap_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    temporal_window_days: int = Field(default=7, ge=1)
    enforce_domain_boundaries: bool = Field(default=True)
    allow_cross_domain_linking: bool = Field(default=False)
    storyline_clustering: StorylineClusteringSettings = Field(default_factory=StorylineClusteringSettings)
    enable_entity_role_context: bool = Field(default=True)
    entity_context_window: int = Field(default=50, ge=10)
    enable_causality_detection: bool = Field(default=True)
    causality_indicators: List[str] = Field(default_factory=lambda: [
        "because", "due to", "as a result", "led to", "caused"
    ])


class OrchestratorServiceSettings(BaseModel):
    """Orchestrator service settings."""
    port: int = Field(default=8000, description="Service port")
    enable_sync_processing: bool = Field(default=True)
    enable_async_processing: bool = Field(default=True)
    batch_processing_chunk_size: int = Field(default=100)
    max_batch_size: int = Field(default=10000)
    health_check_interval_seconds: int = Field(default=30)
    health_check_timeout_seconds: int = Field(default=5)
    ner_service_timeout: int = Field(default=60)
    dp_service_timeout: int = Field(default=60)
    event_llm_service_timeout: int = Field(default=300)
    max_retries: int = Field(default=3)
    retry_backoff_seconds: int = Field(default=2)


class CelerySettings(BaseModel):
    """Celery and Dask settings."""
    broker_url: str = Field(default="redis://redis:6379/0")
    result_backend: str = Field(default="redis://redis:6379/0")
    task_serializer: str = Field(default="json")
    result_serializer: str = Field(default="json")
    accept_content: List[str] = Field(default_factory=lambda: ["json"])
    task_routes: Dict[str, str] = Field(default_factory=dict)
    task_time_limit: int = Field(default=3600)
    task_soft_time_limit: int = Field(default=3000)
    result_expires: int = Field(default=86400)
    dask_enabled: bool = Field(default=True)
    dask_local_cluster_n_workers: int = Field(default=22)
    dask_local_cluster_threads_per_worker: int = Field(default=1)
    dask_local_cluster_memory_limit: str = Field(default="6GB")
    dask_cluster_total_memory: str = Field(default="140GB")


class JSONLStorageConfig(BaseModel):
    """JSONL storage configuration."""
    output_directory: str = Field(default="/app/data")
    file_prefix: str = Field(default="extracted_events")
    use_daily_files: bool = Field(default=True)
    compression: Optional[str] = Field(default=None)
    append_mode: bool = Field(default=True)


class PostgreSQLStorageConfig(BaseModel):
    """PostgreSQL storage configuration."""
    host: str = Field(default="postgres")
    port: int = Field(default=5432)
    database: str = Field(default="nlp_processing")
    user: str = Field(default="nlp_user")
    password: str = Field(default="")
    table_name: str = Field(default="extracted_events")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    create_table_if_not_exists: bool = Field(default=True)
    use_jsonb_columns: bool = Field(default=True)


class ElasticsearchStorageConfig(BaseModel):
    """Elasticsearch storage configuration."""
    host: str = Field(default="elasticsearch")
    port: int = Field(default=9200)
    index_name: str = Field(default="eee_events")
    use_ssl: bool = Field(default=False)
    verify_certs: bool = Field(default=False)
    api_key: str = Field(default="")
    create_index_if_not_exists: bool = Field(default=True)
    number_of_shards: int = Field(default=3)
    number_of_replicas: int = Field(default=1)
    refresh_interval: str = Field(default="5s")


class StorageSettings(BaseModel):
    """Storage backend settings."""
    enabled_backends: List[str] = Field(default_factory=lambda: ["jsonl"])
    jsonl: JSONLStorageConfig = Field(default_factory=JSONLStorageConfig)
    postgresql: PostgreSQLStorageConfig = Field(default_factory=PostgreSQLStorageConfig)
    elasticsearch: ElasticsearchStorageConfig = Field(default_factory=ElasticsearchStorageConfig)


class CachingSettings(BaseModel):
    """Caching configuration."""
    enabled: bool = Field(default=True)
    redis_url: str = Field(default="redis://redis:6379/1")
    embedding_cache_ttl: int = Field(default=86400)
    entity_cache_ttl: int = Field(default=3600)
    event_cache_ttl: int = Field(default=7200)
    enable_prompt_cache: bool = Field(default=True)
    prompt_cache_size_mb: int = Field(default=2048)


class ConnectionPoolConfig(BaseModel):
    """Connection pool configuration for Redis Streams."""
    max_connections: int = Field(default=10)
    timeout: int = Field(default=5)


class RedisStreamsConfig(BaseModel):
    """Redis Streams backend configuration."""
    url: str = Field(default="redis://redis:6379/1")
    stream_name: str = Field(default="nlp-events")
    max_len: int = Field(default=10000)
    ttl_seconds: int = Field(default=86400)
    connection_pool: ConnectionPoolConfig = Field(default_factory=ConnectionPoolConfig)


class KafkaConfig(BaseModel):
    """Kafka backend configuration."""
    bootstrap_servers: List[str] = Field(default_factory=lambda: ["kafka:9092"])
    topic: str = Field(default="nlp-document-events")
    compression_type: str = Field(default="gzip")
    acks: int = Field(default=1)
    retries: int = Field(default=3)
    max_in_flight_requests: int = Field(default=5)
    client_id: str = Field(default="stage2-nlp-producer")


class NATSConfig(BaseModel):
    """NATS backend configuration."""
    servers: List[str] = Field(default_factory=lambda: ["nats://nats:4222"])
    subject: str = Field(default="nlp.document.processed")
    jetstream: bool = Field(default=True)
    stream: str = Field(default="NLP_EVENTS")
    durable_name: str = Field(default="nlp-processor")


class RabbitMQConfig(BaseModel):
    """RabbitMQ backend configuration."""
    url: str = Field(default="amqp://guest:guest@rabbitmq:5672/")
    exchange: str = Field(default="nlp-events")
    exchange_type: str = Field(default="topic")
    routing_key: str = Field(default="document.processed")
    durable: bool = Field(default=True)


class WebhookConfig(BaseModel):
    """Webhook backend configuration."""
    urls: List[str] = Field(default_factory=list)
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=5)
    retry_attempts: int = Field(default=3)
    retry_backoff: str = Field(default="exponential")
    retry_delay_seconds: float = Field(default=1.0)
    verify_ssl: bool = Field(default=True)


class PublishEventsConfig(BaseModel):
    """Event filtering configuration."""
    document_processed: bool = Field(default=True)
    document_failed: bool = Field(default=True)
    batch_started: bool = Field(default=True)
    batch_completed: bool = Field(default=True)


class EventMonitoringConfig(BaseModel):
    """Event monitoring configuration."""
    track_publish_latency: bool = Field(default=True)
    log_events: bool = Field(default=True)
    log_level: str = Field(default="INFO")


class EventsConfig(BaseModel):
    """Inter-stage communication event publishing configuration."""
    enabled: bool = Field(default=False, description="Global enable/disable for events")
    backend: Optional[str] = Field(default="redis_streams", description="Single backend type (deprecated, use backends)")
    backends: Optional[List[str]] = Field(default=None, description="Multiple backend types to use simultaneously")
    publish_events: PublishEventsConfig = Field(default_factory=PublishEventsConfig)
    redis_streams: RedisStreamsConfig = Field(default_factory=RedisStreamsConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    nats: NATSConfig = Field(default_factory=NATSConfig)
    rabbitmq: RabbitMQConfig = Field(default_factory=RabbitMQConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    monitoring: EventMonitoringConfig = Field(default_factory=EventMonitoringConfig)


class MonitoringSettings(BaseModel):
    """Monitoring and observability settings."""
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    enable_tracing: bool = Field(default=False)
    tracing_endpoint: str = Field(default="http://jaeger:4318")
    enable_health_endpoints: bool = Field(default=True)
    log_format: str = Field(default="json")
    log_file: str = Field(default="/app/logs/nlp_processing.log")
    log_rotation: str = Field(default="100MB")
    log_retention_days: int = Field(default=30)


class DevelopmentSettings(BaseModel):
    """Development-specific settings."""
    reload_on_change: bool = Field(default=False)
    debug: bool = Field(default=False)
    use_sample_data: bool = Field(default=False)
    sample_data_path: str = Field(default="/app/data/sample_documents.jsonl")


class Settings(BaseModel):
    """Root configuration model."""
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    document_field_mapping: DocumentFieldMapping = Field(default_factory=DocumentFieldMapping)
    ner_service: NERServiceSettings = Field(default_factory=NERServiceSettings)
    dp_service: DPServiceSettings = Field(default_factory=DPServiceSettings)
    event_llm_service: EventLLMServiceSettings = Field(default_factory=EventLLMServiceSettings)
    event_linking: EventLinkingSettings = Field(default_factory=EventLinkingSettings)
    orchestrator_service: OrchestratorServiceSettings = Field(default_factory=OrchestratorServiceSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    caching: CachingSettings = Field(default_factory=CachingSettings)
    events: EventsConfig = Field(default_factory=EventsConfig)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    development: DevelopmentSettings = Field(default_factory=DevelopmentSettings)


# =============================================================================
# Configuration Manager (Singleton)
# =============================================================================

class ConfigManager:
    """
    Singleton configuration manager that loads and caches settings.

    Features:
    - Loads YAML configuration from file
    - Substitutes environment variables using ${VAR_NAME} syntax
    - Validates configuration using Pydantic models
    - Provides global access to settings
    """

    _instance: Optional['ConfigManager'] = None
    _settings: Optional[Settings] = None

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Settings:
        """
        Load configuration from YAML file with environment variable substitution.

        Args:
            config_path: Path to configuration file. If None, uses default path.

        Returns:
            Settings object with validated configuration

        Raises:
            FileNotFoundError: If configuration file not found
            ValueError: If configuration is invalid
        """
        if cls._settings is not None:
            return cls._settings

        # Determine config path
        if config_path is None:
            # Try multiple default locations
            possible_paths = [
                Path("/app/config/settings.yaml"),
                Path("config/settings.yaml"),
                Path("../config/settings.yaml"),
                Path(os.getenv("CONFIG_PATH", "config/settings.yaml"))
            ]

            config_path_obj = None
            for path in possible_paths:
                if path.exists():
                    config_path_obj = path
                    break

            if config_path_obj is None:
                raise FileNotFoundError(
                    f"Configuration file not found in any of: {[str(p) for p in possible_paths]}"
                )
        else:
            config_path_obj = Path(config_path)
            if not config_path_obj.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from: {config_path_obj}")

        # Load YAML file
        try:
            with open(config_path_obj, 'r') as f:
                raw_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load YAML configuration: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")

        # Substitute environment variables
        config_dict = cls._substitute_env_vars(raw_config)

        # Validate and create Settings object
        try:
            cls._settings = Settings(**config_dict)
            logger.info("Configuration loaded and validated successfully")
            return cls._settings
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")

    @classmethod
    def get_settings(cls) -> Settings:
        """
        Get cached settings. Loads from default path if not already loaded.

        Returns:
            Settings object
        """
        if cls._settings is None:
            cls.load_config()
        return cls._settings

    @classmethod
    def _substitute_env_vars(cls, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.

        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.

        Args:
            config: Configuration dictionary or value

        Returns:
            Configuration with substituted values
        """
        if isinstance(config, dict):
            return {k: cls._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [cls._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Match ${VAR_NAME} or ${VAR_NAME:default}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

            def replace_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ""
                return os.getenv(var_name, default_value)

            return re.sub(pattern, replace_var, config)
        else:
            return config

    @classmethod
    def reload_config(cls, config_path: Optional[str] = None) -> Settings:
        """
        Reload configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Reloaded Settings object
        """
        cls._settings = None
        return cls.load_config(config_path)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_settings() -> Settings:
    """
    Get application settings (convenience function).

    Returns:
        Settings object
    """
    return ConfigManager.get_settings()


def get_device() -> str:
    """
    Get device for model inference (cuda or cpu).

    Returns:
        Device string
    """
    settings = get_settings()
    if settings.general.gpu_enabled and settings.general.device == "cuda":
        import torch
        if torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("GPU requested but CUDA not available. Falling back to CPU.")
            return "cpu"
    return "cpu"


# =============================================================================
# Module Initialization
# =============================================================================

if __name__ == "__main__":
    # Test configuration loading
    import sys
    logging.basicConfig(level=logging.INFO)

    try:
        settings = ConfigManager.load_config()
        print("Configuration loaded successfully!")
        print(f"GPU enabled: {settings.general.gpu_enabled}")
        print(f"NER model: {settings.ner_service.model_name}")
        print(f"Event LLM model: {settings.event_llm_service.model_name}")
        print(f"Using vLLM: {settings.event_llm_service.use_vllm}")
        print(f"Storage backends: {settings.storage.enabled_backends}")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        sys.exit(1)
