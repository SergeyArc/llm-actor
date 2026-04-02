from prometheus_client import REGISTRY as _DEFAULT_REGISTRY
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class MetricsCollector:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        reg = registry if registry is not None else _DEFAULT_REGISTRY
        self.inbox_size_gauge = Gauge(
            "llm_actor_inbox_size",
            "Size of shared pool task queue",
            ["pool_id"],
            registry=reg,
        )
        self.batches_processed_counter = Counter(
            "llm_batches_processed_total",
            "Total number of successfully processed batches",
            ["actor_id"],
            registry=reg,
        )
        self.batches_failed_counter = Counter(
            "llm_batches_failed_total",
            "Total number of failed batches",
            ["actor_id"],
            registry=reg,
        )
        self.batch_processing_duration_histogram = Histogram(
            "llm_batch_processing_duration_seconds",
            "Time spent processing batch",
            ["actor_id"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
            registry=reg,
        )
        self.circuit_breaker_trips_counter = Counter(
            "llm_circuit_breaker_trips_total",
            "Total number of circuit breaker trips",
            registry=reg,
        )
        self.actor_restarts_counter = Counter(
            "llm_actor_restarts_total",
            "Total number of actor restarts",
            ["actor_id", "pool_id"],
            registry=reg,
        )
