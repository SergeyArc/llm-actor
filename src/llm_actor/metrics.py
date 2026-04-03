from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry


_METRICS_INSTALL_HINT = "pip install 'llm-actor[metrics]'"


def is_prometheus_metrics_available() -> bool:
    return importlib.util.find_spec("prometheus_client") is not None


def default_metrics_collector() -> MetricsCollector | None:
    if not is_prometheus_metrics_available():
        return None
    return MetricsCollector()


class MetricsCollector:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        try:
            from prometheus_client import REGISTRY, Counter, Gauge, Histogram
        except ImportError as exc:
            raise ImportError(
                f"Prometheus metrics require optional dependency: {_METRICS_INSTALL_HINT}"
            ) from exc

        reg = registry if registry is not None else REGISTRY
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
