import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from prometheus_client import CollectorRegistry

from llm_actor import LLMActorService
from llm_actor.actors.pool import SupervisedActorPool, _PrioritizedMessage
from llm_actor.actors.worker import ModelActor
from llm_actor.core.messages import ActorMessage
from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import ActorFailedError
from llm_actor.metrics import MetricsCollector, default_metrics_collector, is_prometheus_metrics_available
from llm_actor.settings import LLMActorSettings
from tests.dummy_llm_client import DummyLLMClient


def test_default_metrics_collector_returns_same_instance() -> None:
    if not is_prometheus_metrics_available():
        pytest.skip("prometheus_client not installed")
    first = default_metrics_collector()
    second = default_metrics_collector()
    assert first is not None
    assert first is second


async def test_two_llm_services_with_default_metrics_do_not_duplicate_registry() -> None:
    if not is_prometheus_metrics_available():
        pytest.skip("prometheus_client not installed")
    settings = LLMActorSettings(
        LLM_NUM_ACTORS=1,
        LLM_BATCH_SIZE=1,
        LLM_BATCH_TIMEOUT=0.01,
    )
    for _ in range(2):
        client = DummyLLMClient(settings=settings)
        service = LLMActorService(base_client=client, settings=settings)
        await service.start()
        await service.stop()


@pytest.fixture
def isolated_registry() -> CollectorRegistry:
    return CollectorRegistry()


@pytest_asyncio.fixture
async def metrics_service(isolated_registry: CollectorRegistry):
    settings = LLMActorSettings(
        LLM_NUM_ACTORS=1,
        LLM_BATCH_SIZE=1,
        LLM_BATCH_TIMEOUT=0.01,
    )
    metrics = MetricsCollector(registry=isolated_registry)
    base_client = DummyLLMClient(settings=settings)
    service = LLMActorService(base_client=base_client, settings=settings, metrics=metrics)
    await service.start()
    yield service, isolated_registry
    await service.stop()


async def test_batches_processed_counter_increments(
    metrics_service: tuple[LLMActorService, CollectorRegistry],
) -> None:
    service, registry = metrics_service
    before = registry.get_sample_value("llm_batches_processed_total", {"actor_id": "actor-0"})
    assert before in (None, 0.0)

    await service.generate("metrics_ok_prompt")

    after = registry.get_sample_value("llm_batches_processed_total", {"actor_id": "actor-0"})
    assert after == 1.0


async def test_batches_failed_counter_increments_when_process_batch_raises() -> None:
    """llm_batches_failed_total increments when _process_batch raises."""
    settings = LLMActorSettings(
        LLM_MAX_CONSECUTIVE_FAILURES=10,
        LLM_BATCH_SIZE=1,
        LLM_BATCH_TIMEOUT=0.01,
    )
    registry = CollectorRegistry()
    metrics = MetricsCollector(registry=registry)
    mock_client = AsyncMock()
    mock_client.generate.return_value = "ok"
    queue: asyncio.PriorityQueue[_PrioritizedMessage] = asyncio.PriorityQueue(
        maxsize=settings.LLM_MAX_QUEUE_SIZE
    )
    actor = ModelActor(
        client=mock_client,
        actor_id="metric-actor",
        settings=settings,
        shared_queue=queue,
        metrics=metrics,
    )

    with patch.object(actor, "_process_batch", new=AsyncMock(side_effect=RuntimeError("forced"))):
        await actor.start()
        try:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[str] = loop.create_future()
            msg = ActorMessage(request=LLMRequest(prompt="fail-batch"), future=future)
            await queue.put(_PrioritizedMessage(priority=10, sequence=0, message=msg))

            with pytest.raises(RuntimeError, match="forced"):
                await asyncio.wait_for(future, timeout=3.0)

            value = registry.get_sample_value(
                "llm_batches_failed_total", {"actor_id": "metric-actor"}
            )
            assert value == 1.0
        finally:
            await actor.stop()


async def test_circuit_breaker_trips_counter_increments(
    isolated_registry: CollectorRegistry,
) -> None:
    settings = LLMActorSettings(
        LLM_NUM_ACTORS=1,
        LLM_BATCH_SIZE=1,
        LLM_BATCH_TIMEOUT=0.01,
        LLM_FAILURE_THRESHOLD=5,
    )
    metrics = MetricsCollector(registry=isolated_registry)
    base_client = DummyLLMClient(settings=settings)
    for i in range(6):
        base_client.set_prompt_error(f"cb_fail_{i}", RuntimeError(f"err {i}"))

    service = LLMActorService(base_client=base_client, settings=settings, metrics=metrics)
    await service.start()
    try:
        assert isolated_registry.get_sample_value("llm_circuit_breaker_trips_total") in (None, 0.0)

        for i in range(5):
            with pytest.raises(RuntimeError):
                await service.generate(f"cb_fail_{i}")

        trips = isolated_registry.get_sample_value("llm_circuit_breaker_trips_total")
        assert trips == 1.0
    finally:
        await service.stop()


async def test_actor_restarts_counter_increments(
    isolated_registry: CollectorRegistry,
) -> None:
    settings = LLMActorSettings(
        LLM_NUM_ACTORS=1,
        LLM_MAX_CONSECUTIVE_FAILURES=1,
        LLM_BATCH_SIZE=1,
        LLM_BATCH_TIMEOUT=0.01,
    )
    metrics = MetricsCollector(registry=isolated_registry)
    mock_client = AsyncMock()
    mock_client.generate.return_value = "ok"

    pool = SupervisedActorPool(
        client=mock_client,
        settings=settings,
        metrics=metrics,
        pool_id="metrics-pool",
    )
    await pool.start()

    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    msg = ActorMessage(request=LLMRequest(prompt="requeue-metrics"), future=future)
    failed_exc = ActorFailedError(
        message="forced",
        actor_id="actor-0",
        pending_messages=[msg],
    )

    try:
        pool._actor_tasks[0] = loop.create_task(_raise_actor_failed(failed_exc))
        await asyncio.sleep(0.05)
        await pool._check_actor_tasks()

        result = await asyncio.wait_for(future, timeout=3.0)
        assert result == "ok"

        restarts = isolated_registry.get_sample_value(
            "llm_actor_restarts_total",
            {"actor_id": "actor-0-restart-1", "pool_id": "metrics-pool"},
        )
        assert restarts == 1.0
    finally:
        await pool.stop()


async def _raise_actor_failed(exc: ActorFailedError) -> None:
    raise exc
