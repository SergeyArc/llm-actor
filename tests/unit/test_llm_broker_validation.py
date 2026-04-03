import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from llm_actor import LLMBrokerSettings
from llm_actor.actors.pool import _PrioritizedMessage
from llm_actor.actors.worker import ModelActor
from llm_actor.client.llm import (
    LLMClientWithCircuitBreaker,
    _strip_schema_descriptions,
    build_json_prompt,
)
from llm_actor.core.messages import ActorMessage
from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import ActorFailedError
from llm_actor.resilience.circuit_breaker import CircuitBreaker
from tests.models import User


async def test_ask_with_invalid_json_response_model(service, mock_llm_response):
    """Тест обработки невалидного JSON при использовании response_model."""
    prompt = "invalid_json"
    mock_llm_response[prompt] = "not a json string"

    with pytest.raises(json.JSONDecodeError):
        await service.generate(prompt, response_model=User)


async def test_ask_with_mismatched_json_structure(service, mock_llm_response):
    """Тест обработки JSON с несоответствующей структурой модели."""
    prompt = "mismatched_json"
    mock_llm_response[prompt] = '{"name": "Alice"}'

    with pytest.raises(ValidationError):
        await service.generate(prompt, response_model=User)


async def test_semantic_retry_succeeds_on_second_attempt() -> None:
    settings = LLMBrokerSettings(LLM_VALIDATION_RETRY_MAX_ATTEMPTS=3)
    mock_base_client = AsyncMock()
    mock_base_client.generate_async.side_effect = [
        "not a json string",
        '{"name": "Alice", "age": 30}',
    ]
    client = LLMClientWithCircuitBreaker(
        base_client=mock_base_client,
        circuit_breaker=CircuitBreaker(settings=settings),
        max_validation_attempts=settings.LLM_VALIDATION_RETRY_MAX_ATTEMPTS,
    )
    result = await client.generate(LLMRequest(prompt="get user"), response_model=User)
    assert result.name == "Alice"
    assert result.age == 30
    assert mock_base_client.generate_async.call_count == 2


async def test_semantic_retry_exhausts_all_max_attempts() -> None:
    settings = LLMBrokerSettings(LLM_VALIDATION_RETRY_MAX_ATTEMPTS=3)
    mock_base_client = AsyncMock()
    mock_base_client.generate_async.return_value = "not a json string"
    client = LLMClientWithCircuitBreaker(
        base_client=mock_base_client,
        circuit_breaker=CircuitBreaker(settings=settings),
        max_validation_attempts=settings.LLM_VALIDATION_RETRY_MAX_ATTEMPTS,
    )
    with pytest.raises(json.JSONDecodeError):
        await client.generate(LLMRequest(prompt="get user"), response_model=User)
    assert mock_base_client.generate_async.call_count == 3


def test_build_json_prompt_strips_description_fields():
    prompt = "Сгенерируй пользователя"
    prompt_with_schema = build_json_prompt(prompt, User)

    assert prompt in prompt_with_schema
    assert '"description"' not in prompt_with_schema
    assert '"properties"' in prompt_with_schema


def test_build_json_prompt_contains_exact_json_dumps_schema():
    """AC-4: prompt contains precisely json.dumps(schema) with correct separators."""
    prompt = "Сгенерируй пользователя"
    prompt_with_schema = build_json_prompt(prompt, User)

    cleaned = _strip_schema_descriptions(User.model_json_schema())
    expected_schema_str = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
    assert expected_schema_str in prompt_with_schema


def test_settings_ignores_llm_failure_rate_env_var(monkeypatch):
    """AC-3: LLM_FAILURE_RATE present in environment does not raise on settings load."""
    monkeypatch.setenv("LLM_FAILURE_RATE", "0.5")
    settings = LLMBrokerSettings()
    assert settings is not None


@pytest.mark.asyncio
async def test_actor_raises_actor_failed_error_at_threshold():
    """AC-2: ActorFailedError is raised when consecutive failure threshold is exceeded.

    _process_batch uses gather(return_exceptions=True) so individual client errors don't
    propagate to _safe_process_batch. We patch _process_batch directly to raise a
    batch-level error, which IS what increments _consecutive_failures.
    """
    from unittest.mock import patch

    settings = LLMBrokerSettings(LLM_MAX_CONSECUTIVE_FAILURES=1, LLM_BATCH_SIZE=1)
    mock_client = AsyncMock()
    shared_queue: asyncio.PriorityQueue[Any] = asyncio.PriorityQueue(
        maxsize=settings.LLM_MAX_QUEUE_SIZE
    )

    actor = ModelActor(
        client=mock_client,
        actor_id="test-actor",
        settings=settings,
        shared_queue=shared_queue,
    )

    batch_error = RuntimeError("Forced batch failure")

    with patch.object(actor, "_process_batch", new=AsyncMock(side_effect=batch_error)):
        await actor.start()

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        msg = ActorMessage(request=LLMRequest(prompt="fail"), future=future)
        await shared_queue.put(_PrioritizedMessage(priority=msg.priority, sequence=0, message=msg))

        with pytest.raises(ActorFailedError):
            assert actor.task is not None
            await actor.task

    assert actor.task is not None
    exc = actor.task.exception()
    assert isinstance(exc, ActorFailedError)
    assert exc.actor_id == "test-actor"
    assert msg in exc.pending_messages


@pytest.mark.asyncio
async def test_pool_requeues_pending_messages_on_actor_restart():
    """AC-2: Pending messages from ActorFailedError are returned to the pool queue."""
    from llm_actor.actors.pool import SupervisedActorPool

    settings = LLMBrokerSettings(
        LLM_NUM_ACTORS=1,
        LLM_MAX_CONSECUTIVE_FAILURES=1,
        LLM_BATCH_SIZE=1,
        LLM_BATCH_TIMEOUT=0.01,
    )
    mock_client = AsyncMock()
    mock_client.generate.return_value = "ok"

    pool = SupervisedActorPool(client=mock_client, settings=settings)
    await pool.start()

    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    msg = ActorMessage(request=LLMRequest(prompt="requeue-me"), future=future)

    failed_exc = ActorFailedError(
        message="forced",
        actor_id="actor-0",
        pending_messages=[msg],
    )

    try:
        # Replace actor's task with one that raises ActorFailedError
        pool._actor_tasks[0] = loop.create_task(_raise_exc(failed_exc))
        await asyncio.sleep(0.05)  # let the fake task complete

        # Directly trigger supervisor check — restarts actor and requeues msg
        await pool._check_actor_tasks()

        # Give the new actor time to process the requeued message
        result = await asyncio.wait_for(future, timeout=3.0)
        assert result == "ok"
    finally:
        await pool.stop()


async def _raise_exc(exc: Exception) -> None:
    raise exc
