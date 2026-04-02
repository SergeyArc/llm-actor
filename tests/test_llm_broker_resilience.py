import asyncio
from unittest.mock import patch

import pytest

from llm_actor import LLMBrokerService, LLMBrokerSettings
from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import (
    CircuitBreakerOpenError,
    LLMServiceOverloadedError,
    OverloadError,
)
from tests.dummy_llm_client import DummyLLMClient


async def test_circuit_breaker_opens_after_threshold(service, mock_llm_response):
    """Тест открытия circuit breaker после превышения порога ошибок."""
    failure_prompt = "failure"
    for i in range(6):
        mock_llm_response[f"{failure_prompt}_{i}"] = RuntimeError(f"Error {i}")

    for i in range(5):
        with pytest.raises(RuntimeError):
            await service.generate(f"{failure_prompt}_{i}")

    await asyncio.sleep(0.1)

    with pytest.raises(CircuitBreakerOpenError):
        await service.generate(f"{failure_prompt}_5")


async def test_overload_error_when_queue_full():
    """Тест перегрузки при заполнении очереди актора."""
    settings = LLMBrokerSettings()
    settings.LLM_MAX_QUEUE_SIZE = 10
    settings.LLM_NUM_ACTORS = 1
    settings.LLM_BATCH_SIZE = 1
    settings.LLM_BATCH_TIMEOUT = 10.0

    async def slow_generate_async(self, request: LLMRequest) -> str:
        await asyncio.sleep(2.0)
        return f"Response for {request.prompt}"

    with patch.object(DummyLLMClient, "generate_async", slow_generate_async):
        base_client = DummyLLMClient(settings=settings)
        service = LLMBrokerService(base_client=base_client, settings=settings)
        await service.start()

        try:
            tasks = []
            for i in range(15):
                tasks.append(service.generate(f"request_{i}"))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            overload_errors = [r for r in results if isinstance(r, OverloadError)]
            assert len(overload_errors) > 0
        finally:
            await service.stop()


async def test_retry_before_circuit_breaker():
    """Тест что retry логика работает перед circuit breaker для transient ошибок."""
    settings = LLMBrokerSettings()
    settings.LLM_RETRY_MAX_ATTEMPTS = 3
    settings.LLM_RETRY_BASE_BACKOFF = 0.01
    settings.LLM_FAILURE_THRESHOLD = 10
    settings.LLM_NUM_ACTORS = 1
    settings.LLM_BATCH_SIZE = 1

    base_client = DummyLLMClient(settings=settings)
    base_client.set_prompt_errors(
        "retry_success",
        [
            LLMServiceOverloadedError("Temporary overload"),
        ],
    )
    base_client.set_prompt_errors(
        "retry_fail",
        [
            LLMServiceOverloadedError("Persistent overload"),
            LLMServiceOverloadedError("Persistent overload"),
            LLMServiceOverloadedError("Persistent overload"),
        ],
    )

    service = LLMBrokerService(base_client=base_client, settings=settings)
    await service.start()

    try:
        result = await service.generate("retry_success")
        assert result == "Response for: retry_success"
        assert base_client.prompt_call_counts["retry_success"] == 2

        with pytest.raises(LLMServiceOverloadedError):
            await service.generate("retry_fail")
        assert base_client.prompt_call_counts["retry_fail"] == 3
    finally:
        await service.stop()
