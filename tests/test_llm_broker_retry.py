import asyncio
from typing import cast
from unittest.mock import patch

import pytest

from llm_actor import LLMBrokerService, LLMBrokerSettings
from llm_actor.client.interface import LLMClientWithCircuitBreakerInterface
from llm_actor.client.llm import LLMClientWithCircuitBreaker
from llm_actor.client.retry import LLMClientWithRetry
from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import (
    LLMServiceHTTPError,
    LLMServiceOverloadedError,
    LLMServiceTimeoutError,
    LLMServiceUnavailableError,
)
from llm_actor.resilience.circuit_breaker import CircuitBreaker
from tests.dummy_llm_client import DummyLLMClient


async def test_retry_succeeds_after_transient_error():
    """Тест успешного retry после временной ошибки."""
    settings = LLMBrokerSettings()
    settings.LLM_RETRY_MAX_ATTEMPTS = 3
    settings.LLM_RETRY_BASE_BACKOFF = 0.1
    settings.LLM_RETRY_BACKOFF_CAP = 1.0

    base_client = DummyLLMClient(settings=settings)
    base_client.set_prompt_errors("test", [
        LLMServiceOverloadedError("Temporary overload"),
    ])
    circuit_breaker = CircuitBreaker(settings=settings)
    cb_client = cast(LLMClientWithCircuitBreakerInterface, LLMClientWithCircuitBreaker(base_client, circuit_breaker))
    retry_client = LLMClientWithRetry(cb_client, settings)

    result = await retry_client.generate(LLMRequest(prompt="test"))

    assert result == "Response for: test"
    assert base_client.prompt_call_counts["test"] == 2


async def test_retry_exponential_backoff():
    """Тест exponential backoff при retry."""
    settings = LLMBrokerSettings()
    settings.LLM_RETRY_MAX_ATTEMPTS = 4
    settings.LLM_RETRY_BASE_BACKOFF = 0.1
    settings.LLM_RETRY_BACKOFF_CAP = 0.5

    base_client = DummyLLMClient(settings=settings)
    base_client.base_latency = 0.0
    base_client.latency_variance = 0.0
    base_client.set_prompt_errors("test", [
        LLMServiceOverloadedError("Temporary overload"),
        LLMServiceOverloadedError("Temporary overload"),
        LLMServiceOverloadedError("Temporary overload"),
    ])
    circuit_breaker = CircuitBreaker(settings=settings)
    cb_client = cast(LLMClientWithCircuitBreakerInterface, LLMClientWithCircuitBreaker(base_client, circuit_breaker))
    retry_client = LLMClientWithRetry(cb_client, settings)

    sleep_times = []
    original_sleep = asyncio.sleep

    async def mock_sleep(delay: float) -> None:
        sleep_times.append(delay)
        await original_sleep(0.01)

    with patch("asyncio.sleep", mock_sleep):
        result = await retry_client.generate(LLMRequest(prompt="test"))

    assert result == "Response for: test"
    assert base_client.prompt_call_counts["test"] == 4
    expected_backoffs = [0.1, 0.2, 0.4]
    for expected in expected_backoffs:
        matching = [s for s in sleep_times if abs(s - expected) < 0.01]
        assert len(matching) >= 1, f"Expected backoff {expected} not found in {sleep_times}"


async def test_retry_backoff_cap():
    """Тест ограничения максимальной задержки backoff."""
    settings = LLMBrokerSettings()
    settings.LLM_RETRY_MAX_ATTEMPTS = 5
    settings.LLM_RETRY_BASE_BACKOFF = 1.0
    settings.LLM_RETRY_BACKOFF_CAP = 2.0

    base_client = DummyLLMClient(settings=settings)
    base_client.base_latency = 0.0
    base_client.latency_variance = 0.0
    base_client.set_prompt_errors("test", [
        LLMServiceOverloadedError("Temporary overload"),
        LLMServiceOverloadedError("Temporary overload"),
        LLMServiceOverloadedError("Temporary overload"),
        LLMServiceOverloadedError("Temporary overload"),
    ])
    circuit_breaker = CircuitBreaker(settings=settings)
    cb_client = cast(LLMClientWithCircuitBreakerInterface, LLMClientWithCircuitBreaker(base_client, circuit_breaker))
    retry_client = LLMClientWithRetry(cb_client, settings)

    sleep_times = []
    original_sleep = asyncio.sleep

    async def mock_sleep(delay: float) -> None:
        sleep_times.append(delay)
        await original_sleep(0.01)

    with patch("asyncio.sleep", mock_sleep):
        result = await retry_client.generate(LLMRequest(prompt="test"))

    assert result == "Response for: test"
    retry_sleeps = [s for s in sleep_times if s >= 0.5]
    assert len(retry_sleeps) == 4
    assert all(sleep_time <= 2.0 for sleep_time in retry_sleeps)
    assert retry_sleeps[-1] == pytest.approx(2.0, abs=0.01)


async def test_retry_fails_after_max_attempts():
    """Тест провала после исчерпания всех попыток."""
    settings = LLMBrokerSettings()
    settings.LLM_RETRY_MAX_ATTEMPTS = 3
    settings.LLM_RETRY_BASE_BACKOFF = 0.01

    base_client = DummyLLMClient(settings=settings)
    base_client.set_prompt_errors("test", [
        LLMServiceOverloadedError("Persistent overload"),
        LLMServiceOverloadedError("Persistent overload"),
        LLMServiceOverloadedError("Persistent overload"),
    ])
    circuit_breaker = CircuitBreaker(settings=settings)
    cb_client = cast(LLMClientWithCircuitBreakerInterface, LLMClientWithCircuitBreaker(base_client, circuit_breaker))
    retry_client = LLMClientWithRetry(cb_client, settings)

    with pytest.raises(LLMServiceOverloadedError):
        await retry_client.generate(LLMRequest(prompt="test"))

    assert base_client.prompt_call_counts["test"] == 3


async def test_retry_handles_503_error():
    """Тест обработки ошибки 503 (Service Unavailable)."""
    settings = LLMBrokerSettings()
    settings.LLM_RETRY_MAX_ATTEMPTS = 3
    settings.LLM_RETRY_BASE_BACKOFF = 0.01

    base_client = DummyLLMClient(settings=settings)
    base_client.set_prompt_errors("test", [
        LLMServiceUnavailableError("Service unavailable"),
    ])
    circuit_breaker = CircuitBreaker(settings=settings)
    cb_client = cast(LLMClientWithCircuitBreakerInterface, LLMClientWithCircuitBreaker(base_client, circuit_breaker))
    retry_client = LLMClientWithRetry(cb_client, settings)

    result = await retry_client.generate(LLMRequest(prompt="test"))

    assert result == "Response for: test"
    assert base_client.prompt_call_counts["test"] == 2


async def test_retry_handles_502_504_errors():
    """Тест обработки ошибок 502 и 504 (Bad Gateway, Gateway Timeout)."""
    settings = LLMBrokerSettings()
    settings.LLM_RETRY_MAX_ATTEMPTS = 3
    settings.LLM_RETRY_BASE_BACKOFF = 0.01

    for status_code in [502, 504]:
        base_client = DummyLLMClient(settings=settings)
        base_client.set_prompt_errors("test", [
            LLMServiceHTTPError(f"HTTP {status_code}", status_code=status_code),
        ])
        circuit_breaker = CircuitBreaker(settings=settings)
        cb_client = cast(LLMClientWithCircuitBreakerInterface, LLMClientWithCircuitBreaker(base_client, circuit_breaker))
        retry_client = LLMClientWithRetry(cb_client, settings)

        result = await retry_client.generate(LLMRequest(prompt="test"))

        assert result == "Response for: test"
        assert base_client.prompt_call_counts["test"] == 2


async def test_retry_handles_llm_service_timeout_error():
    """Тест retry при LLMServiceTimeoutError."""
    settings = LLMBrokerSettings()
    settings.LLM_RETRY_MAX_ATTEMPTS = 3
    settings.LLM_RETRY_BASE_BACKOFF = 0.01

    base_client = DummyLLMClient(settings=settings)
    base_client.set_prompt_errors("test", [
        LLMServiceTimeoutError("Request timeout"),
    ])
    circuit_breaker = CircuitBreaker(settings=settings)
    cb_client = cast(LLMClientWithCircuitBreakerInterface, LLMClientWithCircuitBreaker(base_client, circuit_breaker))
    retry_client = LLMClientWithRetry(cb_client, settings)

    result = await retry_client.generate(LLMRequest(prompt="test"))

    assert result == "Response for: test"
    assert base_client.prompt_call_counts["test"] == 2


async def test_retry_does_not_retry_non_transient_errors():
    """Тест что не-transient ошибки не повторяются."""
    settings = LLMBrokerSettings()
    settings.LLM_RETRY_MAX_ATTEMPTS = 3

    base_client = DummyLLMClient(settings=settings)
    base_client.set_prompt_error("test", LLMServiceHTTPError("Bad Request", status_code=400))
    circuit_breaker = CircuitBreaker(settings=settings)
    cb_client = cast(LLMClientWithCircuitBreakerInterface, LLMClientWithCircuitBreaker(base_client, circuit_breaker))
    retry_client = LLMClientWithRetry(cb_client, settings)

    with pytest.raises(LLMServiceHTTPError):
        await retry_client.generate(LLMRequest(prompt="test"))

    assert base_client.prompt_call_counts["test"] == 1


async def test_retry_integration_with_service():
    """Интеграционный тест retry логики в составе LLMBrokerService."""
    settings = LLMBrokerSettings()
    settings.LLM_RETRY_MAX_ATTEMPTS = 3
    settings.LLM_RETRY_BASE_BACKOFF = 0.01
    settings.LLM_NUM_ACTORS = 1
    settings.LLM_BATCH_SIZE = 1

    base_client = DummyLLMClient(settings=settings)
    base_client.set_prompt_errors("test", [
        LLMServiceOverloadedError("Temporary overload"),
    ])

    service = LLMBrokerService(base_client=base_client, settings=settings)
    await service.start()

    try:
        result = await service.generate("test")
        assert result == "Response for: test"
        assert base_client.prompt_call_counts["test"] == 2
    finally:
        await service.stop()
