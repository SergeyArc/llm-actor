from unittest.mock import AsyncMock

import pytest

from llm_actor import LLMBrokerService, LLMBrokerSettings
from llm_actor.exceptions import PoolShuttingDownError
from tests.dummy_llm_client import DummyLLMClient


async def test_service_lifecycle(mock_llm_response):
    """Тест полного жизненного цикла сервиса с graceful shutdown."""
    settings = LLMBrokerSettings()
    base_client = DummyLLMClient(settings=settings)
    service = LLMBrokerService(base_client=base_client, settings=settings)
    await service.start()

    prompts = ["test1", "test2", "test3"]
    for prompt in prompts:
        mock_llm_response[prompt] = f"Response for {prompt}"

    results = []
    for prompt in prompts:
        result = await service.generate(prompt)
        results.append(result)

    await service.stop()

    assert len(results) == len(prompts)
    for result in results:
        assert isinstance(result, str)

    health = service.pool.get_health_status()
    assert health.alive_actors == 0


async def test_ask_after_stop_raises_error(mock_llm_response):
    """Тест вызова ask после stop должен вызывать PoolShuttingDownError."""
    settings = LLMBrokerSettings()
    base_client = DummyLLMClient(settings=settings)
    service = LLMBrokerService(base_client=base_client, settings=settings)
    await service.start()
    await service.stop()

    with pytest.raises(PoolShuttingDownError, match="Pool is shutting down"):
        await service.generate("test")


async def test_service_calls_client_close(mock_llm_response):
    """Проверяет, что при остановке сервиса вызывается close() у базового клиента."""
    settings = LLMBrokerSettings()
    base_client = DummyLLMClient(settings=settings)
    base_client.close = AsyncMock()

    service = LLMBrokerService(base_client=base_client, settings=settings)
    await service.start()
    await service.stop()

    base_client.close.assert_called_once()


async def test_service_handles_client_close_error(mock_llm_response):
    """Проверяет, что ошибка при закрытии клиента не прерывает остановку брокера."""
    settings = LLMBrokerSettings()
    base_client = DummyLLMClient(settings=settings)
    base_client.close = AsyncMock(side_effect=RuntimeError("Close failed"))

    service = LLMBrokerService(base_client=base_client, settings=settings)
    await service.start()
    await service.stop()

    base_client.close.assert_called_once()
