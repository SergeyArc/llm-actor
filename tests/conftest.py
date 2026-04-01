import asyncio
from unittest.mock import patch

import pytest
import pytest_asyncio
from prometheus_client import REGISTRY

from llm_actor import LLMBrokerService
from tests.dummy_llm_client import DummyLLMClient


@pytest.fixture(autouse=True)
def clear_prometheus_registry():
    """Очистка реестра Prometheus перед каждым тестом."""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    yield
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


@pytest.fixture
def mock_llm_responses():
    """Фикстура для хранения мок-ответов LLM."""
    return {}


@pytest.fixture
def mock_llm_response(mock_llm_responses):
    """Фикстура для мокирования DummyLLMClient.generate_async с сохранением реалистичных задержек."""

    async def mock_generate_async(self, prompt: str) -> str:
        response_text = f"Response for: {prompt}"
        response = None
        if prompt in mock_llm_responses:
            response = mock_llm_responses[prompt]
        else:
            # Sort longest-first so more specific prefixes win over shorter ones.
            for known_prompt in sorted(mock_llm_responses, key=len, reverse=True):
                if prompt.startswith(known_prompt):
                    response = mock_llm_responses[known_prompt]
                    break

        if response is not None:
            if isinstance(response, Exception):
                await asyncio.sleep(0.01)
                raise response
            response_text = str(response)

        latency = self._calculate_latency(prompt, len(response_text))
        await asyncio.sleep(latency)

        return response_text

    with patch.object(DummyLLMClient, "generate_async", mock_generate_async):
        yield mock_llm_responses


@pytest_asyncio.fixture
async def service(mock_llm_response):
    """Фикстура для инициализированного и запущенного LLMBrokerService."""
    from llm_actor import LLMBrokerSettings

    settings = LLMBrokerSettings()
    base_client = DummyLLMClient(settings=settings)
    service = LLMBrokerService(base_client=base_client, settings=settings)
    await service.start()
    yield service
    await service.stop()
