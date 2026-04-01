
from llm_actor import (
    CircuitBreaker,
    LLMBrokerSettings,
    LLMClientInterface,
    LLMClientWithCircuitBreakerInterface,
    MetricsCollector,
)
from llm_actor.client.llm import LLMClientWithCircuitBreaker
from llm_actor.core.request import LLMRequest
from tests.dummy_llm_client import DummyLLMClient


async def test_llm_client_interface_compliance():
    """Тест проверки соответствия клиентов интерфейсам."""
    settings = LLMBrokerSettings()
    metrics = MetricsCollector()
    dummy_client = DummyLLMClient(settings=settings)
    circuit_breaker = CircuitBreaker(settings=settings, metrics=metrics)
    wrapped_client = LLMClientWithCircuitBreaker(dummy_client, circuit_breaker)

    dummy_result = await dummy_client.generate_async(LLMRequest(prompt="test"))
    wrapped_result = await wrapped_client.generate(LLMRequest(prompt="test"))

    assert isinstance(dummy_client, LLMClientInterface)
    assert isinstance(wrapped_client, LLMClientWithCircuitBreakerInterface)
    assert isinstance(dummy_result, str)
    assert isinstance(wrapped_result, str)


async def test_service_client_implements_interface(service):
    """Тест проверки что client в сервисе реализует интерфейс."""
    assert isinstance(service.client, LLMClientWithCircuitBreakerInterface)
    assert hasattr(service.client, "generate")
