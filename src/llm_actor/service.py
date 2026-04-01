import asyncio
from typing import Any, TypeVar, cast, overload

from llm_actor.actors.pool import SupervisedActorPool
from llm_actor.client.interface import (
    LLMClientInterface,
    LLMClientWithCircuitBreakerInterface,
)
from llm_actor.client.llm import LLMClientWithCircuitBreaker
from llm_actor.client.retry import LLMClientWithRetry
from llm_actor.core.request import LLMRequest
from llm_actor.logger import BrokerLogger
from llm_actor.metrics import MetricsCollector
from llm_actor.resilience.circuit_breaker import CircuitBreaker
from llm_actor.settings import LLMBrokerSettings

T = TypeVar("T", bound=object)


def _coerce_llm_request(prompt_or_request: str | LLMRequest) -> LLMRequest:
    if isinstance(prompt_or_request, LLMRequest):
        return prompt_or_request
    return LLMRequest(prompt=prompt_or_request)


class LLMBrokerService:
    def __init__(
        self,
        base_client: LLMClientInterface,
        settings: LLMBrokerSettings | None = None,
        metrics: MetricsCollector | None = None,
    ):
        self._base_client = base_client
        self._settings = settings or LLMBrokerSettings()
        self._metrics = metrics or MetricsCollector()

        circuit_breaker = CircuitBreaker(settings=self._settings, metrics=self._metrics)
        cb_client = cast(
            LLMClientWithCircuitBreakerInterface,
            LLMClientWithCircuitBreaker(
                base_client,
                circuit_breaker,
                max_validation_attempts=self._settings.LLM_VALIDATION_RETRY_MAX_ATTEMPTS,
            ),
        )
        self._client: LLMClientWithCircuitBreakerInterface = LLMClientWithRetry(
            cb_client, self._settings
        )

        self._pool = SupervisedActorPool(
            client=self._client, settings=self._settings, metrics=self._metrics
        )
        self._logger = BrokerLogger.get_logger(name="llm_actor_service")

    @classmethod
    def from_openai(
        cls,
        *,
        api_key: str,
        model: str,
        settings: LLMBrokerSettings | None = None,
        metrics: MetricsCollector | None = None,
        **client_options: Any,
    ) -> "LLMBrokerService":
        from llm_actor.client.adapters.openai import OpenAIAdapter

        base = OpenAIAdapter(api_key=api_key, model=model, **client_options)
        return cls(base_client=base, settings=settings, metrics=metrics)

    @classmethod
    def from_anthropic(
        cls,
        *,
        api_key: str,
        model: str,
        settings: LLMBrokerSettings | None = None,
        metrics: MetricsCollector | None = None,
        **client_options: Any,
    ) -> "LLMBrokerService":
        from llm_actor.client.adapters.anthropic import AnthropicAdapter

        base = AnthropicAdapter(api_key=api_key, model=model, **client_options)
        return cls(base_client=base, settings=settings, metrics=metrics)

    @classmethod
    def from_openai_compatible(
        cls,
        *,
        api_key: str,
        model: str,
        base_url: str,
        settings: LLMBrokerSettings | None = None,
        metrics: MetricsCollector | None = None,
        **client_options: Any,
    ) -> "LLMBrokerService":
        from llm_actor.client.adapters.openai_compatible import OpenAICompatibleAdapter

        base = OpenAICompatibleAdapter(
            api_key=api_key, model=model, base_url=base_url, **client_options
        )
        return cls(base_client=base, settings=settings, metrics=metrics)

    @property
    def pool(self) -> SupervisedActorPool:
        """Доступ к пулу акторов для мониторинга (используется в тестах и API для get_health_status())."""
        return self._pool

    @property
    def client(self) -> LLMClientWithCircuitBreakerInterface:
        """Доступ к клиенту для проверки интерфейса (используется в тестах)."""
        return self._client

    async def start(self) -> None:
        self._logger.info("Starting LLMBrokerService")
        await self._pool.start()
        self._logger.info("LLMBrokerService started successfully")

    async def stop(self) -> None:
        self._logger.info("Stopping LLMBrokerService")
        await self._pool.stop()
        if hasattr(self._base_client, "close") and callable(self._base_client.close):
            try:
                self._logger.info("Closing base LLM client")
                await self._base_client.close()
            except Exception as exc:
                self._logger.error("Error while closing LLM client: {}", exc, exc_info=True)
        self._logger.info("LLMBrokerService stopped successfully")

    @overload
    async def generate(
        self,
        prompt: str | LLMRequest,
        response_model: None = None,
        *,
        priority: int = 10,
    ) -> str: ...

    @overload
    async def generate(
        self,
        prompt: str | LLMRequest,
        response_model: type[T],
        *,
        priority: int = 10,
    ) -> T: ...

    async def generate(
        self,
        prompt: str | LLMRequest,
        response_model: type[Any] | None = None,
        *,
        priority: int = 10,
    ) -> Any | str:
        request = _coerce_llm_request(prompt)
        self._logger.debug(
            f"Processing generate request (response_model={'provided' if response_model else 'None'})"
        )
        return await self._pool.generate(request, response_model, priority=priority)

    async def generate_batch(
        self,
        requests: list[tuple[str | LLMRequest, type[Any] | None]],
        *,
        priority: int = 10,
    ) -> list[str | Any | Exception]:
        """
        Пакетная обработка запросов с параллельным выполнением.

        Args:
            requests: Список кортежей (промпт или LLMRequest, response_model)

        Returns:
            Список результатов. Каждый элемент может быть:
            - str: если response_model был None
            - объект типа response_model: если валидация прошла успешно
            - Exception: если произошла ошибка при обработке запроса
        """
        self._logger.info(f"Processing batch of {len(requests)} requests")
        tasks = []
        for item, response_model in requests:
            req = _coerce_llm_request(item)
            preview = req.prompt[:100] if req.prompt else ""
            self._logger.info(f"Request: prompt={preview}..., response_model={response_model}")
            tasks.append(self._pool.generate(req, response_model, priority=priority))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        error_count = sum(1 for r in results if isinstance(r, Exception))
        if error_count > 0:
            self._logger.warning(
                f"Batch completed with {error_count} errors out of {len(requests)} requests"
            )
        else:
            self._logger.info(f"Batch completed successfully: {len(requests)} requests")
        return list(results)
