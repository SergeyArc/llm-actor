import asyncio
from typing import Any, TypeVar, cast, overload

from llm_actor import tracing as otel_tracing
from llm_actor.actors.pool import SupervisedActorPool
from llm_actor.client.interface import (
    LLMClientInterface,
    LLMClientWithCircuitBreakerInterface,
)
from llm_actor.client.llm import LLMClientWithCircuitBreaker
from llm_actor.client.retry import LLMClientWithRetry
from llm_actor.client.tool_loop import ToolCallOrchestratorClient
from llm_actor.core.request import LLMRequest
from llm_actor.logger import ActorLogger
from llm_actor.metrics import MetricsCollector, default_metrics_collector
from llm_actor.resilience.circuit_breaker import CircuitBreaker
from llm_actor.settings import LLMActorSettings

T = TypeVar("T", bound=object)


def _coerce_llm_request(prompt_or_request: str | LLMRequest) -> LLMRequest:
    if isinstance(prompt_or_request, LLMRequest):
        return prompt_or_request
    return LLMRequest(prompt=prompt_or_request)


class LLMActorService:
    def __init__(
        self,
        base_client: LLMClientInterface,
        settings: LLMActorSettings | None = None,
        metrics: MetricsCollector | None = None,
    ):
        self._base_client = base_client
        self._settings = settings or LLMActorSettings()
        self._metrics = metrics if metrics is not None else default_metrics_collector()
        self._logger = ActorLogger.get_logger(name="llm_actor_service")

        circuit_breaker = CircuitBreaker(settings=self._settings, metrics=self._metrics)
        cb_client = cast(
            LLMClientWithCircuitBreakerInterface,
            LLMClientWithCircuitBreaker(
                base_client,
                circuit_breaker,
                max_validation_attempts=self._settings.LLM_VALIDATION_RETRY_MAX_ATTEMPTS,
            ),
        )
        orchestrated_client = cast(
            LLMClientWithCircuitBreakerInterface,
            ToolCallOrchestratorClient(cb_client, self._settings),
        )
        self._client: LLMClientWithCircuitBreakerInterface = LLMClientWithRetry(
            orchestrated_client, self._settings
        )

        self._pool = SupervisedActorPool(
            client=self._client, settings=self._settings, metrics=self._metrics
        )

    @classmethod
    def from_openai(
        cls,
        *,
        api_key: str,
        model: str,
        settings: LLMActorSettings | None = None,
        metrics: MetricsCollector | None = None,
        **client_options: Any,
    ) -> "LLMActorService":
        from llm_actor.client.adapters.openai import OpenAIAdapter

        base = OpenAIAdapter(api_key=api_key, model=model, **client_options)
        return cls(base_client=base, settings=settings, metrics=metrics)

    @classmethod
    def from_anthropic(
        cls,
        *,
        api_key: str,
        model: str,
        settings: LLMActorSettings | None = None,
        metrics: MetricsCollector | None = None,
        **client_options: Any,
    ) -> "LLMActorService":
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
        settings: LLMActorSettings | None = None,
        metrics: MetricsCollector | None = None,
        **client_options: Any,
    ) -> "LLMActorService":
        from llm_actor.client.adapters.openai_compatible import OpenAICompatibleAdapter

        base = OpenAICompatibleAdapter(
            api_key=api_key, model=model, base_url=base_url, **client_options
        )
        return cls(base_client=base, settings=settings, metrics=metrics)

    @classmethod
    def from_gigachat(
        cls,
        *,
        credentials: str | None = None,
        model: str | None = None,
        scope: str | None = None,
        verify_ssl_certs: bool = True,
        settings: LLMActorSettings | None = None,
        metrics: MetricsCollector | None = None,
        **client_options: Any,
    ) -> "LLMActorService":
        from llm_actor.client.adapters.gigachat import GigaChatAdapter

        base = GigaChatAdapter(
            credentials=credentials,
            model=model,
            scope=scope,
            verify_ssl_certs=verify_ssl_certs,
            **client_options,
        )
        return cls(base_client=base, settings=settings, metrics=metrics)

    @property
    def pool(self) -> SupervisedActorPool:
        """Actor pool for monitoring (tests and get_health_status())."""
        return self._pool

    @property
    def client(self) -> LLMClientWithCircuitBreakerInterface:
        """Wrapped client for interface checks (used in tests)."""
        return self._client

    async def start(self) -> None:
        self._logger.info("Starting LLMActorService")
        await self._pool.start()
        self._logger.info("LLMActorService started successfully")

    async def stop(self) -> None:
        self._logger.info("Stopping LLMActorService")
        await self._pool.stop()
        if hasattr(self._base_client, "close") and callable(self._base_client.close):
            try:
                self._logger.info("Closing base LLM client")
                await self._base_client.close()
            except Exception as exc:
                self._logger.error("Error while closing LLM client: {}", exc, exc_info=True)
        self._logger.info("LLMActorService stopped successfully")

    async def __aenter__(self) -> "LLMActorService":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.stop()

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
        tracer = otel_tracing.get_tracer()
        preview = otel_tracing.truncate_for_span_attribute(request.prompt)
        with tracer.start_as_current_span(
            "llm_actor.generate",
            attributes={
                "llm_actor.prompt_preview": preview,
                "llm_actor.priority": priority,
            },
        ):
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
        Process many requests concurrently.

        Args:
            requests: List of (prompt or LLMRequest, response_model) pairs.

        Returns:
            One entry per input: ``str`` if ``response_model`` was None; a validated
            model instance on success; or an ``Exception`` if that item failed.
        """
        self._logger.info(f"Processing batch of {len(requests)} requests")
        tracer = otel_tracing.get_tracer()
        with tracer.start_as_current_span(
            "llm_actor.generate_batch",
            attributes={"llm_actor.batch_size": len(requests)},
        ):
            tasks = []
            for item, response_model in requests:
                req = _coerce_llm_request(item)
                preview = req.prompt[:100] if req.prompt else ""
                self._logger.debug(f"Request: prompt={preview}..., response_model={response_model}")
                tasks.append(self.generate(req, response_model, priority=priority))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            error_count = sum(1 for r in results if isinstance(r, Exception))
            if error_count > 0:
                self._logger.warning(
                    f"Batch completed with {error_count} errors out of {len(requests)} requests"
                )
            else:
                self._logger.info(f"Batch completed successfully: {len(requests)} requests")
            return list(results)
