from typing import Any, Protocol, runtime_checkable

from llm_actor.core.request import LLMRequest
from llm_actor.core.tools import LLMResponse, ToolResult


@runtime_checkable
class LLMClientInterface(Protocol):
    """
    Интерфейс для внешних LLM клиентов, используемых в llm_actor.

    ВАЖНО: Внешние клиенты НЕ должны реализовывать retry логику самостоятельно.
    Retry механизм с exponential backoff обеспечивается брокером на уровне
    LLMBrokerService через LLMClientWithRetry.

    Реализации должны просто выбрасывать соответствующие исключения при ошибках:
    - LLMServiceOverloadedError (429) - при перегрузке API
    - LLMServiceUnavailableError (503) - при недоступности сервиса
    - LLMServiceHTTPError (502, 503, 504) - при сетевых ошибках
    - LLMServiceTimeoutError - при таймаутах транспорта к LLM

    Эти ошибки будут автоматически обработаны retry wrapper (LLMClientWithRetry)
    с применением exponential backoff на уровне брокера.
    """

    async def generate_async(self, request: LLMRequest) -> str:
        """
        Генерирует ответ от LLM на основе запроса.

        Внешние клиенты должны выполнять один HTTP-запрос к LLM API без retry логики.
        Retry механизм обеспечивается брокером.

        Args:
            request: Параметры генерации (промпт, температура, extra и т.д.)

        Returns:
            Строка с ответом от LLM (не None)

        Raises:
            Exception: При ошибках обращения к LLM API.
                Временные ошибки (429, 503, 502, 504, timeout) будут
                автоматически обработаны retry механизмом брокера.
                Не нужно реализовывать retry логику в самом клиенте.
        """
        ...


@runtime_checkable
class LLMClientWithCircuitBreakerInterface(Protocol):
    """
    Интерфейс для LLM клиентов с Circuit Breaker, используемых внутри llm_actor.

    Ошибки валидации (ValueError, ValidationError, JSONDecodeError) обрабатываются
    на уровне LLMClientWithRetry с автоматическими повторами запросов.
    """

    async def generate(
        self, request: LLMRequest, response_model: type[Any] | None = None
    ) -> Any | str:
        """
        Генерирует ответ от LLM на основе запроса с поддержкой валидации через Pydantic.

        Args:
            request: Параметры генерации
            response_model: Опциональная Pydantic модель для валидации ответа

        Returns:
            Если response_model указан - валидированный объект модели, иначе строка

        Raises:
            Exception: При ошибках обращения к LLM API или валидации.
                Ошибки валидации будут автоматически обработаны LLMClientWithRetry
                с повторными запросами к LLM.
        """
        ...


@runtime_checkable
class ToolCapableClientInterface(Protocol):
    async def generate_with_tools_async(
        self,
        request: LLMRequest,
        conversation: list[dict[str, Any]],
    ) -> LLMResponse:
        ...

    def format_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        ...
