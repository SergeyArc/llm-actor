from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_actor.core.messages import ActorMessage


class LLMActorError(Exception):
    """Базовое исключение для всех ошибок пакета llm_actor."""


class LLMServiceError(LLMActorError):
    """Базовое исключение для ошибок LLM сервиса."""

    def __init__(self, message: str, status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class LLMServiceUnavailableError(LLMServiceError):
    """Исключение, возникающее когда LLM сервис недоступен."""

    def __init__(self, message: str = "LLM сервис недоступен"):
        super().__init__(
            message=message,
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )


class LLMServiceOverloadedError(LLMServiceError):
    """Исключение, возникающее когда LLM сервис перегружен."""

    def __init__(self, message: str = "LLM сервис перегружен"):
        super().__init__(
            message=message,
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
        )


class LLMServiceHTTPError(LLMServiceError):
    """Исключение, возникающее при HTTP ошибках LLM сервиса."""

    def __init__(self, message: str, status_code: int):
        super().__init__(
            message=message,
            status_code=status_code,
        )


class LLMServiceGeneralError(LLMServiceError):
    """Исключение, возникающее при общих ошибках LLM сервиса."""

    def __init__(self, message: str = "Ошибка LLM сервиса"):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


class LLMServiceTimeoutError(LLMServiceError):
    """Исключение при таймауте запроса к LLM."""

    def __init__(self, message: str = "LLM request timed out"):
        super().__init__(
            message=message,
            status_code=HTTPStatus.GATEWAY_TIMEOUT,
        )


class CircuitBreakerOpenError(LLMActorError):
    """Raised when circuit breaker is open"""

    pass


class OverloadError(LLMActorError):
    """Raised when actor mailbox is full"""

    pass


class PoolShuttingDownError(LLMActorError):
    """Raised when pool is shutting down"""

    pass


class ActorFailedError(LLMActorError):
    """Raised when actor exceeded failure threshold and must be restarted."""

    def __init__(
        self,
        message: str,
        actor_id: str,
        pending_messages: list[ActorMessage[Any]] | None = None,
    ) -> None:
        self.actor_id = actor_id
        self.pending_messages: list[ActorMessage[Any]] = pending_messages or []
        super().__init__(message)


class ToolExecutionError(LLMActorError):
    def __init__(self, tool_name: str, cause: Exception) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"Tool '{tool_name}' execution failed: {cause}")


class ToolExecutionTimeoutError(ToolExecutionError):
    def __init__(self, tool_name: str, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(tool_name, TimeoutError(f"exceeded {timeout}s timeout"))


class ToolLoopMaxIterationsError(LLMActorError):
    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        super().__init__(f"Tool loop exceeded maximum iterations ({max_iterations})")
