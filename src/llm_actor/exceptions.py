from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_actor.core.messages import ActorMessage


class LLMActorError(Exception):
    """Base exception for all llm_actor errors."""


class LLMServiceError(LLMActorError):
    """Base exception for LLM service errors."""

    def __init__(self, message: str, status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class LLMServiceUnavailableError(LLMServiceError):
    """Raised when the LLM service is unavailable."""

    def __init__(self, message: str = "LLM service unavailable"):
        super().__init__(
            message=message,
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )


class LLMServiceOverloadedError(LLMServiceError):
    """Raised when the LLM service is overloaded."""

    def __init__(self, message: str = "LLM service overloaded"):
        super().__init__(
            message=message,
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
        )


class LLMServiceHTTPError(LLMServiceError):
    """Raised on HTTP errors from the LLM service."""

    def __init__(self, message: str, status_code: int):
        super().__init__(
            message=message,
            status_code=status_code,
        )


class LLMServiceGeneralError(LLMServiceError):
    """Raised for general LLM service failures."""

    def __init__(self, message: str = "LLM service error"):
        super().__init__(
            message=message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


class LLMServiceTimeoutError(LLMServiceError):
    """Raised when an LLM request times out."""

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
