from typing import Any, Protocol, runtime_checkable

from llm_actor.core.request import LLMRequest
from llm_actor.core.tools import LLMResponse, ToolResult


@runtime_checkable
class LLMClientInterface(Protocol):
    """
    Contract for external LLM clients used by llm_actor.

    IMPORTANT: Implementations must NOT implement their own transport retry.
    Exponential backoff retries are provided by the broker via LLMClientWithRetry
    inside LLMActorService.

    On failure, raise the appropriate exceptions:
    - LLMServiceOverloadedError (429) — API rate limit / overload
    - LLMServiceUnavailableError (503) — service unavailable
    - LLMServiceHTTPError (502, 503, 504) — transient HTTP errors
    - LLMServiceTimeoutError — transport timeouts to the LLM

    LLMClientWithRetry will handle these with exponential backoff at the broker level.
    """

    async def generate_async(self, request: LLMRequest) -> str:
        """
        Produce a single LLM completion for the given request.

        Perform one HTTP call to the LLM API without retry logic; the broker handles retries.

        Args:
            request: Generation parameters (prompt, temperature, extra, etc.)

        Returns:
            Non-empty response text from the LLM.

        Raises:
            Exception: On LLM API errors. Transient errors (429, 503, 502, 504, timeout)
                are retried by the broker; do not add retry inside the client.
        """
        ...


@runtime_checkable
class LLMClientWithCircuitBreakerInterface(Protocol):
    """
    LLM client contract used inside llm_actor (behind circuit breaker).

    Validation errors (ValueError, ValidationError, JSONDecodeError) are handled
    by LLMClientWithRetry with automatic regeneration attempts.
    """

    async def generate(
        self, request: LLMRequest, response_model: type[Any] | None = None
    ) -> Any | str:
        """
        Generate a response with optional Pydantic validation.

        Args:
            request: Generation parameters.
            response_model: Optional Pydantic model to validate the response.

        Returns:
            Validated model instance if response_model is set; otherwise raw string.

        Raises:
            Exception: On LLM or validation errors. Validation failures may be retried
                by LLMClientWithRetry with new LLM calls.
        """
        ...


@runtime_checkable
class ToolCapableClientInterface(Protocol):
    async def generate_with_tools_async(
        self,
        request: LLMRequest,
        conversation: list[dict[str, Any]],
    ) -> LLMResponse: ...

    def format_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]: ...
