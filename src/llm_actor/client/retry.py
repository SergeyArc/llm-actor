import asyncio
from typing import Any

from llm_actor.client.interface import (
    LLMClientWithCircuitBreakerInterface,
)
from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import (
    LLMServiceHTTPError,
    LLMServiceOverloadedError,
    LLMServiceTimeoutError,
    LLMServiceUnavailableError,
)
from llm_actor.logger import ActorLogger
from llm_actor.settings import LLMActorSettings


def _is_transient_error(exc: Exception) -> bool:
    """
    Return True if the exception is transient and the request may be retried.

    Args:
        exc: Exception to classify.

    Returns:
        True if the error is transient.
    """
    if isinstance(exc, LLMServiceOverloadedError):
        return True
    if isinstance(exc, LLMServiceUnavailableError):
        return True
    if isinstance(exc, LLMServiceTimeoutError):
        return True
    if isinstance(exc, LLMServiceHTTPError):
        transient_status_codes = {502, 503, 504}
        return exc.status_code in transient_status_codes
    return False


class LLMClientWithRetry:
    """
    Transport wrapper that retries the inner LLM client on transient API errors.
    """

    def __init__(
        self,
        base_client: LLMClientWithCircuitBreakerInterface,
        settings: LLMActorSettings,
    ) -> None:
        if settings.LLM_RETRY_MAX_ATTEMPTS < 1:
            raise ValueError(
                f"LLM_RETRY_MAX_ATTEMPTS must be >= 1, got {settings.LLM_RETRY_MAX_ATTEMPTS}"
            )
        self._client = base_client
        self._max_attempts = settings.LLM_RETRY_MAX_ATTEMPTS
        self._base_backoff = settings.LLM_RETRY_BASE_BACKOFF
        self._backoff_cap = settings.LLM_RETRY_BACKOFF_CAP
        self._logger = ActorLogger.get_logger(name="llm_client_retry")

    async def generate(
        self, request: LLMRequest, response_model: type[Any] | None = None
    ) -> Any | str:
        """
        Generate with automatic retries on transient API errors.

        Args:
            request: Generation parameters.
            response_model: Optional Pydantic model for response validation.

        Returns:
            Validated model instance if response_model is set; otherwise string.

        Raises:
            Exception: After all attempts are exhausted or on non-transient errors.
        """
        return await self._generate_with_transport_retry(request, response_model)

    async def _generate_with_transport_retry(
        self, request: LLMRequest, response_model: type[Any] | None = None
    ) -> Any | str:
        for api_attempt in range(1, self._max_attempts + 1):
            try:
                return await self._client.generate(request, response_model)
            except Exception as exc:
                if not _is_transient_error(exc):
                    self._logger.debug(f"Non-retryable error: {type(exc).__name__}: {exc}")
                    raise

                if api_attempt >= self._max_attempts:
                    self._logger.error(
                        f"Failed after {self._max_attempts} API attempts, "
                        f"last error: {type(exc).__name__}: {exc}"
                    )
                    raise

                backoff = min(
                    self._base_backoff * (2 ** (api_attempt - 1)),
                    self._backoff_cap,
                )
                self._logger.info(
                    f"Transient error {type(exc).__name__} "
                    f"(API attempt {api_attempt}/{self._max_attempts}), "
                    f"retrying after {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)

        raise RuntimeError("unreachable: retry loop exhausted without return or raise")
