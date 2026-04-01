import asyncio
import json
from typing import Any

from pydantic import ValidationError

from llm_broker.client.interface import (
    LLMClientWithCircuitBreakerInterface,
)
from llm_broker.exceptions import (
    LLMServiceHTTPError,
    LLMServiceOverloadedError,
    LLMServiceUnavailableError,
)
from llm_broker.logger import BrokerLogger
from llm_broker.settings import LLMBrokerSettings


def _is_transient_error(exc: Exception) -> bool:
    """
    Проверяет, является ли ошибка временной (transient) и подлежит ли retry.

    Args:
        exc: Исключение для проверки

    Returns:
        True если ошибка временная и можно повторить запрос
    """
    if isinstance(exc, LLMServiceOverloadedError):
        return True
    if isinstance(exc, LLMServiceUnavailableError):
        return True
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, LLMServiceHTTPError):
        transient_status_codes = {502, 503, 504}
        return exc.status_code in transient_status_codes
    return False


def _is_validation_error(exc: Exception) -> bool:
    """
    Проверяет, является ли ошибка ошибкой валидации/парсинга.

    Args:
        exc: Исключение для проверки

    Returns:
        True если ошибка связана с валидацией/парсингом
    """
    return isinstance(exc, (ValueError, ValidationError, json.JSONDecodeError))


class LLMClientWithRetry:
    """
    Универсальная обертка над LLM клиентом с поддержкой retry логики.

    Автоматически обрабатывает два типа ошибок независимо:
    - Временные API ошибки (429, 503, 502, 504, timeout) - с exponential backoff
    - Ошибки валидации/парсинга (ValueError, ValidationError, JSONDecodeError) - с немедленным повтором

    Счетчики попыток для каждого типа ошибок независимы, что позволяет обрабатывать
    оба типа ошибок в одном запросе.
    """

    def __init__(
        self,
        base_client: LLMClientWithCircuitBreakerInterface,
        settings: LLMBrokerSettings,
    ) -> None:
        self._client = base_client
        self._settings = settings
        self._max_attempts = settings.LLM_RETRY_MAX_ATTEMPTS
        self._max_validation_attempts = settings.LLM_VALIDATION_RETRY_MAX_ATTEMPTS
        self._base_backoff = settings.LLM_RETRY_BASE_BACKOFF
        self._backoff_cap = settings.LLM_RETRY_BACKOFF_CAP
        self._logger = BrokerLogger.get_logger(name="llm_client_retry")

    async def generate(self, prompt: str, response_model: type[Any] | None = None) -> Any | str:
        """
        Генерирует ответ от LLM с автоматическими повторами при ошибках API и валидации.

        Args:
            prompt: Текст промпта для генерации
            response_model: Опциональная Pydantic модель для валидации ответа

        Returns:
            Если response_model указан - валидированный объект модели, иначе строка

        Raises:
            ValueError, ValidationError, json.JSONDecodeError: При ошибках после исчерпания попыток
        """
        return await self._generate_with_unified_retry(prompt, response_model)

    async def _generate_with_unified_retry(
        self, prompt: str, response_model: type[Any] | None = None
    ) -> Any | str:
        """
        Универсальный retry для API ошибок и ошибок валидации.

        API ошибки обрабатываются с exponential backoff.
        Ошибки валидации обрабатываются немедленным повтором.
        Счетчики попыток для каждого типа ошибок независимы.
        """
        api_attempts = 0
        validation_attempts = 0

        while True:
            try:
                return await self._client.generate(prompt, response_model)

            except Exception as exc:
                if _is_transient_error(exc):
                    api_attempts += 1
                    if api_attempts >= self._max_attempts:
                        self._logger.error(
                            f"Failed after {self._max_attempts} API attempts, "
                            f"last error: {type(exc).__name__}: {exc}"
                        )
                        raise

                    backoff = min(
                        self._base_backoff * (2 ** (api_attempts - 1)),
                        self._backoff_cap,
                    )
                    self._logger.info(
                        f"Transient error {type(exc).__name__} "
                        f"(API attempt {api_attempts}/{self._max_attempts}), "
                        f"retrying after {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                    continue

                if _is_validation_error(exc):
                    validation_attempts += 1
                    if validation_attempts >= self._max_validation_attempts:
                        self._logger.error(
                            f"Failed after {self._max_validation_attempts} validation attempts"
                        )
                        raise

                    self._logger.warning(
                        f"[Validation attempt {validation_attempts}/{self._max_validation_attempts}] "
                        f"Error: {type(exc).__name__}: {exc}"
                    )
                    await asyncio.sleep(0.1)
                    continue

                self._logger.debug(f"Non-retryable error: {type(exc).__name__}: {exc}")
                raise
