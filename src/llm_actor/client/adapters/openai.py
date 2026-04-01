from typing import Any, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import (
    LLMServiceGeneralError,
    LLMServiceHTTPError,
    LLMServiceOverloadedError,
    LLMServiceTimeoutError,
    LLMServiceUnavailableError,
)


def _map_openai_exception(exc: Exception) -> Exception:
    if isinstance(exc, RateLimitError):
        return LLMServiceOverloadedError(str(exc) or "LLM сервис перегружен")
    if isinstance(exc, APITimeoutError):
        return LLMServiceTimeoutError(str(exc) or "LLM request timed out")
    if isinstance(exc, APIConnectionError):
        return LLMServiceUnavailableError(str(exc) or "Ошибка соединения с LLM")
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        if code == 503:
            return LLMServiceUnavailableError(str(exc) or "LLM сервис недоступен")
        if code in (502, 504):
            return LLMServiceHTTPError(str(exc) or "LLM HTTP error", status_code=code)
        return LLMServiceGeneralError(str(exc) or "Ошибка LLM сервиса")
    return exc


class OpenAIAdapter:
    """Адаптер Async OpenAI SDK с маппингом ошибок в доменные исключения брокера."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        **client_options: Any,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, **client_options)

    async def generate_async(self, request: LLMRequest) -> str:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        # extra применяется первым — обязательные поля всегда побеждают
        payload: dict[str, Any] = dict(request.extra)
        payload["model"] = self._model
        payload["messages"] = messages
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop_sequences is not None:
            payload["stop"] = request.stop_sequences

        try:
            completion = await self._client.chat.completions.create(**payload)
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
            raise _map_openai_exception(exc) from exc

        if not completion.choices:
            raise LLMServiceGeneralError("Пустой ответ от OpenAI: нет choices")
        choice = completion.choices[0]
        content = choice.message.content
        if content is None:
            raise LLMServiceGeneralError("Пустой ответ от OpenAI")
        return cast(str, content)

    async def close(self) -> None:
        await self._client.close()
