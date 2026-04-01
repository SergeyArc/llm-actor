from typing import Any

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    RateLimitError,
)

from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import (
    LLMServiceGeneralError,
    LLMServiceHTTPError,
    LLMServiceOverloadedError,
    LLMServiceTimeoutError,
    LLMServiceUnavailableError,
)


def _map_anthropic_exception(exc: Exception) -> Exception:
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


class AnthropicAdapter:
    """Адаптер Async Anthropic SDK с маппингом ошибок в доменные исключения брокера."""

    def __init__(self, *, api_key: str, model: str, **client_options: Any) -> None:
        self._model = model
        self._client = AsyncAnthropic(api_key=api_key, **client_options)

    async def generate_async(self, request: LLMRequest) -> str:
        max_tokens = request.max_tokens if request.max_tokens is not None else 4096

        # extra применяется первым — обязательные поля всегда побеждают
        payload: dict[str, Any] = dict(request.extra)
        payload["model"] = self._model
        payload["max_tokens"] = max_tokens
        payload["messages"] = [{"role": "user", "content": request.prompt}]
        if request.system_prompt is not None:
            payload["system"] = request.system_prompt
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop_sequences is not None:
            payload["stop_sequences"] = request.stop_sequences

        try:
            message = await self._client.messages.create(**payload)
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
            raise _map_anthropic_exception(exc) from exc

        parts: list[str] = []
        for block in message.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        text = "".join(parts).strip()
        if not text:
            if message.content and not parts:
                block_types = [type(b).__name__ for b in message.content]
                raise LLMServiceGeneralError(
                    f"Anthropic вернул нетекстовый ответ (типы блоков: {block_types})"
                )
            raise LLMServiceGeneralError("Пустой ответ от Anthropic")
        return text

    async def close(self) -> None:
        await self._client.close()
