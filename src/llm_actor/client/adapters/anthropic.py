from typing import Any, cast

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    RateLimitError,
)

from llm_actor.core.request import LLMRequest
from llm_actor.core.tools import LLMResponse, Tool, ToolCall, ToolResult
from llm_actor.exceptions import (
    LLMServiceGeneralError,
    LLMServiceHTTPError,
    LLMServiceOverloadedError,
    LLMServiceTimeoutError,
    LLMServiceUnavailableError,
)


def _map_anthropic_exception(exc: Exception) -> Exception:
    if isinstance(exc, RateLimitError):
        return LLMServiceOverloadedError(str(exc) or "LLM service overloaded")
    if isinstance(exc, APITimeoutError):
        return LLMServiceTimeoutError(str(exc) or "LLM request timed out")
    if isinstance(exc, APIConnectionError):
        return LLMServiceUnavailableError(str(exc) or "LLM connection error")
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        if code == 503:
            return LLMServiceUnavailableError(str(exc) or "LLM service unavailable")
        if code in (502, 504):
            return LLMServiceHTTPError(str(exc) or "LLM HTTP error", status_code=code)
        return LLMServiceGeneralError(str(exc) or "LLM service error")
    return exc


class AnthropicAdapter:
    """Async Anthropic SDK adapter with error mapping and tool calling."""

    def __init__(self, *, api_key: str, model: str, **client_options: Any) -> None:
        self._model = model
        self._client = AsyncAnthropic(api_key=api_key, **client_options)

    def _build_messages(
        self,
        request: LLMRequest,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.messages:
            messages.extend(request.messages)
        if request.prompt:
            messages.append({"role": "user", "content": request.prompt})
        if extra_messages:
            messages.extend(extra_messages)
        return messages

    async def generate_async(self, request: LLMRequest) -> str:
        max_tokens = request.max_tokens if request.max_tokens is not None else 4096
        messages = self._build_messages(request)

        payload: dict[str, Any] = dict(request.extra)
        payload["model"] = self._model
        payload["max_tokens"] = max_tokens
        payload["messages"] = messages
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
            reason = getattr(message, "stop_reason", "unknown")
            if message.content and not parts:
                block_types = [type(b).__name__ for b in message.content]
                raise LLMServiceGeneralError(
                    f"Anthropic returned a non-text response (stop_reason: {reason}, blocks: {block_types})"
                )
            raise LLMServiceGeneralError(f"Empty Anthropic response (stop_reason: {reason})")
        return text

    async def generate_with_tools_async(
        self,
        request: LLMRequest,
        conversation: list[dict[str, Any]],
    ) -> LLMResponse:
        max_tokens = request.max_tokens if request.max_tokens is not None else 4096
        messages = self._build_messages(request, extra_messages=conversation or None)
        resolved_tools = cast(list[Tool], request.tools or [])
        tools_schema = [tool.build_anthropic_schema() for tool in resolved_tools]

        payload: dict[str, Any] = dict(request.extra)
        payload["model"] = self._model
        payload["max_tokens"] = max_tokens
        payload["messages"] = messages
        payload["tools"] = tools_schema
        if request.system_prompt is not None:
            payload["system"] = request.system_prompt
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        try:
            message = await self._client.messages.create(**payload)
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
            raise _map_anthropic_exception(exc) from exc

        tool_use_blocks = [
            b for b in message.content if hasattr(b, "type") and b.type == "tool_use"
        ]
        if tool_use_blocks:
            tool_calls = [
                ToolCall(id=b.id, name=b.name, arguments=dict(b.input)) for b in tool_use_blocks
            ]
            all_blocks: list[dict[str, Any]] = []
            for b in message.content:
                if not hasattr(b, "type"):
                    continue
                if b.type == "tool_use":
                    all_blocks.append(
                        {"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)}
                    )
                elif b.type == "text" and hasattr(b, "text"):
                    all_blocks.append({"type": "text", "text": b.text})
            assistant_message: dict[str, Any] = {"role": "assistant", "content": all_blocks}
            return LLMResponse(
                content=None,
                tool_calls=tool_calls,
                assistant_message=assistant_message,
            )

        text_blocks = [b for b in message.content if hasattr(b, "text")]
        text = "".join(b.text for b in text_blocks).strip()
        return LLMResponse(
            content=text,
            assistant_message={"role": "assistant", "content": text},
        )

    def format_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.tool_call_id,
                        "content": (
                            (f"Error: {r.result}" if r.result else "Error (no message)")
                            if r.is_error
                            else (r.result or "(no output)")
                        ),
                        "is_error": r.is_error,
                    }
                    for r in results
                ],
            }
        ]

    async def close(self) -> None:
        await self._client.close()
