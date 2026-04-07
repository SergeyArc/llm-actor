import json
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from llm_actor.core.request import LLMRequest
from llm_actor.core.tools import LLMResponse, Tool, ToolCall, ToolResult
from llm_actor.exceptions import (
    LLMServiceGeneralError,
    LLMServiceHTTPError,
    LLMServiceOverloadedError,
    LLMServiceTimeoutError,
    LLMServiceUnavailableError,
)


def _map_openai_exception(exc: Exception) -> Exception:
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


class OpenAIAdapter:
    """Async OpenAI SDK adapter with error mapping and tool calling."""

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

    def _build_messages(
        self,
        request: LLMRequest,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        if request.messages:
            messages.extend(request.messages)
        if request.prompt:
            messages.append({"role": "user", "content": request.prompt})
        if extra_messages:
            messages.extend(extra_messages)
        return messages

    async def generate_async(self, request: LLMRequest) -> str:
        messages = self._build_messages(request)
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
            completion = await self._client.chat.completions.create(
                **payload,
                extra_headers=request.extra_headers,
            )
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
            raise _map_openai_exception(exc) from exc

        if not completion.choices:
            raise LLMServiceGeneralError("Empty OpenAI response: no choices")

        choice = completion.choices[0]
        content = choice.message.content

        if content is None:
            reason = getattr(choice, "finish_reason", "unknown")
            raise LLMServiceGeneralError(f"Empty LLM response (finish_reason: {reason})")

        return cast(str, content)

    async def generate_with_tools_async(
        self,
        request: LLMRequest,
        conversation: list[dict[str, Any]],
    ) -> LLMResponse:
        messages = self._build_messages(request, extra_messages=conversation or None)
        resolved_tools = cast(list[Tool], request.tools or [])
        tools_schema = [tool.build_openai_schema() for tool in resolved_tools]

        payload: dict[str, Any] = dict(request.extra)
        payload["model"] = self._model
        payload["messages"] = messages
        payload["tools"] = tools_schema

        # Unless overridden in request.extra
        if "tool_choice" not in payload:
            payload["tool_choice"] = "auto"
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        try:
            completion = await self._client.chat.completions.create(
                **payload,
                extra_headers=request.extra_headers,
            )
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
            raise _map_openai_exception(exc) from exc

        if not completion.choices:
            raise LLMServiceGeneralError("Empty OpenAI response: no choices")

        message = completion.choices[0].message

        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise LLMServiceGeneralError(
                        f"Malformed JSON in tool call arguments for '{tc.function.name}': {exc}"
                    ) from exc
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            return LLMResponse(
                content=None,
                tool_calls=tool_calls,
                assistant_message=assistant_message,
            )

        content = message.content
        if content is None:
            raise LLMServiceGeneralError("Empty OpenAI response during tool calling")
        return LLMResponse(
            content=content,
            assistant_message={"role": "assistant", "content": content},
        )

    def format_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        return [
            {
                "role": "tool",
                "tool_call_id": r.tool_call_id,
                "name": r.name,
                "content": f"Error: {r.result}" if r.is_error else r.result,
            }
            for r in results
        ]

    async def close(self) -> None:
        await self._client.close()
