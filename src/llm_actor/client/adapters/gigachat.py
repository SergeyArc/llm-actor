import json
import logging
from typing import Any, cast

from llm_actor.client.interface import ToolCapableClientInterface
from llm_actor.core.request import LLMRequest
from llm_actor.core.tools import LLMResponse, Tool, ToolCall, ToolResult
from llm_actor.exceptions import (
    LLMServiceError,
    LLMServiceGeneralError,
    LLMServiceOverloadedError,
    LLMServiceUnavailableError,
)

logger = logging.getLogger(__name__)


def _gigachat_temperature(request: LLMRequest) -> float:
    return 1.0 if request.temperature is None else request.temperature


def _map_gigachat_exception(exc: Exception) -> Exception:
    try:
        from gigachat.exceptions import (
            AuthenticationError,
            GigaChatException,
            RateLimitError,
        )

        if isinstance(exc, RateLimitError):
            return LLMServiceOverloadedError(str(exc) or "GigaChat: too many requests")
        if isinstance(exc, AuthenticationError):
            return LLMServiceGeneralError(str(exc) or "GigaChat: authentication failed")
        if isinstance(exc, GigaChatException):
            return LLMServiceUnavailableError(str(exc) or "GigaChat: service error")
    except ImportError:
        pass
    return exc


class GigaChatAdapter(ToolCapableClientInterface):
    """
    GigaChat SDK adapter implementing LLMClientInterface and ToolCapableClientInterface.
    """

    def __init__(
        self,
        *,
        credentials: str | None = None,
        scope: str | None = None,
        verify_ssl_certs: bool = True,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from gigachat import GigaChat
        except ImportError as err:
            raise ImportError(
                "Package 'gigachat' is not installed. Run 'pip install gigachat'."
            ) from err

        self._model = model
        self._client = GigaChat(
            credentials=credentials,
            scope=scope,
            verify_ssl_certs=verify_ssl_certs,
            **kwargs,
        )

    def _convert_messages(
        self, request: LLMRequest, conversation: list[dict[str, Any]] | None = None
    ) -> list[Any]:
        from gigachat.models import Messages, MessagesRole

        msgs = []
        if request.system_prompt:
            msgs.append(Messages(role=MessagesRole.SYSTEM, content=request.system_prompt))

        # Tool-loop history takes precedence when present
        if conversation:
            for m in conversation:
                msgs.append(
                    Messages(
                        role=m["role"],
                        content=m.get("content") or "",
                        name=m.get("name"),
                        function_call=m.get("function_call"),
                    )
                )
        elif request.messages:
            for m in request.messages:
                msgs.append(Messages(role=m.get("role", MessagesRole.USER), content=m["content"]))

        if request.prompt:
            msgs.append(Messages(role=MessagesRole.USER, content=request.prompt))

        return msgs

    def _warn_extra_headers_if_needed(self, request: LLMRequest) -> None:
        if request.extra_headers:
            logger.warning(
                "GigaChatAdapter: extra_headers не поддерживаются и будут проигнорированы"
            )

    def _gigachat_functions_from_request(self, request: LLMRequest) -> list[Any]:
        from gigachat.models import Function

        if not request.tools:
            return []
        functions: list[Any] = []
        for tool in cast(list[Tool], request.tools):
            schema = tool.build_openai_schema()["function"]
            functions.append(
                Function(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=schema["parameters"],
                )
            )
        return functions

    def _gigachat_tool_response(self, msg: Any) -> LLMResponse:
        tool_calls: list[ToolCall] = []
        if msg.function_call:
            fc = msg.function_call
            raw_args = (
                json.loads(fc.arguments) if isinstance(fc.arguments, str) else fc.arguments
            )
            tool_args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            tool_calls.append(
                ToolCall(
                    id=f"call_{fc.name}",
                    name=fc.name,
                    arguments=tool_args,
                )
            )
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.function_call:
            assistant_msg["function_call"] = {
                "name": msg.function_call.name,
                "arguments": msg.function_call.arguments,
            }
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            assistant_message=assistant_msg,
        )

    async def generate_async(self, request: LLMRequest) -> str:
        self._warn_extra_headers_if_needed(request)
        try:
            from gigachat.models import Chat

            msgs = self._convert_messages(request)

            chat_kwargs: dict[str, Any] = {
                **request.extra,
                "messages": msgs,
                "model": self._model,
                "temperature": _gigachat_temperature(request),
            }
            if request.max_tokens is not None:
                chat_kwargs["max_tokens"] = request.max_tokens
            if request.top_p is not None:
                chat_kwargs["top_p"] = request.top_p
            if request.stop_sequences is not None:
                chat_kwargs["stop"] = request.stop_sequences

            chat_obj = Chat(**chat_kwargs)
            response = await self._client.achat(chat_obj)
            return response.choices[0].message.content
        except LLMServiceError:
            raise
        except Exception as exc:
            raise _map_gigachat_exception(exc) from exc

    async def generate_with_tools_async(
        self, request: LLMRequest, conversation: list[dict[str, Any]]
    ) -> LLMResponse:
        self._warn_extra_headers_if_needed(request)
        try:
            from gigachat.models import Chat

            msgs = self._convert_messages(request, conversation)
            functions = self._gigachat_functions_from_request(request)
            chat_kwargs: dict[str, Any] = {
                **request.extra,
                "messages": msgs,
                "model": self._model,
                "temperature": _gigachat_temperature(request),
                "functions": functions if functions else None,
            }
            if request.max_tokens is not None:
                chat_kwargs["max_tokens"] = request.max_tokens
            if request.top_p is not None:
                chat_kwargs["top_p"] = request.top_p
            if request.stop_sequences is not None:
                chat_kwargs["stop"] = request.stop_sequences

            response = await self._client.achat(Chat(**chat_kwargs))
            return self._gigachat_tool_response(response.choices[0].message)
        except LLMServiceError:
            raise
        except Exception as exc:
            raise _map_gigachat_exception(exc) from exc

    def format_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        # GigaChat uses role="function" for tool results
        return [{"role": "function", "name": r.name, "content": r.result} for r in results]

    async def close(self) -> None:
        await self._client.aclose()
