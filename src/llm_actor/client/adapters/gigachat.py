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
            return LLMServiceOverloadedError(str(exc) or "GigaChat: слишком много запросов")
        if isinstance(exc, AuthenticationError):
            return LLMServiceGeneralError(str(exc) or "GigaChat: ошибка авторизации")
        if isinstance(exc, GigaChatException):
            return LLMServiceUnavailableError(str(exc) or "GigaChat: ошибка сервиса")
    except ImportError:
        pass
    return exc


class GigaChatAdapter(ToolCapableClientInterface):
    """
    Адаптер для GigaChat SDK от Сбера.
    Реализует LLMClientInterface и ToolCapableClientInterface.
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
                "Пакет 'gigachat' не установлен. Пожалуйста, выполните 'pip install gigachat'."
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

        # Если есть история (из Tool Loop), используем ее
        if conversation:
            for m in conversation:
                msgs.append(Messages(role=m["role"], content=m.get("content") or ""))
        elif request.messages:
            for m in request.messages:
                msgs.append(Messages(role=m.get("role", MessagesRole.USER), content=m["content"]))

        if request.prompt:
            msgs.append(Messages(role=MessagesRole.USER, content=request.prompt))

        return msgs

    async def generate_async(self, request: LLMRequest) -> str:
        try:
            from gigachat.models import Chat

            msgs = self._convert_messages(request)

            chat_obj = Chat(
                messages=msgs,
                model=self._model,
                temperature=_gigachat_temperature(request),
                max_tokens=request.max_tokens,
            )
            response = await self._client.achat(chat_obj)
            return cast(str, response.choices[0].message.content)
        except LLMServiceError:
            raise
        except Exception as exc:
            raise _map_gigachat_exception(exc) from exc

    async def generate_with_tools_async(
        self, request: LLMRequest, conversation: list[dict[str, Any]]
    ) -> LLMResponse:
        try:
            from gigachat.models import Chat, Function

            msgs = self._convert_messages(request, conversation)

            functions = []
            if request.tools:
                for tool in cast(list[Tool], request.tools):
                    # GigaChat ожидает схему параметров напрямую в словаре
                    schema = tool.build_openai_schema()["function"]
                    functions.append(
                        Function(
                            name=schema["name"],
                            description=schema["description"],
                            parameters=schema["parameters"],
                        )
                    )

            chat_obj = Chat(
                messages=msgs,
                model=self._model,
                temperature=_gigachat_temperature(request),
                max_tokens=request.max_tokens,
                functions=functions if functions else None,
            )

            response = await self._client.achat(chat_obj)
            choice = response.choices[0]
            msg = choice.message

            tool_calls = []
            
            # 1. Проверяем стандартный function_call от SDK
            if msg.function_call:
                import json
                fc = msg.function_call
                args = json.loads(fc.arguments) if isinstance(fc.arguments, str) else fc.arguments
                tool_calls.append(
                    ToolCall(
                        id=f"call_{fc.name}",
                        name=fc.name,
                        arguments=args,
                    )
                )
            
            # 2. Если пусто, пробуем распарсить <fuse> теги (специфика GigaChat-Max)
            content = msg.content or ""
            if not tool_calls and "<fuse>" in content:
                import re
                import json
                # Ищем <fuse>name(args)</fuse> или <fuse>name</fuse>
                matches = re.finditer(r"<fuse>(.*?)(?:\((.*?)\))?</fuse>", content)
                for match in matches:
                    name = match.group(1).strip()
                    args_str = match.group(2).strip() if match.group(2) else ""
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except:
                        args = {}
                    
                    tool_calls.append(ToolCall(
                        id=f"fuse_{name}",
                        name=name,
                        arguments=args
                    ))

            # Формируем assistant_message для истории
            assistant_msg = {"role": "assistant", "content": msg.content}
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
        except LLMServiceError:
            raise
        except Exception as exc:
            raise _map_gigachat_exception(exc) from exc

    def format_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        # GigaChat использует role="function" для результатов
        return [{"role": "function", "name": r.name, "content": r.result} for r in results]

    async def close(self) -> None:
        await self._client.aclose()
