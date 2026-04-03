from typing import Any, cast

try:
    from gigachat import GigaChat
    from gigachat.exceptions import AuthenticationError, GigaChatException, RateLimitError
except ImportError:
    # Ожидаем, что пользователь установит gigachat для использования этого адаптера
    pass

from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import (
    LLMServiceGeneralError,
    LLMServiceOverloadedError,
    LLMServiceUnavailableError,
)


def _map_gigachat_exception(exc: Exception) -> Exception:
    if "RateLimitError" in str(type(exc)):
        return LLMServiceOverloadedError(str(exc) or "GigaChat: слишком много запросов")
    if "AuthenticationError" in str(type(exc)):
        return LLMServiceGeneralError(str(exc) or "GigaChat: ошибка авторизации")
    if "GigaChatException" in str(type(exc)):
        return LLMServiceUnavailableError(str(exc) or "GigaChat: ошибка сервиса")
    return exc


class GigaChatAdapter:
    """
    Адаптер для GigaChat SDK от Сбера.
    Реализует LLMClientInterface для использования в llm_actor.

    Требует установленного пакета gigachat.
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
        """
        Инициализация адаптера GigaChat.

        Args:
            credentials: Ключ авторизации (или GIGACHAT_CREDENTIALS в окружении)
            scope: Область доступа (GIGACHAT_API_PERS / GIGACHAT_API_CORP)
            verify_ssl_certs: Проверка SSL сертификатов (установите False для отладки)
            model: Название модели (например, GigaChat-Pro)
            **kwargs: Дополнительные параметры для GigaChat SDK
        """
        try:
            from gigachat import GigaChat
        except ImportError:
            raise ImportError(
                "Пакет 'gigachat' не установлен. Пожалуйста, выполните 'pip install gigachat'."
            )

        self._model = model
        self._client = GigaChat(
            credentials=credentials,
            scope=scope,
            verify_ssl_certs=verify_ssl_certs,
            **kwargs,
        )

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        if request.messages:
            messages.extend(request.messages)
        if request.prompt:
            messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "messages": messages,
        }
        if self._model:
            payload["model"] = self._model
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        # Объединяем с дополнительными параметрами из request.extra
        if request.extra:
            payload.update(request.extra)

        return payload

    async def generate_async(self, request: LLMRequest) -> str:
        try:
            from gigachat.models import Chat, Messages, MessagesRole

            msgs = []
            if request.system_prompt:
                msgs.append(Messages(role=MessagesRole.SYSTEM, content=request.system_prompt))
            if request.messages:
                for m in request.messages:
                    role = m.get("role", MessagesRole.USER)
                    msgs.append(Messages(role=role, content=m["content"]))
            if request.prompt:
                msgs.append(Messages(role=MessagesRole.USER, content=request.prompt))

            chat_obj = Chat(
                messages=msgs,
                model=self._model,
                temperature=request.temperature or 1.0,
                max_tokens=request.max_tokens,
            )
            
            # Добавляем extra поля если они поддерживаются Chat
            for k, v in request.extra.items():
                if hasattr(chat_obj, k):
                    setattr(chat_obj, k, v)

            response = await self._client.achat(chat_obj)
            
            if not response.choices:
                raise LLMServiceGeneralError("GigaChat: пустой ответ (нет choices)")

            content = response.choices[0].message.content
            if content is None:
                raise LLMServiceGeneralError("GigaChat: пустой контент в ответе")

            return cast(str, content)

        except Exception as exc:
            raise _map_gigachat_exception(exc) from exc

    async def close(self) -> None:
        """Закрытие сессии клиента."""
        await self._client.aclose()
