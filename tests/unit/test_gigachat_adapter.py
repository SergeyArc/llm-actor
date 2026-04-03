import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from llm_actor.client.adapters.gigachat import GigaChatAdapter
from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import LLMServiceOverloadedError, LLMServiceGeneralError

# Пропускаем тесты, если gigachat не установлен (хотя в dev группе он должен быть)
pytest.importorskip("gigachat")

@pytest.mark.asyncio
async def test_gigachat_adapter_build_payload_basic():
    # Нам нужно мокать саму библиотеку gigachat внутри __init__
    with patch("gigachat.GigaChat"):
        adapter = GigaChatAdapter(credentials="test_creds", model="GigaChat-Pro")
        
        req = LLMRequest(
            prompt="Hello",
            system_prompt="Be kind",
            temperature=0.5,
            max_tokens=100
        )
        
        # Проверяем внутренний метод _build_payload (хотя generate_async важнее)
        # В GigaChat vLLM/SDK обычно используется объект Chat
        pass

@pytest.mark.asyncio
async def test_gigachat_adapter_generate_async_success():
    with patch("gigachat.GigaChat") as mock_class:
        mock_client = mock_class.return_value
        mock_client.achat = AsyncMock()
        
        # Настраиваем фейковый ответ GigaChat
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "GigaResponse"
        mock_response.choices = [mock_choice]
        mock_client.achat.return_value = mock_response
        
        adapter = GigaChatAdapter(credentials="test", model="GigaChat-Pro")
        res = await adapter.generate_async(LLMRequest(prompt="test"))
        
        assert res == "GigaResponse"
        assert mock_client.achat.called

@pytest.mark.asyncio
async def test_gigachat_adapter_maps_errors():
    from gigachat.exceptions import RateLimitError as GigaChatRateLimitError

    with patch("gigachat.GigaChat") as mock_class:
        mock_client = mock_class.return_value
        rate_exc = GigaChatRateLimitError(
            url="https://example.invalid",
            status_code=429,
            content=b"Too many requests",
            headers=None,
        )
        mock_client.achat = AsyncMock(side_effect=rate_exc)

        adapter = GigaChatAdapter(credentials="test")

        with pytest.raises(LLMServiceOverloadedError):
            await adapter.generate_async(LLMRequest(prompt="test"))

@pytest.mark.asyncio
async def test_from_gigachat_factory():
    with patch("llm_actor.client.adapters.gigachat.GigaChatAdapter") as mock_adapter:
        from llm_actor import LLMActorService
        svc = LLMActorService.from_gigachat(credentials="abc", model="pro")
        assert svc is not None
        mock_adapter.assert_called_once()
