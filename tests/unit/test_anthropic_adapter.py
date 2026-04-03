from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("anthropic")

from anthropic import APIStatusError, APITimeoutError, RateLimitError

from llm_actor import LLMActorService
from llm_actor.client.adapters.anthropic import AnthropicAdapter
from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import (
    LLMServiceGeneralError,
    LLMServiceOverloadedError,
    LLMServiceTimeoutError,
)


def _make_rate_limit_error() -> RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    return RateLimitError("rate limited", response=response, body=None)


def _make_api_status_error(status_code: int) -> APIStatusError:
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    return APIStatusError("error", response=response, body=None)


@pytest.mark.asyncio
async def test_anthropic_adapter_maps_rate_limit_to_overloaded() -> None:
    adapter = AnthropicAdapter(api_key="sk-test", model="claude-3-5-sonnet-20241022")

    async def raise_rl(**_kwargs: object) -> None:
        raise _make_rate_limit_error()

    with patch.object(adapter._client.messages, "create", new=AsyncMock(side_effect=raise_rl)):
        with pytest.raises(LLMServiceOverloadedError):
            await adapter.generate_async(LLMRequest(prompt="x"))
    await adapter.close()


@pytest.mark.asyncio
async def test_anthropic_adapter_maps_timeout_to_domain_error() -> None:
    adapter = AnthropicAdapter(api_key="sk-test", model="claude-3-5-sonnet-20241022")

    async def raise_timeout(**_kwargs: object) -> None:
        raise APITimeoutError(MagicMock())

    with patch.object(adapter._client.messages, "create", new=AsyncMock(side_effect=raise_timeout)):
        with pytest.raises(LLMServiceTimeoutError):
            await adapter.generate_async(LLMRequest(prompt="x"))
    await adapter.close()


@pytest.mark.asyncio
async def test_anthropic_adapter_passes_params_to_sdk() -> None:
    adapter = AnthropicAdapter(api_key="sk-test", model="claude-3-5-sonnet-20241022")
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        block = MagicMock()
        block.text = "response text"
        message = MagicMock()
        message.content = [block]
        return message

    with patch.object(adapter._client.messages, "create", new=AsyncMock(side_effect=fake_create)):
        out = await adapter.generate_async(
            LLMRequest(
                prompt="Hello",
                temperature=0.5,
                system_prompt="Отвечай кратко.",
                extra={"top_k": 10},
            )
        )
    assert out == "response text"
    assert captured["temperature"] == 0.5
    assert captured["system"] == "Отвечай кратко."
    assert captured["top_k"] == 10
    assert captured["model"] == "claude-3-5-sonnet-20241022"
    await adapter.close()


@pytest.mark.asyncio
async def test_anthropic_adapter_extra_does_not_overwrite_mandatory_fields() -> None:
    adapter = AnthropicAdapter(api_key="sk-test", model="claude-3-5-sonnet-20241022")
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        block = MagicMock()
        block.text = "ok"
        message = MagicMock()
        message.content = [block]
        return message

    with patch.object(adapter._client.messages, "create", new=AsyncMock(side_effect=fake_create)):
        await adapter.generate_async(
            LLMRequest(prompt="Hello", extra={"model": "evil-model", "messages": []})
        )
    assert captured["model"] == "claude-3-5-sonnet-20241022"
    messages = captured["messages"]
    assert isinstance(messages, list) and len(messages) > 0
    await adapter.close()


@pytest.mark.asyncio
async def test_anthropic_adapter_stop_sequences_none_not_sent() -> None:
    adapter = AnthropicAdapter(api_key="sk-test", model="claude-3-5-sonnet-20241022")
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        block = MagicMock()
        block.text = "ok"
        message = MagicMock()
        message.content = [block]
        return message

    with patch.object(adapter._client.messages, "create", new=AsyncMock(side_effect=fake_create)):
        await adapter.generate_async(LLMRequest(prompt="x", stop_sequences=None))
    assert "stop_sequences" not in captured
    await adapter.close()


@pytest.mark.asyncio
async def test_anthropic_adapter_stop_sequences_empty_list_sent() -> None:
    adapter = AnthropicAdapter(api_key="sk-test", model="claude-3-5-sonnet-20241022")
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        block = MagicMock()
        block.text = "ok"
        message = MagicMock()
        message.content = [block]
        return message

    with patch.object(adapter._client.messages, "create", new=AsyncMock(side_effect=fake_create)):
        await adapter.generate_async(LLMRequest(prompt="x", stop_sequences=[]))
    assert captured["stop_sequences"] == []
    await adapter.close()


@pytest.mark.asyncio
async def test_anthropic_adapter_non_text_block_raises_descriptive_error() -> None:
    adapter = AnthropicAdapter(api_key="sk-test", model="claude-3-5-sonnet-20241022")

    async def fake_create(**_kwargs: object) -> MagicMock:
        tool_block = MagicMock(spec=[])  # нет атрибута text
        message = MagicMock()
        message.content = [tool_block]
        return message

    with patch.object(adapter._client.messages, "create", new=AsyncMock(side_effect=fake_create)):
        with pytest.raises(LLMServiceGeneralError, match="нетекстовый ответ"):
            await adapter.generate_async(LLMRequest(prompt="x"))
    await adapter.close()


@pytest.mark.asyncio
async def test_from_anthropic_factory_builds_service() -> None:
    mock_base = MagicMock()
    mock_base.close = AsyncMock()
    with patch(
        "llm_actor.client.adapters.anthropic.AnthropicAdapter", return_value=mock_base
    ) as ctor_mock:
        svc = LLMActorService.from_anthropic(api_key="k", model="claude-3-5-sonnet-20241022")
    ctor_mock.assert_called_once_with(api_key="k", model="claude-3-5-sonnet-20241022")
    assert svc._base_client is mock_base
