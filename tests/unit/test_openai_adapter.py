from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openai")

from openai import RateLimitError

from llm_actor import LLMActorService
from llm_actor.client.adapters.openai import OpenAIAdapter
from llm_actor.client.adapters.openai_compatible import OpenAICompatibleAdapter
from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import LLMServiceGeneralError, LLMServiceOverloadedError


@pytest.mark.asyncio
async def test_openai_adapter_maps_rate_limit_to_overloaded() -> None:
    adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")

    async def raise_rl(**_kwargs: object) -> None:
        raise RateLimitError("rate limited", response=MagicMock(), body=None)

    with patch.object(
        adapter._client.chat.completions, "create", new=AsyncMock(side_effect=raise_rl)
    ):
        with pytest.raises(LLMServiceOverloadedError):
            await adapter.generate_async(LLMRequest(prompt="x"))
    await adapter.close()


@pytest.mark.asyncio
async def test_openai_adapter_passes_temperature_and_extra_to_sdk() -> None:
    adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content="ok"))]
        return completion

    with patch.object(
        adapter._client.chat.completions, "create", new=AsyncMock(side_effect=fake_create)
    ):
        out = await adapter.generate_async(
            LLMRequest(prompt="Hello", temperature=0.8, extra={"top_p": 0.9})
        )
    assert out == "ok"
    assert captured["temperature"] == 0.8
    assert captured["top_p"] == 0.9
    assert captured["model"] == "gpt-4o-mini"
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[-1]["content"] == "Hello"
    await adapter.close()


@pytest.mark.asyncio
async def test_openai_adapter_extra_does_not_overwrite_mandatory_fields() -> None:
    """request.extra must not overwrite model or messages."""
    adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content="ok"))]
        return completion

    with patch.object(
        adapter._client.chat.completions, "create", new=AsyncMock(side_effect=fake_create)
    ):
        await adapter.generate_async(
            LLMRequest(prompt="Hello", extra={"model": "evil-model", "messages": []})
        )
    assert captured["model"] == "gpt-4o-mini"
    messages = captured["messages"]
    assert isinstance(messages, list) and len(messages) > 0
    await adapter.close()


@pytest.mark.asyncio
async def test_openai_adapter_empty_choices_raises_domain_error() -> None:
    adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")

    async def fake_create(**_kwargs: object) -> MagicMock:
        completion = MagicMock()
        completion.choices = []
        return completion

    with patch.object(
        adapter._client.chat.completions, "create", new=AsyncMock(side_effect=fake_create)
    ):
        with pytest.raises(LLMServiceGeneralError):
            await adapter.generate_async(LLMRequest(prompt="x"))
    await adapter.close()


@pytest.mark.asyncio
async def test_openai_adapter_stop_sequences_none_not_sent() -> None:
    """stop_sequences=None is omitted from the SDK payload."""
    adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content="ok"))]
        return completion

    with patch.object(
        adapter._client.chat.completions, "create", new=AsyncMock(side_effect=fake_create)
    ):
        await adapter.generate_async(LLMRequest(prompt="x", stop_sequences=None))
    assert "stop" not in captured
    await adapter.close()


@pytest.mark.asyncio
async def test_openai_adapter_stop_sequences_empty_list_sent() -> None:
    """stop_sequences=[] is passed explicitly to the SDK."""
    adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
    captured: dict[str, object] = {}

    async def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content="ok"))]
        return completion

    with patch.object(
        adapter._client.chat.completions, "create", new=AsyncMock(side_effect=fake_create)
    ):
        await adapter.generate_async(LLMRequest(prompt="x", stop_sequences=[]))
    assert captured["stop"] == []
    await adapter.close()


@pytest.mark.asyncio
async def test_from_openai_factory_builds_service() -> None:
    mock_base = MagicMock()
    mock_base.close = AsyncMock()
    with patch(
        "llm_actor.client.adapters.openai.OpenAIAdapter", return_value=mock_base
    ) as ctor_mock:
        svc = LLMActorService.from_openai(api_key="k", model="m", extra_opt=1)
    ctor_mock.assert_called_once_with(api_key="k", model="m", extra_opt=1)
    assert svc._base_client is mock_base


@pytest.mark.asyncio
async def test_openai_compatible_adapter_passes_base_url() -> None:
    """OpenAICompatibleAdapter forwards base_url to AsyncOpenAI."""
    with patch("llm_actor.client.adapters.openai.AsyncOpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()
        OpenAICompatibleAdapter(api_key="k", model="m", base_url="http://localhost:11434/v1")
    mock_openai_cls.assert_called_once_with(api_key="k", base_url="http://localhost:11434/v1")


@pytest.mark.asyncio
async def test_from_openai_compatible_factory_builds_service() -> None:
    mock_base = MagicMock()
    mock_base.close = AsyncMock()
    with patch(
        "llm_actor.client.adapters.openai_compatible.OpenAICompatibleAdapter",
        return_value=mock_base,
    ) as ctor_mock:
        svc = LLMActorService.from_openai_compatible(
            api_key="k", model="m", base_url="http://localhost:11434/v1"
        )
    ctor_mock.assert_called_once_with(api_key="k", model="m", base_url="http://localhost:11434/v1")
    assert svc._base_client is mock_base
