import asyncio
import contextvars
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openai")
pytest.importorskip("anthropic")

from llm_actor import LLMActorService
from llm_actor.actors.pool import SupervisedActorPool
from llm_actor.client.adapters.anthropic import AnthropicAdapter
from llm_actor.client.adapters.gigachat import GigaChatAdapter
from llm_actor.client.adapters.openai import OpenAIAdapter
from llm_actor.core.messages import ActorMessage
from llm_actor.core.request import LLMRequest
from llm_actor.settings import LLMActorSettings

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


class ContextCapturingClient:
    """Spy-клиент: записывает (prompt, context_value) при каждом вызове."""

    def __init__(self) -> None:
        self.captured: list[tuple[str, str | None]] = []

    async def generate_async(self, request: LLMRequest) -> str:
        self.captured.append((request.prompt, request_id_var.get()))
        return "ok"

    async def close(self) -> None:
        pass


async def test_context_propagated_to_worker() -> None:
    client = ContextCapturingClient()
    settings = LLMActorSettings(LLM_NUM_ACTORS=1)
    async with LLMActorService(base_client=client, settings=settings) as svc:
        request_id_var.set("trace-123")
        await svc.generate("hello")
    assert client.captured == [("hello", "trace-123")]


async def test_context_isolation_between_concurrent_calls() -> None:
    client = ContextCapturingClient()
    settings = LLMActorSettings(LLM_NUM_ACTORS=3)

    async def call_with_id(svc: LLMActorService, trace_id: str) -> None:
        request_id_var.set(trace_id)
        await svc.generate(f"prompt-{trace_id}")

    async with LLMActorService(base_client=client, settings=settings) as svc:
        await asyncio.gather(*[call_with_id(svc, f"id-{i}") for i in range(5)])

    assert len(client.captured) == 5
    for prompt, ctx_value in client.captured:
        expected_id = prompt.removeprefix("prompt-")
        assert ctx_value == expected_id, (
            f"prompt '{prompt}' ожидал контекст '{expected_id}', получил '{ctx_value}'"
        )


async def test_extra_headers_passed_to_openai_sdk() -> None:
    adapter = OpenAIAdapter(api_key="test-key", model="gpt-4o")
    mock_create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="response", tool_calls=None))]
        )
    )
    with patch.object(adapter._client.chat.completions, "create", mock_create):
        request = LLMRequest(prompt="hi", extra_headers={"X-Trace-Id": "abc"})
        await adapter.generate_async(request)
    _, kwargs = mock_create.call_args
    assert kwargs.get("extra_headers") == {"X-Trace-Id": "abc"}


async def test_extra_headers_passed_to_anthropic_sdk() -> None:
    adapter = AnthropicAdapter(api_key="test-key", model="claude-3-5-sonnet-20241022")
    mock_block = MagicMock()
    mock_block.text = "response"
    mock_create = AsyncMock(return_value=MagicMock(content=[mock_block], stop_reason="end_turn"))
    with patch.object(adapter._client.messages, "create", mock_create):
        request = LLMRequest(prompt="hi", extra_headers={"X-Trace-Id": "abc"})
        await adapter.generate_async(request)
    _, kwargs = mock_create.call_args
    assert kwargs.get("extra_headers") == {"X-Trace-Id": "abc"}


async def test_gigachat_extra_headers_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    try:
        from gigachat.models import Chat  # noqa: F401
    except ImportError:
        pytest.skip("gigachat SDK not installed")

    adapter = GigaChatAdapter(credentials="fake")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    with patch.object(adapter._client, "achat", AsyncMock(return_value=mock_response)):
        request = LLMRequest(prompt="hi", extra_headers={"X-Custom": "val"})
        with caplog.at_level(logging.WARNING):
            result = await adapter.generate_async(request)
    assert result == "ok"
    assert any("extra_headers" in record.message for record in caplog.records)


async def test_extra_headers_none_does_not_break_openai_adapter() -> None:
    """AC6: OpenAI адаптер работает штатно когда extra_headers не задан (None)."""
    adapter = OpenAIAdapter(api_key="test-key", model="gpt-4o")
    mock_create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="response", tool_calls=None))]
        )
    )
    with patch.object(adapter._client.chat.completions, "create", mock_create):
        result = await adapter.generate_async(LLMRequest(prompt="hello"))
    assert result == "response"
    _, kwargs = mock_create.call_args
    assert kwargs.get("extra_headers") is None


async def test_extra_headers_none_does_not_break_anthropic_adapter() -> None:
    """AC6: Anthropic адаптер работает штатно когда extra_headers не задан (None)."""
    adapter = AnthropicAdapter(api_key="test-key", model="claude-3-5-sonnet-20241022")
    mock_block = MagicMock()
    mock_block.text = "response"
    mock_create = AsyncMock(return_value=MagicMock(content=[mock_block], stop_reason="end_turn"))
    with patch.object(adapter._client.messages, "create", mock_create):
        result = await adapter.generate_async(LLMRequest(prompt="hello"))
    assert result == "response"
    _, kwargs = mock_create.call_args
    assert kwargs.get("extra_headers") is None


async def test_context_no_leakage_between_sequential_calls() -> None:
    client = ContextCapturingClient()
    settings = LLMActorSettings(LLM_NUM_ACTORS=1)
    async with LLMActorService(base_client=client, settings=settings) as svc:
        request_id_var.set("first")
        await svc.generate("prompt-first")
        request_id_var.set("second")
        await svc.generate("prompt-second")

    assert client.captured == [("prompt-first", "first"), ("prompt-second", "second")]


async def test_extra_headers_passed_to_openai_tools_sdk() -> None:
    from llm_actor.core.tools import Tool

    adapter = OpenAIAdapter(api_key="test-key", model="gpt-4o")
    mock_create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="done", tool_calls=None))]
        )
    )
    with patch.object(adapter._client.chat.completions, "create", mock_create):
        request = LLMRequest(
            prompt="hi",
            extra_headers={"X-Trace-Id": "abc"},
            tools=[Tool(func=lambda: "result", name="noop", description="noop")],
        )
        await adapter.generate_with_tools_async(request, conversation=[])
    _, kwargs = mock_create.call_args
    assert kwargs.get("extra_headers") == {"X-Trace-Id": "abc"}


async def test_extra_headers_passed_to_anthropic_tools_sdk() -> None:
    from llm_actor.core.tools import Tool

    adapter = AnthropicAdapter(api_key="test-key", model="claude-3-5-sonnet-20241022")
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = "done"
    mock_create = AsyncMock(return_value=MagicMock(content=[mock_block], stop_reason="end_turn"))
    with patch.object(adapter._client.messages, "create", mock_create):
        request = LLMRequest(
            prompt="hi",
            extra_headers={"X-Trace-Id": "abc"},
            tools=[Tool(func=lambda: "result", name="noop", description="noop")],
        )
        await adapter.generate_with_tools_async(request, conversation=[])
    _, kwargs = mock_create.call_args
    assert kwargs.get("extra_headers") == {"X-Trace-Id": "abc"}


async def test_extra_headers_passed_to_openai_compatible_sdk() -> None:
    """AC7: OpenAICompatibleAdapter наследует поддержку extra_headers от OpenAIAdapter."""
    from llm_actor.client.adapters.openai_compatible import OpenAICompatibleAdapter

    adapter = OpenAICompatibleAdapter(
        api_key="test-key", model="local-model", base_url="http://localhost:8000/v1"
    )
    mock_create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="response", tool_calls=None))]
        )
    )
    with patch.object(adapter._client.chat.completions, "create", mock_create):
        request = LLMRequest(prompt="hi", extra_headers={"X-Trace-Id": "abc"})
        await adapter.generate_async(request)
    _, kwargs = mock_create.call_args
    assert kwargs.get("extra_headers") == {"X-Trace-Id": "abc"}


async def test_gigachat_extra_headers_logs_warning_in_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T8: GigaChat логирует warning для extra_headers в generate_with_tools_async."""
    try:
        from gigachat.models import Chat  # noqa: F401
    except ImportError:
        pytest.skip("gigachat SDK not installed")

    from llm_actor.core.tools import Tool

    adapter = GigaChatAdapter(credentials="fake")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    mock_response.choices[0].message.function_call = None
    with patch.object(adapter._client, "achat", AsyncMock(return_value=mock_response)):
        request = LLMRequest(
            prompt="hi",
            extra_headers={"X-Custom": "val"},
            tools=[Tool(func=lambda: "result", name="noop", description="noop")],
        )
        with caplog.at_level(logging.WARNING):
            await adapter.generate_with_tools_async(request, conversation=[])
    assert any("extra_headers" in record.message for record in caplog.records)


async def test_caller_context_preserved_on_requeue() -> None:
    """TD6: _requeue_pending_messages сохраняет caller_context (в отличие от otel_context)."""
    ctx = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    msg = ActorMessage(
        request=LLMRequest(prompt="requeue-test"),
        future=future,
        otel_context={"traceparent": "00-abc-01"},
        caller_context=ctx,
    )

    mock_client = MagicMock()
    pool = SupervisedActorPool(client=mock_client, settings=LLMActorSettings(LLM_NUM_ACTORS=1))

    with patch.object(pool, "_put_in_queue", AsyncMock()):
        await pool._requeue_pending_messages([msg])

    assert msg.otel_context is None, "otel_context должен быть сброшен при requeue"
    assert msg.caller_context is ctx, "caller_context должен быть сохранён при requeue"

    future.cancel()
