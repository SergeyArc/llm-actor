from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openai")
pytest.importorskip("anthropic")

from llm_actor.client.adapters.anthropic import AnthropicAdapter
from llm_actor.client.adapters.openai import OpenAIAdapter
from llm_actor.core.request import LLMRequest
from llm_actor.core.tools import ToolResult


def test_openai_adapter_format_tool_results() -> None:
    adapter = OpenAIAdapter(api_key="sk-test", model="gpt-4o-mini")
    results = [
        ToolResult(tool_call_id="1", name="fn", result="ok"),
        ToolResult(tool_call_id="2", name="fn2", result="err", is_error=True),
    ]
    msgs = adapter.format_tool_results(results)
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["content"] == "ok"
    assert msgs[1]["content"] == "Error: err"


def test_anthropic_adapter_format_tool_results() -> None:
    adapter = AnthropicAdapter(api_key="k", model="claude-3")
    results = [
        ToolResult(tool_call_id="t1", name="fn", result="ok"),
        ToolResult(tool_call_id="t2", name="fn2", result="bad", is_error=True),
    ]
    msgs = adapter.format_tool_results(results)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert isinstance(content, list) and len(content) == 2
    assert content[0]["type"] == "tool_result"
    assert content[0]["tool_use_id"] == "t1"
    assert content[0]["content"] == "ok"
    assert content[1]["is_error"] is True


@pytest.mark.asyncio
async def test_openai_adapter_messages_history_before_prompt() -> None:
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
            LLMRequest(
                prompt="follow up",
                messages=[
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                ],
            )
        )
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["content"] == "first"
    assert messages[1]["content"] == "reply"
    assert messages[-1]["content"] == "follow up"
    await adapter.close()


@pytest.mark.asyncio
async def test_openai_adapter_empty_prompt_skipped_with_messages_only() -> None:
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
            LLMRequest(
                prompt="",
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            )
        )
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["content"] == "hi"
    assert messages[1]["content"] == "hello"
    await adapter.close()
