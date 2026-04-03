import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_actor.client.tool_loop import ToolCallOrchestratorClient
from llm_actor.core.request import LLMRequest
from llm_actor.core.tools import LLMResponse, Tool, ToolCall
from llm_actor.exceptions import (
    LLMServiceGeneralError,
    ToolExecutionTimeoutError,
    ToolLoopMaxIterationsError,
)
from llm_actor.service import LLMActorService
from llm_actor.settings import LLMActorSettings


class _FakeToolCapableClient:
    def __init__(self) -> None:
        self.generate_with_tools_async = AsyncMock()
        self.format_tool_results = MagicMock(return_value=[])
        self.generate = AsyncMock(return_value="plain")


@pytest.mark.asyncio
async def test_tool_loop_single_iteration() -> None:
    cb = _FakeToolCapableClient()
    cb.generate_with_tools_async.return_value = LLMResponse(content="Result", tool_calls=[])
    orchestrator = ToolCallOrchestratorClient(cb, LLMActorSettings())  # type: ignore[arg-type]
    req = LLMRequest(prompt="hi", tools=[Tool(func=lambda: 1, name="x")])
    out = await orchestrator.generate(req)
    assert out == "Result"
    assert cb.generate_with_tools_async.await_count == 1


@pytest.mark.asyncio
async def test_tool_loop_with_tool_execution() -> None:
    async def get_weather(city: str) -> str:
        return "Sunny"

    cb = _FakeToolCapableClient()
    cb.generate_with_tools_async.side_effect = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="get_weather", arguments={"city": "London"})],
            assistant_message={"role": "assistant", "content": None, "tool_calls": []},
        ),
        LLMResponse(content="Weather: Sunny", tool_calls=[]),
    ]
    cb.format_tool_results.return_value = [{"role": "tool", "content": "Sunny"}]

    orchestrator = ToolCallOrchestratorClient(cb, LLMActorSettings())  # type: ignore[arg-type]
    req = LLMRequest(prompt="w", tools=[Tool(func=get_weather)])
    out = await orchestrator.generate(req)
    assert out == "Weather: Sunny"
    assert cb.format_tool_results.call_count == 1


@pytest.mark.asyncio
async def test_tool_loop_sync_tool() -> None:
    def sync_add(a: int, b: int) -> int:
        return a + b

    cb = _FakeToolCapableClient()
    cb.generate_with_tools_async.side_effect = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="sync_add", arguments={"a": 2, "b": 3})],
            assistant_message={"role": "assistant", "content": None, "tool_calls": []},
        ),
        LLMResponse(content="5", tool_calls=[]),
    ]
    cb.format_tool_results.return_value = []

    with patch(
        "llm_actor.client.tool_loop.asyncio.to_thread", new_callable=AsyncMock
    ) as mock_to_thread:
        mock_to_thread.return_value = 5
        orchestrator = ToolCallOrchestratorClient(cb, LLMActorSettings())  # type: ignore[arg-type]
        req = LLMRequest(prompt="q", tools=[Tool(func=sync_add)])
        out = await orchestrator.generate(req)
    assert out == "5"
    mock_to_thread.assert_called_once_with(sync_add, a=2, b=3)


@pytest.mark.asyncio
async def test_tool_loop_tool_timeout() -> None:
    async def slow_tool() -> str:
        await asyncio.sleep(100)
        return "n"

    cb = _FakeToolCapableClient()
    cb.generate_with_tools_async.return_value = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="1", name="slow_tool", arguments={})],
        assistant_message={"role": "assistant", "content": None, "tool_calls": []},
    )

    orchestrator = ToolCallOrchestratorClient(cb, LLMActorSettings())  # type: ignore[arg-type]
    req = LLMRequest(prompt="q", tools=[Tool(func=slow_tool)], tool_timeout=0.01)
    with pytest.raises(ToolExecutionTimeoutError):
        await orchestrator.generate(req)


@pytest.mark.asyncio
async def test_tool_loop_max_iterations() -> None:
    cb = _FakeToolCapableClient()
    cb.generate_with_tools_async.return_value = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="1", name="t", arguments={})],
        assistant_message={"role": "assistant", "content": None, "tool_calls": []},
    )
    cb.format_tool_results.return_value = []

    settings = LLMActorSettings(LLM_TOOL_MAX_ITERATIONS=3)
    orchestrator = ToolCallOrchestratorClient(cb, settings)  # type: ignore[arg-type]

    async def dummy() -> str:
        return "x"

    req = LLMRequest(prompt="q", tools=[Tool(func=dummy, name="t")])
    with pytest.raises(ToolLoopMaxIterationsError) as exc_info:
        await orchestrator.generate(req)
    assert exc_info.value.max_iterations == 3
    assert cb.generate_with_tools_async.await_count == 3


@pytest.mark.asyncio
async def test_tool_loop_unknown_tool() -> None:
    async def known_fn() -> str:
        return "ok"

    cb = _FakeToolCapableClient()
    cb.generate_with_tools_async.side_effect = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="unknown_tool", arguments={})],
            assistant_message={"role": "assistant", "content": None, "tool_calls": []},
        ),
        LLMResponse(content="done", tool_calls=[]),
    ]
    cb.format_tool_results.return_value = []

    orchestrator = ToolCallOrchestratorClient(cb, LLMActorSettings())  # type: ignore[arg-type]
    req = LLMRequest(prompt="q", tools=[Tool(func=known_fn)])
    out = await orchestrator.generate(req)
    assert out == "done"
    fmt_arg = cb.format_tool_results.call_args[0][0]
    assert len(fmt_arg) == 1
    assert fmt_arg[0].is_error is True


def test_llmrequest_normalizes_callables() -> None:
    def named_fn() -> None:
        pass

    req = LLMRequest(prompt="x", tools=[named_fn])  # type: ignore[list-item]
    assert req.tools is not None
    assert isinstance(req.tools[0], Tool)
    assert req.tools[0].name == "named_fn"


def test_tool_lambda_requires_explicit_name() -> None:
    with pytest.raises(ValueError, match="explicit name"):
        Tool(func=lambda: None)


@pytest.mark.asyncio
async def test_service_generate_with_tool_end_to_end() -> None:
    """AC1: LLMActorService.generate() with tools executes the full client stack."""
    tool_called = False

    async def my_tool() -> str:
        nonlocal tool_called
        tool_called = True
        return "tool-output"

    class _ToolCapableBase:
        def __init__(self) -> None:
            self.generate_with_tools_async = AsyncMock(
                side_effect=[
                    LLMResponse(
                        content=None,
                        tool_calls=[ToolCall(id="1", name="my_tool", arguments={})],
                        assistant_message={"role": "assistant", "content": "", "tool_calls": []},
                    ),
                    LLMResponse(content="done", tool_calls=[]),
                ]
            )
            self.format_tool_results = MagicMock(
                return_value=[{"role": "tool", "content": "tool-output"}]
            )

        async def generate_async(self, request: LLMRequest) -> str:
            return "plain"

    base_client = _ToolCapableBase()
    service = LLMActorService(base_client=base_client, settings=LLMActorSettings())  # type: ignore[arg-type]
    await service.start()
    try:
        req = LLMRequest(prompt="q", tools=[Tool(func=my_tool)])
        result = await service.generate(req)
        assert result == "done"
        assert tool_called is True
    finally:
        await service.stop()


def test_tool_schema_inference() -> None:
    def search(query: str, limit: int = 10) -> str:
        return query

    tool = Tool(func=search)
    schema = tool._infer_schema()
    assert schema["properties"]["query"]["type"] == "string"
    assert "query" in schema.get("required", [])
    assert "limit" not in schema.get("required", [])


@pytest.mark.asyncio
async def test_generate_with_tools_raises_if_response_model_set() -> None:
    cb = _FakeToolCapableClient()

    class _M:
        pass

    orchestrator = ToolCallOrchestratorClient(cb, LLMActorSettings())  # type: ignore[arg-type]
    req = LLMRequest(prompt="q", tools=[Tool(func=lambda: 1, name="t")])
    with pytest.raises(ValueError, match="Combining tools with response_model is not supported"):
        await orchestrator.generate(req, response_model=_M)


@pytest.mark.asyncio
async def test_generate_with_tools_raises_if_client_not_tool_capable() -> None:
    class _Bare:
        async def generate(
            self, request: LLMRequest, response_model: type[Any] | None = None
        ) -> str:
            return "x"

    orchestrator = ToolCallOrchestratorClient(_Bare(), LLMActorSettings())  # type: ignore[arg-type]
    req = LLMRequest(prompt="q", tools=[Tool(func=lambda: 1, name="t")])
    with pytest.raises(
        LLMServiceGeneralError, match="does not implement ToolCapableClientInterface"
    ):
        await orchestrator.generate(req)


@pytest.mark.asyncio
async def test_tool_loop_async_tool_not_to_thread() -> None:
    called = False

    async def async_tool() -> str:
        nonlocal called
        called = True
        return "async-result"

    cb = _FakeToolCapableClient()
    cb.generate_with_tools_async.side_effect = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="1", name="async_tool", arguments={})],
            assistant_message={"role": "assistant", "content": None, "tool_calls": []},
        ),
        LLMResponse(content="ok", tool_calls=[]),
    ]
    cb.format_tool_results.return_value = []

    with patch("llm_actor.client.tool_loop.asyncio.to_thread") as mock_to_thread:
        orchestrator = ToolCallOrchestratorClient(cb, LLMActorSettings())  # type: ignore[arg-type]
        req = LLMRequest(prompt="q", tools=[Tool(func=async_tool)])
        await orchestrator.generate(req)
    assert called is True
    mock_to_thread.assert_not_called()
