from __future__ import annotations

import asyncio
from collections.abc import Generator, Sequence

import opentelemetry.trace as _trace_module
import pytest
from opentelemetry.propagate import get_global_textmap, set_global_textmap
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from llm_actor import LLMActorService
from llm_actor.settings import LLMActorSettings
from tests.dummy_llm_client import DummyLLMClient
from tests.models import User


class _CapturingSpanExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _force_set_tracer_provider(provider: TracerProvider) -> None:
    """Force-set TracerProvider, bypassing the API's set-once guard."""
    _trace_module._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    _trace_module._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    _trace_module._set_tracer_provider(provider, log=False)  # type: ignore[attr-defined]


@pytest.fixture
def otel_memory_exporter() -> Generator[_CapturingSpanExporter]:
    """Per-test fresh TracerProvider and W3C trace context propagator."""
    exporter = _CapturingSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    original_provider = _trace_module._TRACER_PROVIDER  # type: ignore[attr-defined]
    original_done = _trace_module._TRACER_PROVIDER_SET_ONCE._done  # type: ignore[attr-defined]
    original_textmap = get_global_textmap()

    _force_set_tracer_provider(provider)
    set_global_textmap(TraceContextTextMapPropagator())
    try:
        yield exporter
    finally:
        _trace_module._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
        _trace_module._TRACER_PROVIDER = None  # type: ignore[attr-defined]
        if original_provider is not None:
            _trace_module._set_tracer_provider(original_provider, log=False)  # type: ignore[attr-defined]
        else:
            _trace_module._TRACER_PROVIDER_SET_ONCE._done = original_done  # type: ignore[attr-defined]
        set_global_textmap(original_textmap)


def _span_by_name(exporter: _CapturingSpanExporter, name: str) -> ReadableSpan | None:
    for span in exporter.spans:
        if span.name == name:
            return span
    return None


@pytest.mark.asyncio
async def test_generate_creates_root_pool_actor_and_llm_spans(
    mock_llm_response: dict[str, str],
    otel_memory_exporter: _CapturingSpanExporter,
) -> None:
    settings = LLMActorSettings()
    base_client = DummyLLMClient(settings=settings)
    service = LLMActorService(base_client=base_client, settings=settings)
    await service.start()
    try:
        prompt = "trace-me"
        mock_llm_response[prompt] = "plain text"
        await service.generate(prompt)
    finally:
        await service.stop()

    spans = {s.name: s for s in otel_memory_exporter.spans}
    assert "llm_actor.generate" in spans
    assert "llm_pool.wait" in spans
    assert "llm_actor.actor_process" in spans
    assert "llm_actor.llm_request" in spans

    generate_root = spans["llm_actor.generate"]
    wait = spans["llm_pool.wait"]
    actor = spans["llm_actor.actor_process"]
    llm_req = spans["llm_actor.llm_request"]

    assert wait.parent is not None
    assert wait.parent.span_id == generate_root.context.span_id
    assert actor.parent is not None
    assert actor.parent.span_id == generate_root.context.span_id
    assert llm_req.parent is not None
    assert llm_req.parent.span_id == actor.context.span_id


@pytest.mark.asyncio
async def test_structured_response_emits_validate_span(
    mock_llm_response: dict[str, str],
    otel_memory_exporter: _CapturingSpanExporter,
) -> None:
    settings = LLMActorSettings()
    base_client = DummyLLMClient(settings=settings)
    service = LLMActorService(base_client=base_client, settings=settings)
    await service.start()
    try:
        prompt = "structured"
        mock_llm_response[prompt] = '{"name": "A", "age": 1}'
        await service.generate(prompt, response_model=User)
    finally:
        await service.stop()

    names = {s.name for s in otel_memory_exporter.spans}
    assert "llm_actor.validate" in names

    validate_span = _span_by_name(otel_memory_exporter, "llm_actor.validate")
    assert validate_span is not None
    actor_span = _span_by_name(otel_memory_exporter, "llm_actor.actor_process")
    assert actor_span is not None
    assert validate_span.parent is not None
    assert validate_span.parent.span_id == actor_span.context.span_id


@pytest.mark.asyncio
async def test_queue_wait_span_finished_with_valid_times(
    mock_llm_response: dict[str, str],
    otel_memory_exporter: _CapturingSpanExporter,
) -> None:
    settings = LLMActorSettings()
    base_client = DummyLLMClient(settings=settings)
    service = LLMActorService(base_client=base_client, settings=settings)
    await service.start()
    try:
        mock_llm_response["qwait"] = "ok"
        await service.generate("qwait")

        wait = _span_by_name(otel_memory_exporter, "llm_pool.wait")
        generate_root = _span_by_name(otel_memory_exporter, "llm_actor.generate")
        assert wait is not None
        assert generate_root is not None
        assert wait.parent is not None
        assert wait.parent.span_id == generate_root.context.span_id
        assert wait.end_time is not None and wait.start_time is not None
        assert wait.end_time >= wait.start_time
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_queue_wait_span_captures_actual_queue_time(
    mock_llm_response: dict[str, str],
    otel_memory_exporter: _CapturingSpanExporter,
) -> None:
    """Queue delay via two concurrent requests to a single-actor pool (batch_size=1).

    The second request waits while the first runs; both llm_pool.wait spans must end
    with valid durations.
    """
    settings = LLMActorSettings(LLM_NUM_ACTORS=1, LLM_BATCH_SIZE=1, LLM_BATCH_TIMEOUT=0.05)
    base_client = DummyLLMClient(settings=settings)
    service = LLMActorService(base_client=base_client, settings=settings)
    await service.start()
    try:
        mock_llm_response["qdelay1"] = "r1"
        mock_llm_response["qdelay2"] = "r2"
        await asyncio.gather(service.generate("qdelay1"), service.generate("qdelay2"))
    finally:
        await service.stop()

    wait_spans = [s for s in otel_memory_exporter.spans if s.name == "llm_pool.wait"]
    assert len(wait_spans) >= 1
    for ws in wait_spans:
        assert ws.end_time is not None and ws.start_time is not None
        assert ws.end_time >= ws.start_time


@pytest.mark.asyncio
async def test_library_works_without_configured_tracer_provider(
    mock_llm_response: dict[str, str],
) -> None:
    """AC5: library runs without error when no SDK / no configured provider."""
    # Reset to no-op proxy — equivalent to an environment without opentelemetry-sdk.
    original_provider = _trace_module._TRACER_PROVIDER  # type: ignore[attr-defined]
    original_done = _trace_module._TRACER_PROVIDER_SET_ONCE._done  # type: ignore[attr-defined]
    _trace_module._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    _trace_module._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    try:
        settings = LLMActorSettings()
        base_client = DummyLLMClient(settings=settings)
        service = LLMActorService(base_client=base_client, settings=settings)
        await service.start()
        try:
            mock_llm_response["no-sdk"] = "ok"
            result = await service.generate("no-sdk")
            assert result == "ok"
        finally:
            await service.stop()
    finally:
        _trace_module._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
        _trace_module._TRACER_PROVIDER = None  # type: ignore[attr-defined]
        if original_provider is not None:
            _trace_module._set_tracer_provider(original_provider, log=False)  # type: ignore[attr-defined]
        else:
            _trace_module._TRACER_PROVIDER_SET_ONCE._done = original_done  # type: ignore[attr-defined]
