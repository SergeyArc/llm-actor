from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager

from opentelemetry import propagate, trace
from opentelemetry.context import Context, attach, detach

_TRACER_NAME = "llm_actor"

PROMPT_PREVIEW_MAX_LEN = 256


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


def inject_context(carrier: MutableMapping[str, str] | None = None) -> dict[str, str]:
    out: dict[str, str] = dict(carrier) if carrier is not None else {}
    propagate.inject(out)
    return out


def extract_context(carrier: Mapping[str, str]) -> Context:
    return propagate.extract(carrier)


@contextmanager
def attach_extracted_context(carrier: Mapping[str, str] | None) -> Iterator[None]:
    if carrier is None:
        yield
        return
    token = attach(extract_context(carrier))
    try:
        yield
    finally:
        detach(token)


def truncate_for_span_attribute(text: str | None) -> str:
    if not text:
        return ""
    if len(text) <= PROMPT_PREVIEW_MAX_LEN:
        return text
    return text[:PROMPT_PREVIEW_MAX_LEN]
