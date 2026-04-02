from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llm_actor.core.tools import Tool


@dataclass
class LLMRequest:
    prompt: str
    temperature: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    system_prompt: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] | None = None
    tools: list[Tool | Callable[..., Any]] | None = None
    tool_timeout: float | None = None

    def __post_init__(self) -> None:
        if self.tools is not None:
            self.tools = [t if isinstance(t, Tool) else Tool(func=t) for t in self.tools]
