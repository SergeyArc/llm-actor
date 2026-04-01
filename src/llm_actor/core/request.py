from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRequest:
    prompt: str
    temperature: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    system_prompt: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
