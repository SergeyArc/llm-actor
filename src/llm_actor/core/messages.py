import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from llm_actor.core.request import LLMRequest


@dataclass
class ActorMessage[T]:
    request: LLMRequest
    response_model: type[T] | None = None
    id: str | None = None
    future: asyncio.Future[T] | None = None
    priority: int = 10
    enqueue_sequence: int | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid4())
