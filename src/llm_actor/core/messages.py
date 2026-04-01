import asyncio
from dataclasses import dataclass
from uuid import uuid4


@dataclass
class ActorMessage[T]:
    prompt: str
    response_model: type[T] | None = None
    id: str | None = None
    future: asyncio.Future[T] | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid4())
