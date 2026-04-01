import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from llm_actor import LLMBrokerService, LLMBrokerSettings
from llm_actor.actors.pool import _PrioritizedMessage
from llm_actor.actors.worker import ModelActor
from llm_actor.core.messages import ActorMessage
from tests.dummy_llm_client import DummyLLMClient


def test_priority_field_default() -> None:
    msg = ActorMessage(prompt="x")
    assert msg.priority == 10


def test_prioritized_message_ordering() -> None:
    items = [
        _PrioritizedMessage(5, 0, ActorMessage(prompt="a")),
        _PrioritizedMessage(0, 1, ActorMessage(prompt="b")),
        _PrioritizedMessage(10, 2, ActorMessage(prompt="c")),
    ]
    ordered = sorted(items)
    assert [p.priority for p in ordered] == [0, 5, 10]


def test_prioritized_message_fifo_within_same_priority() -> None:
    items = [
        _PrioritizedMessage(1, 2, ActorMessage(prompt="a")),
        _PrioritizedMessage(1, 0, ActorMessage(prompt="b")),
        _PrioritizedMessage(1, 1, ActorMessage(prompt="c")),
    ]
    ordered = sorted(items)
    assert [p.sequence for p in ordered] == [0, 1, 2]


@pytest.mark.asyncio
async def test_generate_accepts_priority_parameter(service) -> None:
    r0 = await service.generate("test", priority=0)
    r10 = await service.generate("test", priority=10)
    assert isinstance(r0, str)
    assert isinstance(r10, str)


@pytest.mark.asyncio
async def test_high_priority_processed_before_low() -> None:
    start_event = asyncio.Event()
    call_order: list[str] = []

    async def blocking_mock(self, prompt: str) -> str:
        call_order.append(prompt)
        if prompt == "block":
            await start_event.wait()
        return f"Response for: {prompt}"

    settings = LLMBrokerSettings()
    settings.LLM_NUM_ACTORS = 1
    settings.LLM_BATCH_SIZE = 1
    settings.LLM_BATCH_TIMEOUT = 0.01

    with patch.object(DummyLLMClient, "generate_async", blocking_mock):
        base_client = DummyLLMClient(settings=settings)
        service = LLMBrokerService(base_client=base_client, settings=settings)
        await service.start()
        try:
            tasks: list[asyncio.Task[str]] = [
                asyncio.create_task(service.generate("block", priority=50))
            ]
            await asyncio.sleep(0)
            tasks.extend(
                asyncio.create_task(service.generate("low", priority=100)) for _ in range(3)
            )
            tasks.append(asyncio.create_task(service.generate("high", priority=0)))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            start_event.set()
            await asyncio.gather(*tasks)
        finally:
            await service.stop()

    assert call_order[0] == "block"
    assert call_order[1] == "high"
    assert call_order[2:] == ["low", "low", "low"]


@pytest.mark.asyncio
async def test_ac5b_item_not_lost_when_get_and_stop_simultaneous() -> None:
    """AC 5b: если get_fut и stop_fut оба в done — элемент не теряется."""
    settings = LLMBrokerSettings()
    queue: asyncio.PriorityQueue[Any] = asyncio.PriorityQueue()
    msg = ActorMessage(prompt="simultaneous")
    item = _PrioritizedMessage(priority=10, sequence=0, message=msg)
    await queue.put(item)

    actor = ModelActor(
        client=None,  # type: ignore[arg-type]
        actor_id="test-ac5b",
        settings=settings,
        shared_queue=queue,
    )

    get_fut: asyncio.Future[Any] = asyncio.ensure_future(queue.get())
    await asyncio.sleep(0)

    done: set[asyncio.Future[Any]] = {get_fut}
    succeeded = actor._append_item_if_get_succeeded(done, get_fut)

    assert succeeded is True
    assert len(actor._pending) == 1
    assert actor._pending[0] is msg
