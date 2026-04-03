import json

import pytest
import pytest_asyncio

from llm_actor import LLMActorService, LLMActorSettings
from tests.dummy_llm_client import DummyLLMClient
from tests.models import User


@pytest_asyncio.fixture
async def service_with_fast_dummy():
    settings = LLMActorSettings()
    settings.LLM_NUM_ACTORS = 1
    settings.LLM_BATCH_SIZE = 1
    base_client = DummyLLMClient(settings=settings)
    base_client.base_latency = 0.0
    base_client.latency_variance = 0.0
    service = LLMActorService(base_client=base_client, settings=settings)
    await service.start()
    yield service, base_client
    await service.stop()


@pytest.mark.asyncio
async def test_semantic_retry_succeeds_after_json_decode_and_validation_errors(
    service_with_fast_dummy,
):
    service, dummy = service_with_fast_dummy
    dummy.set_prompt_response_sequence(
        "retry_user",
        [
            '{"name": "Bob", "age": 30',
            '{"name": "Bob", "age": "thirty"}',
            '{"name": "Bob", "age": 30}',
        ],
    )

    result = await service.generate("retry_user", response_model=User)

    assert result == User(name="Bob", age=30)
    assert dummy.prompt_response_call_indices["retry_user"] == 3


@pytest.mark.asyncio
async def test_semantic_retry_exhausted_returns_last_json_decode_error(service_with_fast_dummy):
    service, dummy = service_with_fast_dummy
    max_attempts = service._settings.LLM_VALIDATION_RETRY_MAX_ATTEMPTS

    dummy.set_prompt_response_sequence(
        "always_bad",
        [f"not json {index}" for index in range(max_attempts)],
    )

    with pytest.raises(json.JSONDecodeError):
        await service.generate("always_bad", response_model=User)

    assert dummy.prompt_response_call_indices["always_bad"] == max_attempts
