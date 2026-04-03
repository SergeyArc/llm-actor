import os

import pytest

from llm_actor import LLMActorService, LLMRequest

# Integration tests for OpenAI-compatible APIs (vLLM, Ollama).
# Run with pytest --integration.


@pytest.fixture
def openai_api_key():
    key = os.getenv("LLM_API_KEY")
    if not key:
        pytest.skip("LLM_API_KEY not set")
    return key


@pytest.mark.asyncio
async def test_real_openai_compatible_tool_calling(openai_api_key):
    model = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

    service = LLMActorService.from_openai(api_key=openai_api_key, model=model, base_url=base_url)

    async def get_current_time() -> str:
        """Returns current system time."""
        return "12:34:56"

    await service.start()
    try:
        prompt = "Который сейчас час?"
        res = await service.generate(LLMRequest(prompt=prompt, tools=[get_current_time]))
        assert "12:34:56" in res or "12" in res
    finally:
        await service.stop()
