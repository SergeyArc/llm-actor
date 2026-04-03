import os

import pytest

from llm_actor import LLMActorService, LLMRequest

# Integration tests for Anthropic SDK (or Qwen-compatible endpoints).
# Run with pytest --integration.


@pytest.fixture
def anthropic_api_key():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


@pytest.mark.asyncio
async def test_real_anthropic_tool_calling(anthropic_api_key):
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    service = LLMActorService.from_anthropic(
        api_key=anthropic_api_key, model=model, base_url=base_url
    )

    async def get_weather(city: str) -> str:
        """Returns weather in city."""
        return f"Sunny in {city}, +25C"

    await service.start()
    try:
        prompt = "What is the weather in Moscow?"
        res = await service.generate(LLMRequest(prompt=prompt, tools=[get_weather]))
        assert "Moscow" in res
        assert "25" in res or "Sunny" in res
    finally:
        await service.stop()
