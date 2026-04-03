import os
import pytest
from llm_actor import LLMBrokerService, LLMRequest

# Интеграционный тест для GigaChat через реальное API
# Запускается только с флагом --integration

@pytest.fixture
def gigachat_credentials():
    creds = os.getenv("GIGACHAT_CREDENTIALS")
    if not creds:
        pytest.skip("GIGACHAT_CREDENTIALS not set")
    return creds

@pytest.mark.asyncio
async def test_real_gigachat_tool_calling(gigachat_credentials):
    model = os.getenv("GIGACHAT_MODEL", "Sber/GigaChat-Max-V2")
    base_url = os.getenv("LLM_BASE_URL", "https://inference.airi.net:46783/v1")
    
    service = LLMBrokerService.from_gigachat(
        credentials=gigachat_credentials,
        model=model,
        base_url=base_url
    )
    
    async def get_current_time() -> str:
        """Returns current system time."""
        return "12:34:56"
        
    await service.start()
    try:
        prompt = "Который сейчас час? Используй инструмент."
        res = await service.generate(LLMRequest(prompt=prompt, tools=[get_current_time]))
        assert "12:34:56" in res or "12" in res # Минимум проверки на вызов инструмента
    finally:
        await service.stop()
