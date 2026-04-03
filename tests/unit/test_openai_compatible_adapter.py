import pytest
from unittest.mock import MagicMock, patch
from llm_actor.client.adapters.openai_compatible import OpenAICompatibleAdapter

@pytest.mark.asyncio
async def test_openai_compatible_adapter_initialization():
    """Проверка правильности передачи параметров в AsyncOpenAI."""
    with patch("llm_actor.client.adapters.openai.AsyncOpenAI") as mock_openai_cls:
        adapter = OpenAICompatibleAdapter(
            api_key="sk-test", 
            model="local-model", 
            base_url="http://localhost:8000/v1"
        )
        mock_openai_cls.assert_called_once_with(
            api_key="sk-test", 
            base_url="http://localhost:8000/v1"
        )
        assert adapter._model == "local-model"

@pytest.mark.asyncio
async def test_openai_compatible_adapter_is_tool_capable():
    """OpenAICompatibleAdapter должен реализовывать ToolCapableClientInterface."""
    from llm_actor.client.interface import ToolCapableClientInterface
    
    with patch("llm_actor.client.adapters.openai.AsyncOpenAI"):
        adapter = OpenAICompatibleAdapter(api_key="k", model="m", base_url="b")
        assert isinstance(adapter, ToolCapableClientInterface)
        assert hasattr(adapter, "generate_with_tools_async")
        assert hasattr(adapter, "format_tool_results")
