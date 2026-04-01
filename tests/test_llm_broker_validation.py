import pytest

from tests.models import User


async def test_ask_with_invalid_json_response_model(service, mock_llm_response):
    """Тест обработки невалидного JSON при использовании response_model."""
    prompt = "invalid_json"
    mock_llm_response[prompt] = "not a json string"

    with pytest.raises(ValueError, match="Failed to parse JSON"):
        await service.generate(prompt, response_model=User)


async def test_ask_with_mismatched_json_structure(service, mock_llm_response):
    """Тест обработки JSON с несоответствующей структурой модели."""
    prompt = "mismatched_json"
    mock_llm_response[prompt] = '{"name": "Alice"}'

    with pytest.raises(ValueError, match="Validation failed"):
        await service.generate(prompt, response_model=User)
