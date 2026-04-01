import json

from pydantic import ValidationError

from tests.models import User


async def test_ask_batch_mixed_requests(service, mock_llm_response):
    """Тест пакетной обработки смешанных запросов с/без response_model."""
    requests = [
        ("What is the weather?", None),
        ("Who is Alice?", User),
        ("What is 2+2?", None),
        ("Who is Bob?", User),
        ("Tell me a joke", None),
    ]

    mock_llm_response["What is the weather?"] = "Sunny"
    mock_llm_response["Who is Alice?"] = '{"name": "Alice", "age": 25}'
    mock_llm_response["What is 2+2?"] = "4"
    mock_llm_response["Who is Bob?"] = '{"name": "Bob", "age": 35}'
    mock_llm_response["Tell me a joke"] = "Why did the chicken cross the road?"

    results = await service.generate_batch(requests)

    assert len(results) == len(requests)
    assert isinstance(results[0], str)
    assert isinstance(results[1], User)
    assert results[1].name == "Alice"
    assert results[1].age == 25
    assert isinstance(results[2], str)
    assert isinstance(results[3], User)
    assert results[3].name == "Bob"
    assert results[3].age == 35
    assert isinstance(results[4], str)

    for result in results:
        assert not isinstance(result, Exception)


async def test_ask_batch_handles_errors(service, mock_llm_response):
    """Тест обработки ошибок в пакетных запросах."""
    requests = [
        ("success", None),
        ("error", None),
        ("success2", None),
    ]

    mock_llm_response["success"] = "OK"
    mock_llm_response["error"] = Exception("Simulated API failure")
    mock_llm_response["success2"] = "OK2"

    results = await service.generate_batch(requests)

    assert len(results) == len(requests)
    assert isinstance(results[0], str)
    assert isinstance(results[1], Exception)
    assert isinstance(results[2], str)
    assert results[0] == "OK"
    assert results[2] == "OK2"


async def test_ask_batch_empty_list(service):
    """Тест пакетной обработки пустого списка запросов."""
    requests = []

    results = await service.generate_batch(requests)

    assert isinstance(results, list)
    assert len(results) == 0


async def test_ask_batch_single_request(service, mock_llm_response):
    """Тест пакетной обработки одного запроса."""
    prompt = "single_request"
    mock_llm_response[prompt] = "Single response"
    requests = [(prompt, None)]

    results = await service.generate_batch(requests)

    assert len(results) == 1
    assert isinstance(results[0], str)
    assert results[0] == "Single response"


async def test_ask_batch_with_pydantic_validation_error(service, mock_llm_response):
    """Тест ошибки валидации Pydantic при невалидном JSON в response_model."""
    requests = [
        ("valid", User),
        ("invalid_json", User),
        ("mismatched", User),
    ]

    mock_llm_response["valid"] = '{"name": "Alice", "age": 30}'
    mock_llm_response["invalid_json"] = "not json"
    mock_llm_response["mismatched"] = '{"name": "Bob"}'

    results = await service.generate_batch(requests)

    assert len(results) == 3
    assert isinstance(results[0], User)
    assert results[0].name == "Alice"
    assert isinstance(results[1], Exception)
    assert isinstance(results[1], json.JSONDecodeError)
    assert isinstance(results[2], Exception)
    assert isinstance(results[2], ValidationError)
