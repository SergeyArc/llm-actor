from tests.models import User


async def test_ask_without_response_model(service, mock_llm_response):
    """generate without response_model returns str."""
    prompt = "What is the weather?"
    expected_response = "Sunny and warm"
    mock_llm_response[prompt] = expected_response

    result = await service.generate(prompt)

    assert isinstance(result, str)
    assert result == expected_response


async def test_ask_with_response_model(service, mock_llm_response):
    """generate with response_model returns validated model instance."""
    prompt = "Who is Alice?"
    json_response = '{"name": "Alice", "age": 30}'
    mock_llm_response[prompt] = json_response

    result = await service.generate(prompt, response_model=User)

    assert isinstance(result, User)
    assert result.name == "Alice"
    assert result.age == 30


async def test_ask_empty_prompt(service, mock_llm_response):
    """Empty string prompt is handled."""
    prompt = ""
    mock_llm_response[prompt] = "Empty prompt response"

    result = await service.generate(prompt)

    assert isinstance(result, str)
    assert result == "Empty prompt response"


async def test_ask_health_status(service, mock_llm_response):
    """Health status stays healthy after requests."""
    prompts = ["test1", "test2", "test3"]
    for prompt in prompts:
        mock_llm_response[prompt] = f"Response for {prompt}"

    for prompt in prompts:
        await service.generate(prompt)

    health = service.pool.get_health_status()
    assert health.status == "healthy"
    assert health.alive_actors == health.total_actors
    assert health.alive_actors > 0
