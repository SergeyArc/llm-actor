import json
import re
from typing import Any, TypeVar, overload

from pydantic import ValidationError

from llm_broker.client.interface import LLMClientInterface
from llm_broker.logger import BrokerLogger
from llm_broker.resilience.circuit_breaker import CircuitBreaker

T = TypeVar("T", bound=object)


def _extract_json_from_response(response: str) -> str:
    """Извлекает JSON из ответа, удаляя markdown обёртки и лишний текст.

    Порядок попыток:
    1. Строгий code-fence ````json\\n...\\n````
    2. Гибкий code-fence ```` ```...``` ````
    3. Поиск первого валидного JSON-объекта или массива через JSONDecoder.raw_decode
       (корректно обрабатывает вложенные структуры и фигурные скобки в строках)
    """
    if response is None:
        raise TypeError("Got None response from model")

    response = response.strip()

    strict_pattern = r"```json\n(.*?)\n```"
    match = re.search(strict_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    flexible_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    match = re.search(flexible_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Используем JSONDecoder для корректного поиска объекта или массива.
    # raw_decode обрабатывает вложенность и фигурные скобки внутри строк.
    decoder = json.JSONDecoder()
    for start_char in ("{", "["):
        idx = response.find(start_char)
        while idx != -1:
            try:
                _, end = decoder.raw_decode(response, idx)
                return response[idx:end].strip()
            except json.JSONDecodeError:
                idx = response.find(start_char, idx + 1)

    return response


class LLMClientWithCircuitBreaker:
    """LLM client wrapper with circuit breaker"""

    def __init__(self, base_client: LLMClientInterface, circuit_breaker: CircuitBreaker) -> None:
        self._client: LLMClientInterface = base_client
        self._circuit_breaker = circuit_breaker
        self._logger = BrokerLogger.get_logger(name="llm_client")

    @overload
    async def generate(self, prompt: str, response_model: None = None) -> str: ...

    @overload
    async def generate(self, prompt: str, response_model: type[T]) -> T: ...

    async def generate(self, prompt: str, response_model: type[Any] | None = None) -> Any | str:
        response_str = await self._circuit_breaker.call(self._client.generate_async, prompt)

        if response_str is None:
            raise ValueError("Received None response from client")

        if response_model is not None:
            try:
                cleaned_response = _extract_json_from_response(response_str)
                parsed_data = json.loads(cleaned_response)
                validated = response_model(**parsed_data)
                self._logger.debug(
                    f"Response validated successfully for model {response_model.__name__}"
                )
                return validated
            except json.JSONDecodeError as e:
                self._logger.error(
                    f"JSON decode error: {e}, response preview: {response_str[:200]}"
                )
                raise ValueError(f"Failed to parse JSON: {e}") from e
            except ValidationError as e:
                self._logger.error(
                    f"Pydantic validation error for model {response_model.__name__}: {e}, "
                    f"response preview: {response_str[:200]}"
                )
                raise ValueError(f"Validation failed: {e}") from e

        return response_str
