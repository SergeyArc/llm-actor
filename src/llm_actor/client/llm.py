import json
import re
from dataclasses import replace
from typing import Any, TypeVar, cast, overload

from pydantic import ValidationError

from llm_actor.client.interface import LLMClientInterface, ToolCapableClientInterface
from llm_actor.core.request import LLMRequest
from llm_actor.core.tools import LLMResponse, ToolResult
from llm_actor.exceptions import LLMServiceGeneralError
from llm_actor.logger import BrokerLogger
from llm_actor.resilience.circuit_breaker import CircuitBreaker

T = TypeVar("T", bound=object)


def _strip_schema_descriptions(schema: Any) -> Any:
    """Рекурсивно удаляет все ключи 'description' из JSON Schema.

    Применяется ко всему дереву схемы, включая $defs и вложенные объекты.
    Уменьшает размер промпта и защищает от token overflow на больших схемах.
    """
    if isinstance(schema, dict):
        return {
            key: _strip_schema_descriptions(value)
            for key, value in schema.items()
            if key != "description"
        }
    if isinstance(schema, list):
        return [_strip_schema_descriptions(item) for item in schema]
    return schema


def build_json_prompt(prompt: str, response_model: type[Any]) -> str:
    if not hasattr(response_model, "model_json_schema"):
        raise TypeError(
            f"response_model must be a Pydantic BaseModel subclass (no model_json_schema): {response_model!r}"
        )
    cleaned_schema = _strip_schema_descriptions(response_model.model_json_schema())
    serialized_schema = json.dumps(cleaned_schema, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{prompt}\n\n"
        "Return only valid JSON that strictly matches this JSON Schema:\n"
        f"{serialized_schema}"
    )


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

    def __init__(
        self,
        base_client: LLMClientInterface,
        circuit_breaker: CircuitBreaker,
        max_validation_attempts: int = 5,
    ) -> None:
        if max_validation_attempts < 1:
            raise ValueError(f"max_validation_attempts must be >= 1, got {max_validation_attempts}")
        self._client: LLMClientInterface = base_client
        self._circuit_breaker = circuit_breaker
        self._max_validation_attempts = max_validation_attempts
        self._logger = BrokerLogger.get_logger(name="llm_client")

    async def _generate_raw_text(self, request: LLMRequest) -> str:
        response_str = await self._circuit_breaker.call(self._client.generate_async, request)
        if response_str is None:
            raise ValueError("Received None response from client")
        return cast(str, response_str)

    def _validate_response_model_output(self, response_str: str, response_model: type[Any]) -> Any:
        cleaned_response = _extract_json_from_response(response_str)
        parsed_data = json.loads(cleaned_response)
        if hasattr(response_model, "model_validate"):
            return response_model.model_validate(parsed_data)
        if not isinstance(parsed_data, dict):
            raise TypeError(
                f"Expected JSON object for {response_model.__name__}, "
                f"got {type(parsed_data).__name__}"
            )
        return response_model(**parsed_data)

    def _log_validation_retry(
        self,
        validation_attempt: int,
        response_model: type[Any],
        response_str: str,
        error: Exception,
    ) -> None:
        is_last_attempt = validation_attempt >= self._max_validation_attempts

        if isinstance(error, json.JSONDecodeError):
            if is_last_attempt:
                self._logger.error(
                    f"JSON decode error after {self._max_validation_attempts} attempts: {error}, "
                    f"response preview: {response_str[:200]}"
                )
                raise error
            self._logger.warning(
                f"Validation retry {validation_attempt}/{self._max_validation_attempts} "
                f"due to JSON decode error: {error}"
            )
            return

        if isinstance(error, ValidationError):
            if is_last_attempt:
                self._logger.error(
                    f"Pydantic validation error after {self._max_validation_attempts} attempts "
                    f"for model {response_model.__name__}: {error}, "
                    f"response preview: {response_str[:200]}"
                )
                raise error
            self._logger.warning(
                f"Validation retry {validation_attempt}/{self._max_validation_attempts} "
                f"for model {response_model.__name__}: {error}"
            )
            return

        if isinstance(error, TypeError):
            if is_last_attempt:
                self._logger.error(
                    f"Type error after {self._max_validation_attempts} attempts "
                    f"for model {response_model.__name__}: {error}, "
                    f"response preview: {response_str[:200]}"
                )
                raise error
            self._logger.warning(
                f"Validation retry {validation_attempt}/{self._max_validation_attempts} "
                f"due to type error for model {response_model.__name__}: {error}"
            )
            return

        raise error

    @overload
    async def generate(self, request: LLMRequest, response_model: None = None) -> str: ...

    @overload
    async def generate(self, request: LLMRequest, response_model: type[T]) -> T: ...

    async def generate(
        self, request: LLMRequest, response_model: type[Any] | None = None
    ) -> Any | str:
        if response_model is None:
            return await self._generate_raw_text(request)

        prompt_with_schema = build_json_prompt(request.prompt, response_model)
        structured_request = replace(request, prompt=prompt_with_schema)
        for validation_attempt in range(1, self._max_validation_attempts + 1):
            response_str = await self._generate_raw_text(structured_request)

            try:
                validated = self._validate_response_model_output(response_str, response_model)
                self._logger.debug(
                    f"Response validated successfully for model {response_model.__name__}"
                )
                return validated
            except (json.JSONDecodeError, ValidationError, TypeError) as error:
                self._log_validation_retry(
                    validation_attempt=validation_attempt,
                    response_model=response_model,
                    response_str=response_str,
                    error=error,
                )

        raise RuntimeError("unreachable: validation loop exhausted without return or raise")

    async def generate_with_tools_async(
        self,
        request: LLMRequest,
        conversation: list[dict[str, Any]],
    ) -> LLMResponse:
        if not isinstance(self._client, ToolCapableClientInterface):
            raise LLMServiceGeneralError(
                f"Client {type(self._client).__name__} does not implement ToolCapableClientInterface"
            )
        return cast(
            LLMResponse,
            await self._circuit_breaker.call(
                self._client.generate_with_tools_async, request, conversation
            ),
        )

    def format_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        if not isinstance(self._client, ToolCapableClientInterface):
            raise LLMServiceGeneralError(
                f"Client {type(self._client).__name__} does not implement ToolCapableClientInterface"
            )
        return self._client.format_tool_results(results)
