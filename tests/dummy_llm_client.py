import asyncio
import logging
import random

from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import (
    LLMServiceHTTPError,
    LLMServiceOverloadedError,
)
from llm_actor.settings import LLMActorSettings

logger = logging.getLogger(__name__)


class DummyLLMClient:
    """Test double mimicking a real LLM HTTP client."""

    def __init__(self, settings: LLMActorSettings) -> None:
        self.call_count = 0

        self.base_latency = getattr(settings, "DUMMY_BASE_LATENCY", 0.05)
        self.latency_variance = getattr(settings, "DUMMY_LATENCY_VARIANCE", 0.03)
        self.tokens_per_second = getattr(settings, "DUMMY_TOKENS_PER_SECOND", 50)

        self.overload_rate = getattr(settings, "DUMMY_OVERLOAD_RATE", 0.0)
        self.http_error_rate = getattr(settings, "DUMMY_HTTP_ERROR_RATE", 0.0)

        self.prompt_errors: dict[str, list[Exception]] = {}
        self.prompt_call_counts: dict[str, int] = {}

        self.prompt_response_sequences: dict[str, list[str]] = {}
        self.prompt_response_call_indices: dict[str, int] = {}

        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count (~4 characters per token)."""
        return max(1, len(text) // 4)

    def set_prompt_response_sequence(self, prompt_key: str, responses: list[str] | None) -> None:
        if not responses:
            self.prompt_response_sequences.pop(prompt_key, None)
            self.prompt_response_call_indices.pop(prompt_key, None)
        else:
            self.prompt_response_sequences[prompt_key] = list(responses)
            self.prompt_response_call_indices.pop(prompt_key, None)

    def _resolve_prompt_sequence_key(self, prompt: str) -> str | None:
        if prompt in self.prompt_response_sequences:
            seq = self.prompt_response_sequences[prompt]
            if seq:
                return prompt
        for key in sorted(self.prompt_response_sequences.keys(), key=len, reverse=True):
            if prompt.startswith(key):
                seq = self.prompt_response_sequences[key]
                if seq:
                    return key
        return None

    def _take_response_from_sequence(self, prompt: str) -> str | None:
        key = self._resolve_prompt_sequence_key(prompt)
        if key is None:
            return None
        sequence = self.prompt_response_sequences[key]
        index = self.prompt_response_call_indices.get(key, 0)
        if index >= len(sequence):
            return None
        self.prompt_response_call_indices[key] = index + 1
        return sequence[index]

    def _calculate_latency(self, prompt: str, response_text: str) -> float:
        """Simulated latency from prompt and response length."""
        input_tokens = self._estimate_tokens(prompt)
        output_tokens = self._estimate_tokens(response_text)

        processing_time = (input_tokens + output_tokens) / self.tokens_per_second
        variance = random.uniform(-self.latency_variance, self.latency_variance)
        latency = self.base_latency + processing_time + variance

        return max(0.01, latency)

    def set_prompt_error(self, prompt: str, error: Exception | None) -> None:
        if error is None:
            self.prompt_errors.pop(prompt, None)
        else:
            self.prompt_errors[prompt] = [error]

    def set_prompt_errors(self, prompt: str, errors: list[Exception]) -> None:
        if not errors:
            self.prompt_errors.pop(prompt, None)
        else:
            self.prompt_errors[prompt] = errors

    def _should_fail(self, prompt: str) -> Exception | None:
        """Return the next configured error for this prompt, if any."""
        self.call_count += 1

        if prompt in self.prompt_errors:
            errors = self.prompt_errors[prompt]
            if prompt not in self.prompt_call_counts:
                self.prompt_call_counts[prompt] = 0
            self.prompt_call_counts[prompt] += 1
            call_index = self.prompt_call_counts[prompt] - 1

            if call_index < len(errors):
                return errors[call_index]

        if self.overload_rate > 0:
            if random.random() < self.overload_rate:
                return LLMServiceOverloadedError("Simulated overload")

        if self.http_error_rate > 0:
            if random.random() < self.http_error_rate:
                status_code = random.choice([400, 401, 403, 500, 502, 503])
                return LLMServiceHTTPError("Simulated HTTP error", status_code=status_code)

        return None

    async def generate_async(self, request: LLMRequest) -> str:
        prompt = request.prompt
        error = self._should_fail(prompt)
        if error:
            raise error

        from_sequence = self._take_response_from_sequence(prompt)
        response_text = from_sequence if from_sequence is not None else f"Response for: {prompt}"
        latency = self._calculate_latency(prompt, response_text)

        await asyncio.sleep(latency)

        self.total_input_tokens += self._estimate_tokens(prompt)
        self.total_output_tokens += self._estimate_tokens(response_text)

        logger.debug(
            f"DummyLLMClient: processed prompt (latency={latency:.3f}s, "
            f"input_tokens={self._estimate_tokens(prompt)}, "
            f"output_tokens={self._estimate_tokens(response_text)})"
        )

        return response_text

    async def close(self) -> None:
        return None
