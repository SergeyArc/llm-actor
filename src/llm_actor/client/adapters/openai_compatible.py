from typing import Any

from llm_actor.client.adapters.openai import OpenAIAdapter


class OpenAICompatibleAdapter(OpenAIAdapter):
    """OpenAI-compatible endpoint (vLLM, LM Studio, LocalAI, etc.) via AsyncOpenAI."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        **client_options: Any,
    ) -> None:
        super().__init__(api_key=api_key, model=model, base_url=base_url, **client_options)
