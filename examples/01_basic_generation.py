"""
Example 01: Basic Generation
Minimal example for quick start with any OpenAI-compatible provider.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from llm_actor import LLMActorService, LLMActorSettings


async def main() -> None:
    # Load configuration from .env
    load_dotenv()
    api_key = os.environ.get("LLM_API_KEY", "your-api-key-here")
    base_url = os.environ.get("LLM_BASE_URL")  # Optional: for vLLM/Ollama/OpenAIProxy
    model = os.environ.get("LLM_MODEL_NAME", "gpt-4o")

    # 1. Setup Service with a small worker pool
    settings = LLMActorSettings(LLM_NUM_ACTORS=2)

    if base_url:
        service = LLMActorService.from_openai_compatible(
            api_key=api_key,
            model=model,
            base_url=base_url,
            settings=settings,
        )
    else:
        service = LLMActorService.from_openai(api_key=api_key, model=model, settings=settings)

    # 2. Use the service via context manager (handles start/stop)
    async with service:
        # 3. Simple text generation
        response = await service.generate("What is the Actor Model in one sentence?")
        print("\nPrompt: What is the Actor Model in one sentence?")
        print(f"Response: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
