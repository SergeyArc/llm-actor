"""
Example 02: Structured Output (Pydantic V2)
Library handles schema injection and response validation automatically.
"""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llm_actor import LLMActorService


# 1. Define your output structure
class UserProfile(BaseModel):
    """Extracted user data."""

    name: str = Field(description="User's full name")
    age: int = Field(description="User's age in years")
    interests: list[str] = Field(description="Interests derived from the text")


async def main() -> None:
    # Setup configuration
    load_dotenv()
    api_key = os.environ.get("LLM_API_KEY", "your-key-here")
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL_NAME", "gpt-4o")

    # 2. Setup Service
    if base_url:
        service = LLMActorService.from_openai_compatible(
            api_key=api_key, model=model, base_url=base_url
        )
    else:
        service = LLMActorService.from_openai(api_key=api_key, model=model)

    # 3. Use via context manager
    async with service:
        # 3. Simple text to structured data
        prompt = "Extract profile: John is 30, works at Microsoft, and loves hiking and Python."
        print(f"\nProcessing: {prompt}")

        # The library takes the Pydantic model and returns its instance
        profile = await service.generate(prompt, response_model=UserProfile)

        print("\nParsed Data:")
        print(f"Name: {profile.name}")
        print(f"Age: {profile.age}")
        print(f"Interests: {', '.join(profile.interests)}\n")

        # Accessing as dict
        print("JSON Dump:")
        print(json.dumps(profile.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
