"""
Example 03: Tool Calling
LLM Actor manages the orchestration loop and parallel tool execution automatically.
"""

from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv
from typing import Literal
from llm_actor import LLMActorService, LLMRequest

# 1. Define tools as regular Python functions
# The functions should have docstrings as the LLM uses them to understand the tool.
def get_current_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current weather for a specific location."""
    # Simulation
    if "London" in location:
        return f"22° {unit}, Cloudy"
    elif "Moscow" in location:
        return f"18° {unit}, Sunny"
    else:
        return f"25° {unit}, Unknown condition"

def get_city_forecast(city: str) -> str:
    """Get a 3-day weather forecast for the city."""
    # Simulation
    return f"Next 3 days in {city}: rain tomorrow, clear later."

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
        # 3. Create a request with tools
        request = LLMRequest(
            prompt="What's the weather in London and Moscow right now?",
            # You can pass regular functions; LLM Actor handles registration.
            tools=[get_current_weather, get_city_forecast], 
            # Optional: System instructions
            system_prompt="You are a helpful weather assistant."
        )

        print("\nRequest: What's the weather in London and Moscow right now?\n")
        
        # 4. LLM Actor will call both London and Moscow functions in parallel!
        answer = await service.generate(request)
        
        print(f"Final Answer: {answer}\n")

if __name__ == "__main__":
    asyncio.run(main())
