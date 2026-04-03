"""Built-in LLM provider adapters; each depends on optional extras."""

__all__: list[str] = []

try:
    from llm_actor.client.adapters.openai import OpenAIAdapter
    from llm_actor.client.adapters.openai_compatible import OpenAICompatibleAdapter

    __all__ += ["OpenAIAdapter", "OpenAICompatibleAdapter"]
except ImportError:
    pass

try:
    from llm_actor.client.adapters.anthropic import AnthropicAdapter

    __all__ += ["AnthropicAdapter"]
except ImportError:
    pass

try:
    from llm_actor.client.adapters.gigachat import GigaChatAdapter

    __all__ += ["GigaChatAdapter"]
except ImportError:
    pass
