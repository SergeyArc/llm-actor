# LLM Actor

*Russian documentation: [docs/README.ru.md](docs/README.ru.md)*

A standalone Python package for efficient Large Language Model (LLM) request handling. It delivers high throughput via an actor pool, resilience via a circuit breaker and retries, and a built-in tool-calling loop.

## Features

- **Multi-Provider Support**: Native integration with OpenAI, Anthropic, and Sber GigaChat *(experimental; see below)*.
- **Self-Hosted LLM Ready**: Full support for vLLM, Ollama, and other OpenAI/Anthropic-compatible proxies. Works with open-weight models (Llama, Qwen, GigaChat-Max).
- **Parallel Tool Execution**: Multiple tools can run concurrently, cutting end-to-end latency in complex flows.
- **Actor-Inspired Design**: A single **priority queue** plus a pool of worker actors supervised for automatic recovery.
- **Priority Management**: Task priority levels so UI traffic can preempt background work.
- **Resilience**:
    - **Circuit Breaker**: Limits cascading provider failures with fail-fast under overload.
    - **Transport Retry**: Automatic retries on transient HTTP errors (502, 503, 504, 429) with exponential backoff.
    - **Semantic Retry**: Regenerates when the LLM response fails Pydantic validation.
- **Observability**: OpenTelemetry tracing and structured logging with context (`actor_id`, `trace_id`).
- **Structured Output**: Pydantic V2 integration for strictly typed responses.

## Requirements

- **Python**: 3.13
- **Core dependencies**: Pydantic V2, loguru, OpenTelemetry API.

## Installation

```bash
# Core only
pip install llm-actor

# With provider SDKs
pip install "llm-actor[openai]"    # OpenAI SDK
pip install "llm-actor[anthropic]" # Anthropic SDK
pip install "llm-actor[gigachat]"  # Sber GigaChat SDK

# All providers + Prometheus metrics
pip install "llm-actor[openai,anthropic,gigachat,metrics]"
```

## Quick start

### 1. Create a service

Use the built-in factories for common providers:

```python
from llm_actor import LLMActorService, LLMActorSettings

settings = LLMActorSettings(
    LLM_NUM_ACTORS=10,
    LLM_RETRY_MAX_ATTEMPTS=3,
)

# OpenAI / OpenAI-compatible (vLLM, Ollama)
service = LLMActorService.from_openai(api_key="...", model="gpt-4o", settings=settings)

# GigaChat (experimental)
service = LLMActorService.from_gigachat(credentials="...", model="GigaChat-Max-V2")
```

## Development and testing

The library splits fast unit tests from heavier integration runs.

### Environment

Copy the example env and add your keys:

```bash
cp .env.example .env
```

### Running tests

```bash
uv sync --all-extras --group dev

# 1. Unit tests (mocked, fast)
pytest tests/unit

# 2. Integration tests (real models; requires API keys in .env)
pytest tests/integration --integration
```

Integration tests are skipped if `--integration` is not passed or if required API keys are missing.

## Provider support

| Provider | Basic generation | Tool calling | Tested |
|---|---|---|---|
| OpenAI / compatible | ✅ | ✅ | ✅ Against real models |
| Anthropic | ✅ | ✅ | ✅ Against real models |
| Sber GigaChat | ✅ | ⚠️ | ❌ Not verified against the official provider |

> [!WARNING]
> **GigaChat support is experimental.** The adapter follows the GigaChat SDK docs but has not been fully validated against Sber’s official API (`gigachat.devices.sberdevices.ru`).
>
> **vLLM proxy caveat:** Tool calling requires server flags `--enable-auto-tool-choice` and `--tool-call-parser`. That is an infrastructure constraint, not a library bug.

## Design principles

- **Shared priority queue**: One queue for global prioritization.
- **Failure isolation**: A crashed worker does not take down the pool.
- **Backpressure**: Protects the provider from overload.
