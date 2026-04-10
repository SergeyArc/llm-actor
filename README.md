# LLM Actor: High-Performance Orchestration for Self-Hosted Inference

<p align="center">
  <a href="https://pypi.org/project/llm-actor/"><img src="https://img.shields.io/pypi/v/llm-actor.svg" alt="PyPI version"></a>
  <a href="https://github.com/SergeyArc/llm-actor/actions/workflows/test.yml"><img src="https://github.com/SergeyArc/llm-actor/actions/workflows/test.yml/badge.svg" alt="Tests status"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/release/python-3130/"><img src="https://img.shields.io/badge/python-3.13-blue.svg" alt="Python 3.13"></a>
</p>

<p align="center">
  <i>Documentation: <b>English</b> | <a href="docs/README.ru.md">Russian</a></i>
</p>

**LLM Actor** is an efficient actor pool for **shared self-hosted inference** (vLLM, Ollama) or proxy services. It covers the “last mile”: managing concurrent requests, prioritizing them, and delivering guaranteed structured output without extra boilerplate.

---

## Why LLM Actor?

While cloud providers have their own rate limits, production self-hosted inference (or shared cloud endpoints) hits other bottlenecks:
- **GPU Oversaturation**: Too many concurrent requests crash the inference engine.
- **Queue Hoarding**: Background batch tasks block high-priority user UI requests.
- **Unreliable Structuring**: Getting **guaranteed Structured Output** from local models is hard.
- **Lack of Resilience**: One slow response shouldn't hang your entire orchestration layer.

**LLM Actor** addresses these via **Global Concurrency Control** (actor pool) and a **Global Circuit Breaker**. It prioritizes **reactive resilience** over proactive token counting, making it a perfect fit for private cloud infrastructure.

---

## Key Features

- **Configurable Actor Pool**: Efficiently queue and dispatch requests across a pool of managed workers.
- **Global Resilience**: 
    - **Circuit Breaker**: Failure state is shared within the process. If one worker detects a provider failure, all workers in the pool "fail fast" immediately to protect your infrastructure.
    - **Exponential Backoff**: Automatic retries for transient HTTP errors (429, 502, 503).
    - **Semantic Validation**: Typed response validation with Pydantic; auto-retry on schema mismatch.
- **Built-in Tool Calling Loop**: Native support for complex agentic flows. Run multiple tools **in parallel** to slash latency.
- **Global Priority Queue**: Assign priorities to tasks. Ensure user-facing interactions always jump to the front of the line.
- **Multi-Provider Support**: 
    - Native adapters: OpenAI, Anthropic, Sber GigaChat.
    - Proxy support: **vLLM**, **Ollama**, and any OpenAI-compatible endpoint.
- **Deep Observability**: Full **OpenTelemetry** integration. Trace every request from the queue through the actor to the final provider response.
- **Caller context & extra HTTP headers**: Python `contextvars` set before `generate()` are restored in the worker when your LLM client runs; `LLMRequest.extra_headers` is forwarded to the OpenAI and Anthropic SDKs (GigaChat logs a warning and ignores extra headers).

---

## Limitations

To keep `LLM Actor` lightweight and universal, we made specific architectural trade-offs:

- **In-process state**: Circuit Breaker status and Queue state are local to the process. Designed for single-node deployment or vertical scaling.
- **No proactive token rate limiting**: We don't counting tokens *before* sending them. Ideal for self-hosted inference (vLLM/Ollama) where TPM quotas aren't the primary bottleneck. For external APIs with strict TPM limits, we rely on **Reactive Resilience** (Backoff + CB).
- **Single-provider per pool**: A pool is tied to a single client instance. For multi-provider routing with independent circuit breaking, use separate `LLMActorService` instances.

## The Problem: What Production Code Usually Looks Like

Most teams start with a simple async client. It works great in development.

```python
# A typical "good enough" async client
class ModelClient:
    async def generate(self, messages: list[dict]) -> str:
        try:
            response = await self.client.chat.completions.create(...)
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return f"Error: {e}"  # Silently swallowed. Caller never knows.
```

Then you hit production with a shared GPU node and run 200 tasks:

```python
# Seems reasonable. It's not.
results = await asyncio.gather(*[client.generate(msg) for msg in messages])
```

**What actually happens:**
- 200 concurrent requests hit your vLLM instance → OOM or queue overflow
- Mass timeouts → every call returns `"Error: ..."` silently
- High-priority UI request waits behind 199 background batch tasks
- One slow response hangs your entire orchestration layer
- No visibility into what failed, when, or why

---

## The Fix: LLM Actor

```python
from llm_actor import LLMActorService, LLMActorSettings, Priority
from pydantic import BaseModel

class SummaryResult(BaseModel):
    summary: str
    key_points: list[str]

service = LLMActorService.from_openai(
    api_key="...",
    model="gpt-4o",
    settings=LLMActorSettings(
        LLM_NUM_ACTORS=8,          # Hard concurrency limit — vLLM won't be flooded
        LLM_MAX_QUEUE_SIZE=500,    # Bounded queue — no unbounded memory growth
    )
)

async with service:
    # Background batch — 200 tasks, queued and dispatched safely
    batch = [
        service.request(msg, response_model=SummaryResult)
        for msg in messages
    ]

    # High-priority UI request jumps the queue
    urgent = service.request(
        user_message,
        response_model=SummaryResult,
        priority=Priority.HIGH,
    )

    # Typed results — auto-retry on schema mismatch, no manual JSON parsing
    results = [r.get() for r in batch]
    user_result = urgent.get()
```

**What actually happens now:**
- Requests are dispatched via `LLM_NUM_ACTORS=8` workers — vLLM gets controlled concurrency
- `Priority.HIGH` task processes before all 200 batch tasks regardless of queue depth
- `429 / 503` → automatic exponential backoff, transparent to caller
- Failure threshold exceeded → Circuit Breaker opens, remaining tasks fail fast with `CircuitBreakerOpenError` instead of timing out
- Queue full → `OverloadError` immediately, not after a 120s timeout
- Every request traced end-to-end: queue wait time + inference time via OpenTelemetry

---

## Handling Overload Gracefully

When the system is saturated, `OverloadError` is a signal — not a crash.
Recommended pattern:

```python
from tenacity import retry, wait_exponential, retry_if_exception_type, stop_after_attempt
from llm_actor.exceptions import OverloadError

@retry(
    retry=retry_if_exception_type(OverloadError),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(5),
)
async def resilient_request(service, message):
    return service.request(message, response_model=SummaryResult).get()
```

Or degrade gracefully — fall back to a lighter model, return a cached response,
or surface an explicit "service busy" message to the user.

---

## Installation

```bash
# Install core package
pip install llm-actor

# Install with your preferred providers
pip install "llm-actor[openai,anthropic]"

# Full installation (all providers + metrics)
pip install "llm-actor[all]"
```

---

## Quick Start: 60 Seconds to Scale

Create a service and start processing tasks with priority and auto-recovery:

```python
from llm_actor import LLMActorService, LLMActorSettings, Priority
from pydantic import BaseModel

# 1. Setup Service
service = LLMActorService.from_openai(
    api_key="sk-...", 
    model="gpt-4o",
    settings=LLMActorSettings(LLM_NUM_ACTORS=10) # 10 concurrent workers
)

# 2. Define Output Schema
class UserProfile(BaseModel):
    name: str
    skills: list[str]

# 3. Use via Context Manager (handles Start/Stop automatically)
async with service:
    # 4. Queue a High-Priority Task
    request = service.request(
        "Extract profile from: Alex is a Senior Python Dev with LLM expertise.",
        response_model=UserProfile,
        priority=Priority.HIGH
    )

    # 5. Get Your Results (Blocking or Async)
    result = request.get()
    print(f"Found: {result.name} with skills: {result.skills}")
```

---

## Full Control over LLM Settings

Every request can be fine-tuned via `LLMRequest`. We support common parameters natively and provider-specific ones via the `extra` field:

```python
from llm_actor import LLMRequest

request = LLMRequest(
    prompt="Explain quantum entanglement",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    stop_sequences=["###"],
    # For parameters not supported natively:
    extra={"presence_penalty": 0.5, "seed": 42}
)
await service.generate(request)
```

---

## Provider Support Matrix

| Provider | Generations | Parallel Tools | Tested |
|---|---|---|---|
| **OpenAI / compatible** | Yes | Yes | Yes Full |
| **Anthropic** | Yes | Yes | Yes Full |
| **vLLM / Ollama** | Yes | Yes* | Yes Full |
| **Sber GigaChat** | Yes | Warning | Experimental |

*\*Tool calling in vLLM requires specific server-side flags.*

---

## Architecture: Reactive Resilience

`LLM Actor` is built as an **In-Process Orchestrator**. Instead of complex pre-emptive traffic shaping, we use a **reactive chain**: `Exponential Backoff` (managed by the client) -> `Circuit Breaker` (managed by the pool). 

This approach minimizes internal overhead and is ideal for self-hosted inference (like vLLM), where TPM quotas are absent and reactive 429 handling is sufficient. It prevents GPU memory saturation by limiting active connections (`LLM_NUM_ACTORS`) while ensuring high-priority requests are always processed first.

---

## Contributing

We love contributions! Whether it's adding a new provider adapter, fixing a bug, or improving documentation.

1. Fork the repo.
2. Install dev dependencies: `uv sync --all-extras --group dev`
3. Run tests: `pytest tests/unit`
4. Submit your PR!

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## Examples & Advanced Usage

Check out the [examples/](examples/) directory for complete, runnable scripts:

1.  **[Basic Generation](examples/01_basic_generation.py)**: Quick start with any provider.
2.  **[Structured Output](examples/02_structured_output.py)**: Extract data into Pydantic models.
3.  **[Tool Calling](examples/03_tool_calling.py)**: Orchestrate complex agentic loops with parallel tool execution.

---

<p align="center">Built for the AI Developer Community.</p>
