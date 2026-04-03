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

**LLM Actor** — это эффективный Actor Pool для **shared self-hosted inference** (vLLM, Ollama) или прокси-сервисов. Мы решаем задачу «последней мили»: управление конкурентными запросами, их приоритизация и гарантированный структурированный вывод без лишнего boilerplate.

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

- **High Throughput Actor Pool**: Efficiently manage hundreds of concurrent requests using a dedicated worker pool.
- **Global Resilience**: 
    - **Circuit Breaker**: Failure state is synchronized across all workers in the pool. If one worker detects a provider failure, the entire pool "fails fast" to protect your shared infrastructure.
    - **Exponential Backoff**: Automatic retries for transient HTTP errors (429, 502, 503).
    - **Semantic Validation**: Typed response validation with Pydantic; auto-retry on schema mismatch.
- **Built-in Tool Calling Loop**: Native support for complex agentic flows. Run multiple tools **in parallel** to slash latency.
- **Global Priority Queue**: Assign priorities to tasks. Ensure user-facing interactions always jump to the front of the line.
- **Multi-Provider & Self-Hosted**: 
    - Native support: OpenAI, Anthropic, Sber GigaChat.
    - Proxy support: **vLLM**, **Ollama**, and any OpenAI-compatible endpoint.
- **Deep Observability**: Full **OpenTelemetry** integration. Trace every request from the queue through the actor to the final provider response.

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

## Provider Support Matrix

| Provider | Generations | Parallel Tools | Tested |
|---|---|---|---|
| **OpenAI / compatible** | Yes | Yes | Yes Full |
| **Anthropic** | Yes | Yes | Yes Full |
| **vLLM / Ollama** | Yes | Yes* | Yes Full |
| **Sber GigaChat** | Yes | Warning | Experimental |

*\*Tool calling in vLLM requires specific server-side flags.*

---

## Architecture & Limitations (The Honest Part)

`LLM Actor` is built as an **In-Process Orchestrator**. To keep the library lightweight and universal, we made specific trade-offs:

- **Reactive Resilience**: We don't counting tokens *before* sending them (No Token Bucket). Instead, we use a **reactive chain**: `Exponential Backoff` -> `Circuit Breaker`. If a provider returns a 429, the pool backs off or pauses, protecting your infrastructure without requiring heavy tokenizers.
- **In-Memory State**: Circuit Breaker status and Queue state are local to the process. This is ideal for vertical scaling and single-node service instances, but not a replacement for distributed traffic shaping (like Redis-based Rate Limiters).
- **GPU Protection**: High-throughput mode is specifically tuned for **vLLM** and **Ollama**, where limiting the number of concurrent connections (`LLM_NUM_ACTORS`) is the most effective way to prevent GPU memory saturation.

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
