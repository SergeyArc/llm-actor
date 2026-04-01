# LLM Actor

Изолированный Python-пакет для эффективного управления запросами к Large Language Models (LLM). Обеспечивает высокую производительность через пул акторов, надежность через Circuit Breaker и повторные попытки (Retry), а также мониторинг через Prometheus.

## Основные возможности

- **Actor-inspired Design**: Внутренняя архитектура на базе пула акторов с супервизором для изоляции сбоев воркеров.
- **Resilience (Отказоустойчивость)**:
    - **Circuit Breaker**: Защита от каскадных сбоев провайдера. Быстрый отказ (fail-fast) при перегрузке.
    - **Transport Retry**: Автоматические повторы при сетевых ошибках (502, 503, 504, 429) с экспоненциальной задержкой.
    - **Semantic Retry**: Повторная генерация, если ответ LLM не прошел валидацию схемы (JSON Schema).
- **Embedded Batching**: Прозрачная группировка запросов для эффективного использования Batch API провайдеров (скидки до 50% у ряда вендоров на апрель 2026).
- **Structured Output**: Глубокая интеграция с Pydantic V2 для получения строго типизированных данных без оверхеда на сторонние фреймворки.
- **Monitoring**: Нативный экспорт метрик Prometheus (latency, error rate, actor pool state).
- **Backpressure Control**: Управление конкурентностью для локальных моделей (vLLM, Ollama), предотвращающее перегрузку GPU.

## Требования

- **Python**: 3.13
- **Ключевые зависимости**: Pydantic V2, loguru, prometheus-client.

## Установка

```bash
# Базовая установка
pip install llm-actor

# С поддержкой GigaChat
pip install llm-actor[gigachat]

# С поддержкой OpenAI
pip install llm-actor[openai]
```

## Быстрый старт

### 1. Реализация клиента

Для работы брокера необходимо предоставить реализацию `LLMClientInterface`.

```python
from llm_actor import LLMClientInterface
import asyncio

class MyLLMClient(LLMClientInterface):
    async def generate_async(self, prompt: str) -> str:
        # Ваш код обращения к API (без логики retry)
        await asyncio.sleep(0.1)
        return "Generated response"

    async def close(self) -> None:
        pass
```

### 2. Запуск сервиса

```python
import asyncio
from pydantic import BaseModel
from llm_actor import LLMBrokerService, LLMBrokerSettings

class MyResponse(BaseModel):
    answer: str
    confidence: float

async def main():
    settings = LLMBrokerSettings(
        LLM_MAX_CONCURRENT=5,
        LLM_VALIDATION_RETRY_MAX_ATTEMPTS=3
    )
    
    base_client = MyLLMClient()
    service = LLMBrokerService(base_client=base_client, settings=settings)
    
    await service.start()
    
    try:
        # Обычный текстовый запрос
        text = await service.generate("Расскажи шутку про Python")
        print(f"Текст: {text}")
        
        # Запрос с валидацией через Pydantic
        obj = await service.generate(
            "Извлеки данные: уверенность 0.9, ответ 'Привет'", 
            response_model=MyResponse
        )
        print(f"Объект: {obj.answer} ({obj.confidence})")
        
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Продвинутое использование

### Пакетная обработка (Batching)

Сервис поддерживает эффективную параллельную обработку пачки запросов:

```python
requests = [
    ("Промпт 1", None),
    ("Промпт 2", MyResponse),
    ("Промпт 3", None),
]
results = await service.generate_batch(requests)
```

### Мониторинг

Пакет собирает следующие метрики Prometheus:
- `llm_actor_requests_total`: общее количество запросов.
- `llm_actor_request_duration_seconds`: гистограмма времени выполнения запросов.
- `llm_actor_circuit_breaker_state`: текущее состояние circuit breaker (open/closed/half-open).
- `llm_actor_actor_pool_size`: текущее количество активных акторов.

## Сравнение с аналогами (2026)

| Возможность | llm-actor | LiteLLM | Instructor | LangChain |
| :--- | :---: | :---: | :---: | :---: |
| **Режим работы** | Embedded (библиотека) | Proxy / SDK | SDK wrapper | Framework |
| **Worker Pool** | ✅ Встроенный | ❌ | ❌ | ❌ |
| **Авто-батчинг** | ✅ Из коробки | ❌ | ❌ | ⚠️ Частично |
| **Circuit Breaker** | ✅ Интегрирован | ✅ (только в прокси) | ❌ | ❌ |
| **Validation Retry**| ✅ Семантический | ❌ | ✅ | ❌ |

## Философия дизайна

`llm-actor` — это **Actor-inspired concurrent worker pool**, прагматичная адаптация идей Erlang/Elixir для мира Python AsyncIO.

- **Изоляция сбоев**: Падение одного воркера при обработке сложного промпта не аффектит весь пул. Супервизор (`SupervisedActorPool`) автоматически перезапустит воркера и вернет его в строй.
- **Backpressure**: Пакет выступает защитным буфером между вашим приложением и LLM. Вместо бесконечного создания задач, вызывающих `MemoryError`, брокер удерживает нагрузку в пределах заданного пула.
- **Zero-Dependency Core**: Мы не тянем за собой тяжелые фреймворки. Ядро брокера использует только `pydantic`, `loguru` и `prometheus-client`.

## Производительность (Benchmark)

Использование встроенного батчинга позволяет достичь значительного ускорения на больших объемах данных за счет параллелизации и оптимального использования HTTP-соединений:

- **100 запросов (Data Extraction)**:
    - Последовательно: ~45 секунд
    - `llm-actor` (pool=10): ~4.8 секунды (**ускорение в 9.3 раза**)


## Разработка

Для запуска тестов и линтеров:

```bash
cd llm_actor
uv sync --all-extras
pytest
ruff check .
mypy src
```
