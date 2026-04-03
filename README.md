# LLM Actor

Изолированный Python-пакет для эффективного управления запросами к Large Language Models (LLM). Обеспечивает высокую производительность через пул акторов, надежность через Circuit Breaker и повторные попытки (Retry), а также встроенный цикл вызова инструментов (Tool Calling).

## Основные возможности

- **Actor-inspired Design**: Внутренняя архитектура на базе единой **Priority Queue** и пула акторов с супервизором.
- **Priority Management**: Поддержка уровней приоритета для задач (UI-запросы могут обгонять тяжелую фоновую аналитику).
- **Resilience (Отказоустойчивость)**:
    - **Circuit Breaker**: Защита от каскадных сбоев провайдера. Быстрый отказ (fail-fast) при перегрузке.
    - **Transport Retry**: Автоматические повторы при сетевых ошибках (502, 503, 504, 429) с экспоненциальной задержкой.
    - **Semantic Retry**: Повторная генерация, если ответ LLM не прошел валидацию схемы (Pydantic/JSON Schema).
- **Tool Calling Orchestration**: Встроенный цикл «запрос -> вызов инструмента -> результат -> ответ», поддерживающий вложенные вызовы и параллельное выполнение инструментов.
- **Parallel Batching**: Эффективная параллельная обработка пачек запросов через пул воркеров.
- **Structured Output**: Глубокая интеграция с Pydantic V2 для получения строго типизированных данных.
- **LLMRequest и адаптеры**: Единый DTO запроса (поддержка `messages`, `tools`, `system_prompt`); готовые адаптеры OpenAI, Anthropic и GigaChat.
- **Monitoring**: Опциональный экспорт метрик Prometheus (`llm-actor[metrics]`).
- **Tracing**: Интеграция с OpenTelemetry для глубокой трассировки жизненного цикла запроса.

## Требования

- **Python**: 3.13
- **Ключевые зависимости**: Pydantic V2, loguru, OpenTelemetry API.

## Установка

```bash
# Базовая установка (только ядро)
pip install llm-actor

# С поддержкой GigaChat
pip install llm-actor[gigachat]

# С поддержкой OpenAI
pip install llm-actor[openai]

# С метриками Prometheus
pip install llm-actor[metrics]

# Адаптер Anthropic требует ручной установки пакета anthropic
pip install anthropic
```

## Быстрый старт

### 1. Создание сервиса

Используйте удобные фабрики для популярных провайдеров:

```python
from llm_actor import LLMBrokerService, LLMBrokerSettings

settings = LLMBrokerSettings(
    LLM_NUM_ACTORS=5,                # Количество параллельных воркеров
    LLM_RETRY_MAX_ATTEMPTS=3         # Повторы при сетевых ошибках
)

# OpenAI
service = LLMBrokerService.from_openai(api_key="...", model="gpt-4o", settings=settings)

# Anthropic
service = LLMBrokerService.from_anthropic(api_key="...", model="claude-3-5-sonnet-latest")

# GigaChat (требуется [gigachat])
service = LLMBrokerService.from_gigachat(credentials="...", model="GigaChat-Pro")
```

### 2. Генерация текста и объектов

```python
import asyncio
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

async def main():
    await service.start()
    try:
        # Простой текст
        answer = await service.generate("Кто такой Python?")
        
        # Структурированный вывод
        user = await service.generate(
            "Ивлеки: Иван, 25 лет", 
            response_model=UserInfo,
            priority=0 # Высокий приоритет
        )
        print(f"{user.name}, {user.age}")
    finally:
        await service.stop()

asyncio.run(main())
```

## Продвинутое использование

### Вызов инструментов (Tool Calling)

Брокер берет на себя цикл выполнения инструментов. Вам достаточно передать функции:

```python
def get_weather(city: str) -> str:
    return f"В {city} солнечно, +25°C"

async def get_stock_price(ticker: str) -> float:
    return 150.5

# В запросе
result = await service.generate(
    "Какая погода в Москве и сколько стоят акции AAPL?",
    tools=[get_weather, get_stock_price]
)
```

Брокер сам вызовет функции (включая асинхронные), передаст результаты обратно в LLM и вернет финальный текстовый ответ.

### Пакетная обработка (Batching)

Параллельное выполнение множества запросов через пул акторов:

```python
requests = [
    ("Промпт 1", None),
    (LLMRequest(prompt="Промпт 2", temperature=0.7), UserInfo),
]
# Все запросы выполняются параллельно в рамках лимитов пула
results = await service.generate_batch(requests, priority=50)
```

> [!NOTE] 
> `generate_batch` использует параллельное выполнение онлайн-запросов. Это ускоряет обработку, но отличается от "Batch API" (оффлайн-обработки со скидками), так как требует активного соединения.

### Логирование

Пакет использует **loguru** через обёртку `BrokerLogger`. По умолчанию пакет работает в «бережном» режиме: он только обогащает записи логов метаданными (тегами акторов, пулов и трассировки), не меняя ваши настройки вывода (sinks).

**Интеграция с вашим приложением**:

1.  **По умолчанию**: Брокер просто пишет в ваш существующий конфиг loguru. Чтобы увидеть теги брокера, добавьте в ваш формат строки следующие поля: `{extra[trace_tag]}{extra[actor_tag]}{extra[pool_tag]}`.
2.  **Фирменный стиль**: Если вы хотите использовать предустановленный цветной формат вывода `llm-actor`, вызовите метод настройки в точке входа вашего приложения:

```python
from llm_actor import BrokerLogger

# ОСТОРОЖНО: Это удалит все текущие sink'и loguru и настроит вывод в stderr
BrokerLogger.setup_standard_logging(level="DEBUG")
```

**Доступные поля в `extra`**:

| Поле | Смысл |
| :--- | :--- |
| `trace_tag` | Префикс вида `[trace=<id>] `, если активен OpenTelemetry context. |
| `actor_tag` | Префикс `[actor-id] `, если лог идет из конкретного воркера. |
| `pool_tag` | Префикс `[pool-id] ` для идентификации пула. |

### Мониторинг (Prometheus)

При установке `llm-actor[metrics]` доступны метрики:
- `llm_actor_inbox_size`: размер очереди.
- `llm_batch_processing_duration_seconds`: время обработки.
- `llm_circuit_breaker_trips_total`: срабатывания защиты.
- `llm_actor_restarts_total`: перезапуски упавших акторов.

## Философия дизайна

- **Shared Priority Queue**: Единая очередь с приоритетами гарантирует честное распределение ресурсов.
- **Изоляция сбоев**: Падение воркера не влияет на остальные. Супервизор автоматически восстанавливает пул.
- **Backpressure**: Пакет защищает LLM и ваше приложение от перегрузки, удерживая нагрузку в рамках `LLM_NUM_ACTORS`.

## Разработка

```bash
uv sync --all-extras --group dev
pytest
ruff check .
mypy src
```
