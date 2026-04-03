# LLM Actor

Изолированный Python-пакет для эффективного управления запросами к Large Language Models (LLM). Обеспечивает высокую производительность через пул акторов, надежность через Circuit Breaker и повторные попытки (Retry), а также опциональный мониторинг через Prometheus (`llm-actor[metrics]`).

## Основные возможности

- **Actor-inspired Design**: Внутренняя архитектура на базе единой **Priority Queue** и пула акторов с супервизором.
- **Priority Management**: Поддержка уровней приоритета для задач (UI-запросы могут обгонять тяжелую фоновую аналитику).
- **Resilience (Отказоустойчивость)**:
    - **Circuit Breaker**: Защита от каскадных сбоев провайдера. Быстрый отказ (fail-fast) при перегрузке.
    - **Transport Retry**: Автоматические повторы при сетевых ошибках (502, 503, 504, 429) с экспоненциальной задержкой.
    - **Semantic Retry**: Повторная генерация, если ответ LLM не прошел валидацию схемы (JSON Schema).
- **Embedded Batching**: Прозрачная группировка запросов для эффективного использования Batch API провайдеров (скидки до 50% у ряда вендоров на апрель 2026).
- **Structured Output**: Глубокая интеграция с Pydantic V2 для получения строго типизированных данных без оверхеда на сторонние фреймворки.
- **LLMRequest и адаптеры**: Единый DTO запроса; готовые адаптеры OpenAI, OpenAI-compatible и Anthropic с маппингом ошибок провайдера в исключения брокера; фабрики `LLMBrokerService.from_*`.
- **Monitoring**: Опциональный экспорт метрик Prometheus при установке extra `[metrics]` (latency, error rate, actor pool state).
- **Логирование**: Единый loguru-based `BrokerLogger` с контекстом пула/актора/запроса и опциональной подстановкой `trace_id` из OpenTelemetry в каждую запись.
- **Backpressure Control**: Управление конкурентностью для локальных моделей (vLLM, Ollama), предотвращающее перегрузку GPU.

## Требования

- **Python**: 3.13
- **Ключевые зависимости**: Pydantic V2, loguru, OpenTelemetry API (только для контекста трасс и поля `trace_id` в логах; экспорт спанов настраивает приложение-хост). Метрики Prometheus — через optional extra `[metrics]` (`prometheus-client`).

## Установка

```bash
# Базовая установка
pip install llm-actor

# С поддержкой GigaChat
pip install llm-actor[gigachat]

# С поддержкой OpenAI (SDK для встроенного адаптера)
pip install llm-actor[openai]

# С метриками Prometheus (по умолчанию брокер работает без них)
pip install llm-actor[metrics]

# Несколько extras сразу (пример)
pip install llm-actor[openai,metrics]

# Адаптер Anthropic: отдельно ставится пакет anthropic (в pyproject нет extra [anthropic])
pip install anthropic
```

Без extra `[metrics]` зависимость `prometheus-client` не ставится: брокер работает без экспорта метрик. Явный `MetricsCollector()` без установленного extra завершится `ImportError` с подсказкой установить `[metrics]`.

## Быстрый старт

### 1. Запрос `LLMRequest` и клиент

Транспортный слой принимает `LLMRequest` (промпт, опционально `temperature`, `max_tokens`, `system_prompt`, `stop_sequences`, произвольные ключи провайдера в `extra`). Публичный метод `LLMBrokerService.generate` также принимает обычную строку — она оборачивается в `LLMRequest(prompt=...)`.

Реализуйте `LLMClientInterface` или используйте встроенные адаптеры (опциональные зависимости `openai`, `anthropic`):

```python
from llm_actor import LLMClientInterface, LLMRequest
import asyncio

class MyLLMClient(LLMClientInterface):
    async def generate_async(self, request: LLMRequest) -> str:
        await asyncio.sleep(0.1)
        return f"Echo: {request.prompt}"

    async def close(self) -> None:
        pass
```

Фабрики сервиса (маппинг ошибок провайдера в исключения брокера внутри адаптера):

```python
from llm_actor import LLMBrokerService

# pip install llm-actor[openai]
service = LLMBrokerService.from_openai(api_key="...", model="gpt-4o")
# OpenAI-compatible base_url (vLLM, LM Studio, …)
service = LLMBrokerService.from_openai_compatible(
    api_key="...", model="...", base_url="http://localhost:11434/v1"
)
# pip install anthropic
service = LLMBrokerService.from_anthropic(api_key="...", model="claude-3-5-sonnet-20241022")
```

### 2. Запуск сервиса

```python
import asyncio
from pydantic import BaseModel
from llm_actor import LLMBrokerService, LLMBrokerSettings, LLMRequest

class MyResponse(BaseModel):
    answer: str
    confidence: float

async def main():
    settings = LLMBrokerSettings(
        LLM_MAX_CONCURRENT=5,
        LLM_VALIDATION_RETRY_MAX_ATTEMPTS=3
    )

    # Клиент из п.1; либо: service = LLMBrokerService.from_openai(..., settings=settings)
    base_client = MyLLMClient()
    # Метрики: по умолчанию только при pip install llm-actor[metrics].
    # Без extra коллектор не создаётся; свой: metrics=MetricsCollector() после установки [metrics].
    service = LLMBrokerService(base_client=base_client, settings=settings)

    await service.start()
    
    try:
        text = await service.generate("Расскажи шутку про Python", priority=0)
        print(f"Текст: {text}")

        tuned = await service.generate(
            LLMRequest(
                prompt="Коротко: что такое asyncio?",
                temperature=0.2,
                system_prompt="Отвечай по-русски.",
            )
        )
        print(tuned)

        obj = await service.generate(
            "Извлеки данные: уверенность 0.9, ответ 'Привет'",
            response_model=MyResponse,
            priority=5,
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
from llm_actor import LLMRequest

# MyResponse — ваша Pydantic-модель (как в примере выше)
requests = [
    ("Промпт 1", None),
    (LLMRequest(prompt="Промпт 2", temperature=0.5), MyResponse),
    ("Промпт 3", None),
]
results = await service.generate_batch(requests, priority=20)
```

### Мониторинг

Prometheus — **опционально** (`prometheus-client` только в extra `[metrics]`).

При `pip install llm-actor[metrics]` по умолчанию `LLMBrokerService` создаёт `MetricsCollector`. Без extra внутренний `metrics` остаётся `None` (нет накладных расходов и зависимости). Свой коллектор передайте в `LLMBrokerService(..., metrics=...)`, либо используйте `default_metrics_collector()` и `is_prometheus_metrics_available()` из пакета.

Список метрик ниже актуален **только** при установленном `[metrics]` и созданном `MetricsCollector`:

Пакет собирает следующие ключевые метрики Prometheus:
- `llm_actor_inbox_size`: текущий размер общей очереди задач (shared priority queue).
- `llm_batch_processing_duration_seconds`: гистограмма времени обработки батча (по акторам).
- `llm_batches_processed_total`: общее количество успешно обработанных батчей.
- `llm_batches_failed_total`: количество батчей, завершившихся ошибкой.
- `llm_circuit_breaker_trips_total`: счетчик срабатываний Circuit Breaker.
- `llm_actor_restarts_total`: счетчик перезапусков акторов супервизором.

### Логирование

Пакет пишет диагностические сообщения через **`BrokerLogger`** ([`llm_actor.logger`](src/llm_actor/logger.py)) — обёртку над **loguru** с общим форматом и патчером записей.

**Поведение**

- При первом обращении к `BrokerLogger` (например, из `LLMBrokerService`) выполняется однократная настройка: снимаются все текущие sink’и loguru и добавляется один вывод в **stderr** с цветным форматом, уровнем по умолчанию `INFO` и полями `backtrace` / `diagnose` для ошибок.
- Уровень до первого лога можно задать явно: `BrokerLogger.configure(level="DEBUG")` (повторный вызов не меняет уже сконфигурированный логгер).

**Контекст в строке лога**

Патчер заполняет служебные поля `extra`, которые попадают в шаблон сообщения:

| Поле (extra) | Смысл |
| :--- | :--- |
| `trace_tag` | Если установлен **OpenTelemetry** `TracerProvider` и у текущего контекста есть валидный span — префикс вида `[trace=<32 hex символа>] `; иначе пустая строка. Позволяет сопоставлять логи со спанами в Jaeger/Tempo и т.п. |
| `actor_tag` | Префикс `[<actor_id>] `, если логгер создан через `BrokerLogger.bind_context(actor_id=...)`. |
| `pool_tag` | Префикс `[pool <первые 8 символов pool_id>] ` при переданном `pool_id`. |

Внутри пула и акторов используется `BrokerLogger.bind_context(pool_id=..., actor_id=..., request_id=...)`: в формат вывода напрямую попадают только теги пула и актора; `request_id` остаётся в `extra` и доступен для расширения собственных sink’ов или фильтров.

**Интеграция с вашим приложением**

- Если вы уже настраиваете loguru глобально, имейте в виду: первый вызов `BrokerLogger` **заменит** список sink’ов на конфигурацию пакета. Чтобы избежать неожиданного порядка импортов, вызовите `BrokerLogger.configure(level=...)` в точке входа приложения до остальных компонентов `llm_actor` либо настройте свой sink после старта сервиса, если нужен кастомный вывод.
- Корреляция лог↔трассы работает там, где при обработке запроса активен OTEL-контекст (как у брокера после настройки провайдера на стороне хоста).

## Сравнение с аналогами (2026)

| Возможность | llm-actor | LiteLLM | Instructor | LangChain |
| :--- | :---: | :---: | :---: | :---: |
| **Режим работы** | Embedded (библиотека) | Proxy / SDK | SDK wrapper | Framework |
| **Worker Pool** | ✅ Встроенный | ❌ | ❌ | ❌ |
| **Авто-батчинг** | ✅ Из коробки | ❌ | ❌ | ⚠️ Частично |
| **Circuit Breaker** | ✅ Интегрирован | ✅ (только в прокси) | ❌ | ❌ |
| **Validation Retry**| ✅ Семантический | ❌ | ✅ | ❌ |
| **Prometheus-метрики** | ✅ Опционально (`[metrics]`) | ⚠️ В прокси | ❌ | ⚠️ Частично |

## Философия дизайна

`llm-actor` — это **Actor-inspired concurrent worker pool**, прагматичная адаптация идей Erlang/Elixir для мира Python AsyncIO.

- **Shared Priority Queue**: Все задачи попадают в единую очередь с приоритетами. Это гарантирует, что высокоприоритетные задачи обрабатываются в первую очередь, а нагрузка распределяется между акторами максимально равномерно (pull model).
- **Изоляция сбоев**: Падение одного воркера при обработке сложного промпта не аффектит весь пул. Супервизор (`SupervisedActorPool`) автоматически перезапустит воркера и вернет его в строй.
- **Backpressure**: Пакет выступает защитным буфером между вашим приложением и LLM. Вместо бесконечного создания задач, вызывающих `MemoryError`, брокер удерживает нагрузку в пределах заданного пула.
- **Лёгкое ядро**: Без тяжёлых оркестраторов. Ядро опирается на `pydantic`, `loguru` и `opentelemetry-api` (без обязательного SDK/экспортёра в рантайме библиотеки — их подключает хост). `prometheus-client` подключается только с extra `[metrics]`.

## Производительность (Benchmark)

Использование встроенного батчинга позволяет достичь значительного ускорения на больших объемах данных за счет параллелизации и оптимального использования HTTP-соединений:

- **100 запросов (Data Extraction)**:
    - Последовательно: ~45 секунд
    - `llm-actor` (pool=10): ~4.8 секунды (**ускорение в 9.3 раза**)


## Разработка

Для запуска тестов и линтеров:

```bash
cd llm_actor
uv sync --all-extras --group dev   # extras + dev (в dev входит prometheus-client для pytest/скриптов)
pytest
ruff check .
mypy src
```

## Roadmap

Ниже приведен план развития `llm-actor` на ближайшее время:

1.  **Провайдеры "из коробки"**: Адаптеры OpenAI, OpenAI-compatible и Anthropic уже входят в пакет (см. `LLMBrokerService.from_*`). Дополнительные вендоры и унификация GigaChat/vLLM — по мере необходимости.
2.  **OpenTelemetry**: В пакете уже есть спаны жизненного цикла (брокер, очередь, актор, LLM-запрос, валидация, tool calls) и проброс контекста; хост подключает SDK и экспортёры. Дальнейшее развитие — семантические конвенции, тонкая настройка атрибутов, опциональные переключатели «шумных» спанов.
3.  **Cost & Token Tracking**: Встроенный подсчет токенов и примерной стоимости каждого запроса на основе актуальных тарифов провайдеров. Экспорт статистики через метрики Prometheus (при установленном extra `[metrics]`).
4.  **Priority Queuing Improvements**: Динамическая переприоритизация задач в очереди на основе их "возраста" (предотвращение голодания низкоприоритетных задач).

