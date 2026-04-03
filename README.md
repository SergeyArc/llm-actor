# LLM Actor

Изолированный Python-пакет для эффективного управления запросами к Large Language Models (LLM). Обеспечивает высокую производительность через пул акторов, надежность через Circuit Breaker и повторные попытки (Retry), а также встроенный цикл вызова инструментов (Tool Calling).

## Основные возможности

- **Multi-Provider Support**: Нативная интеграция с OpenAI, Anthropic и Sber GigaChat *(экспериментально, см. ниже)*.
- **Self-Hosted LLM Ready**: Полная поддержка vLLM, Ollama и других OpenAI/Anthropic-совместимых прокси. Работает с open-weight моделями (Llama, Qwen, GigaChat-Max).
- **Parallel Tool Execution**: Выполнение запросов к нескольким инструментам одновременно, что радикально снижает общую задержку (latency) при сложных сценариях.
- **Actor-inspired Design**: Внутренняя архитектура на базе единой **Priority Queue** и пула воркеров (акторов) с супервизором для автоматического восстановления.
- **Priority Management**: Поддержка уровней приоритета для задач (UI-запросы могут мгновенно обгонять фоновую аналитику).
- **Resilience (Отказоустойчивость)**:
    - **Circuit Breaker**: Защита от каскадных сбоев провайдера. Быстрый отказ (fail-fast) при перегрузке.
    - **Transport Retry**: Автоматические повторы при сетевых ошибках (502, 503, 504, 429) с экспоненциальной задержкой.
    - **Semantic Retry**: Повторная генерация, если ответ LLM не прошел Pydantic-валидацию.
- **Observability**: Глубокая трассировка через OpenTelemetry и структурированное логирование с контекстом (actor_id, trace_id).
- **Structured Output**: Интеграция с Pydantic V2 для гарантированного получения строго типизированных данных.

## Требования

- **Python**: 3.13
- **Ключевые зависимости**: Pydantic V2, loguru, OpenTelemetry API.

## Установка

```bash
# Базовая установка (только ядро)
pip install llm-actor

# С поддержкой провайдеров
pip install "llm-actor[openai]"    # OpenAI SDK
pip install "llm-actor[anthropic]" # Anthropic SDK
pip install "llm-actor[gigachat]"  # Sber GigaChat SDK

# Все провайдеры + метрики Prometheus
pip install "llm-actor[openai,anthropic,gigachat,metrics]"
```

## Быстрый старт

### 1. Создание сервиса

Используйте удобные фабрики для популярных провайдеров:

```python
from llm_actor import LLMActorService, LLMActorSettings

# Конфигурация брокера
settings = LLMActorSettings(
    LLM_NUM_ACTORS=10,               # Размер пула воркеров
    LLM_RETRY_MAX_ATTEMPTS=3,        # Повторы при сетевых ошибках
)

# OpenAI / OpenAI Compatible (vLLM, Ollama)
service = LLMActorService.from_openai(api_key="...", model="gpt-4o", settings=settings)

# GigaChat (экспериментально)
service = LLMActorService.from_gigachat(credentials="...", model="GigaChat-Max-V2")
```

## Разработка и Тестирование

Библиотека разделяет быстрые Unit-тесты и тяжелые интеграционные проверки.

### Настройка окружения
Скопируйте пример настроек и укажите свои ключи:
```bash
cp .env.example .env
```

### Запуск тестов
```bash
uv sync --all-extras --group dev

# 1. Unit-тесты (на моках, быстро, запуск при каждой правке)
pytest tests/unit

# 2. Интеграционные тесты (на реальных моделях)
# Требует активных API-ключей в .env
pytest tests/integration --integration
```

Интеграционные тесты автоматически пропускаются, если не передан флаг `--integration` или если отсутствуют необходимые API-ключи.

## Статус поддержки провайдеров

| Провайдер | Базовая генерация | Tool Calling | Протестировано |
|---|---|---|---|
| OpenAI / Compatible | ✅ | ✅ | ✅ Проверено на реальных моделях |
| Anthropic | ✅ | ✅ | ✅ Проверено на реальных моделях |
| Sber GigaChat | ✅ | ⚠️ | ❌ Не тестировалось против официального провайдера |

> [!WARNING]
> **GigaChat — экспериментальная поддержка.** Адаптер реализован по документации GigaChat SDK, но не прошёл полного тестирования против официального API Сбера (`gigachat.devices.sberdevices.ru`).
>
> **Известные ограничения при использовании через vLLM-прокси:** Tool Calling не работает без флагов `--enable-auto-tool-choice` и `--tool-call-parser` на стороне сервера. Это ограничение инфраструктуры, а не библиотеки.

## Философия дизайна

- **Shared Priority Queue**: Единая очередь обеспечивающая приоритезацию.
- **Изоляция сбоев**: Падение воркера не вешает всю систему.
- **Backpressure**: Защита провайдера от перегрузки.
