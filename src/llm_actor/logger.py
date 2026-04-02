from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

try:
    from opentelemetry import trace as _otel_trace
except ImportError:
    _otel_trace = None  # type: ignore[assignment]


def _broker_log_record_patcher(record: Record) -> None:
    extra = record["extra"]
    actor_id = extra.get("actor_id")
    pool_id = extra.get("pool_id")
    extra["actor_tag"] = f"[{actor_id}] " if actor_id else ""
    if pool_id:
        pool_prefix = pool_id[:8] if len(pool_id) > 8 else pool_id
        extra["pool_tag"] = f"[pool {pool_prefix}] "
    else:
        extra["pool_tag"] = ""
    if _otel_trace is not None:
        span = _otel_trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            tid = format(ctx.trace_id, "032x")
            extra["trace_tag"] = f"[trace={tid}] "
        else:
            extra["trace_tag"] = ""
    else:
        extra["trace_tag"] = ""


class BrokerLogger:
    """Централизованный логгер для пакета llm_actor на основе loguru."""

    _configured = False

    @classmethod
    def configure(
        cls,
        level: str = "INFO",
    ) -> None:
        """
        Убирает встроенный stderr-sink loguru (id 0) и добавляет единый формат для пакета.

        Снимает все существующие sink'и loguru и добавляет один stderr-sink с единым форматом.
        Иначе возможны дубли (дефолтный sink + наш и/или повторный add).

        Args:
            level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if cls._configured:
            return

        logger.remove()

        logger.configure(patcher=_broker_log_record_patcher)

        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "{extra[trace_tag]}{extra[actor_tag]}{extra[pool_tag]}"
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stderr,
            format=console_format,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        cls._configured = True

    @classmethod
    def get_logger(cls, name: str | None = None) -> Any:
        """
        Возвращает настроенный логгер.

        Args:
            name: Имя модуля для логирования (опционально)

        Returns:
            Экземпляр loguru logger
        """
        if not cls._configured:
            cls.configure()

        if name:
            return logger.bind(name=name)
        return logger

    @classmethod
    def bind_context(
        cls,
        pool_id: str | None = None,
        actor_id: str | None = None,
        request_id: str | None = None,
    ) -> Any:
        """
        Создает логгер с привязанным контекстом.

        Args:
            pool_id: ID пула акторов
            actor_id: ID актора
            request_id: ID запроса

        Returns:
            Логгер с привязанным контекстом
        """
        if not cls._configured:
            cls.configure()

        context = {}
        if pool_id:
            context["pool_id"] = pool_id
        if actor_id:
            context["actor_id"] = actor_id
        if request_id:
            context["request_id"] = request_id

        return logger.bind(**context)
