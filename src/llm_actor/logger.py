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


def _actor_log_record_patcher(record: Record) -> None:
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


class ActorLogger:
    """Central loguru-based logger for llm_actor."""

    _configured = False

    @classmethod
    def configure(cls) -> None:
        """
        Non-invasive setup: only registers the context tag patcher.
        Does not remove existing sinks or add new ones.
        """
        if cls._configured:
            return

        # Patch records with actor_tag, trace_tag, etc.
        # Safe for host apps: if the format string omits these fields, they remain in ``extra``.
        logger.configure(patcher=_actor_log_record_patcher)
        cls._configured = True

    @classmethod
    def setup_standard_logging(cls, level: str = "INFO") -> None:
        """
        Opinionated console logging for applications.

        WARNING: removes all existing loguru sinks.

        Call only from your app entrypoint if you want llm_actor-style console output.
        """
        logger.remove()
        cls.configure()

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

    @classmethod
    def get_logger(cls, name: str | None = None) -> Any:
        """
        Return a configured logger.

        Args:
            name: Optional module name binding.

        Returns:
            A loguru logger instance.
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
        Logger with bound pool/actor/request context.

        Args:
            pool_id: Actor pool identifier.
            actor_id: Actor identifier.
            request_id: Request identifier.

        Returns:
            Bound loguru logger.
        """
        if not cls._configured:
            cls.configure()

        context: dict[str, Any] = {}
        if pool_id:
            context["pool_id"] = pool_id
        if actor_id:
            context["actor_id"] = actor_id
        if request_id:
            context["request_id"] = request_id

        return logger.bind(**context)
