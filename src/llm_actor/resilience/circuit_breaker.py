import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from llm_actor.exceptions import CircuitBreakerOpenError
from llm_actor.logger import ActorLogger
from llm_actor.metrics import MetricsCollector
from llm_actor.settings import LLMActorSettings


class CircuitBreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for external API calls"""

    def __init__(
        self,
        settings: LLMActorSettings,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._failure_threshold = settings.LLM_FAILURE_THRESHOLD
        self._recovery_timeout = settings.LLM_RECOVERY_TIMEOUT
        self._metrics = metrics
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()
        self._logger = ActorLogger.get_logger(name="circuit_breaker")

    async def call(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._failure_count = 0
                    self._logger.info(
                        f"Circuit breaker attempting reset (HALF_OPEN) after {self._recovery_timeout}s"
                    )
                else:
                    if self._last_failure_time is None:
                        self._logger.warning("Circuit breaker OPEN (no recovery time set)")
                        raise CircuitBreakerOpenError("Circuit breaker OPEN")
                    retry_in = self._recovery_timeout - (time.time() - self._last_failure_time)
                    self._logger.warning(f"Circuit breaker OPEN, retry in {retry_in:.1f}s")
                    raise CircuitBreakerOpenError(f"Circuit breaker OPEN, retry in {retry_in:.1f}s")
            elif self._state == CircuitBreakerState.HALF_OPEN:
                self._logger.warning("Circuit breaker HALF_OPEN: probe already in progress")
                raise CircuitBreakerOpenError(
                    "Circuit breaker HALF_OPEN: another request is probing recovery"
                )

        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                self._on_success()
            return result
        except Exception:
            async with self._lock:
                self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        return (
            self._last_failure_time is not None
            and time.time() - self._last_failure_time >= self._recovery_timeout
        )

    def _on_success(self) -> None:
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._logger.info("Circuit breaker recovered (CLOSED)")
        elif self._failure_count > 0:
            self._logger.debug(
                f"Request succeeded, failure count reset (was {self._failure_count})"
            )
        self._failure_count = 0
        self._state = CircuitBreakerState.CLOSED

    def _on_failure(self) -> None:
        half_open_probe = self._state == CircuitBreakerState.HALF_OPEN
        self._failure_count += 1
        self._last_failure_time = time.time()

        should_open = half_open_probe or self._failure_count >= self._failure_threshold
        if should_open:
            if self._state != CircuitBreakerState.OPEN:
                self._state = CircuitBreakerState.OPEN
                if self._metrics:
                    self._metrics.circuit_breaker_trips_counter.inc()
                if half_open_probe:
                    self._logger.error("Circuit breaker OPEN: HALF_OPEN probe failed")
                else:
                    self._logger.error(
                        f"Circuit breaker OPEN after {self._failure_count} failures "
                        f"(threshold: {self._failure_threshold})"
                    )
            else:
                self._logger.debug(
                    f"Failure recorded ({self._failure_count}/{self._failure_threshold}), "
                    f"circuit breaker already OPEN"
                )
        else:
            self._logger.debug(
                f"Failure recorded ({self._failure_count}/{self._failure_threshold}), "
                f"state: {self._state}"
            )
