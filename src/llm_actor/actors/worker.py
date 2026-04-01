import asyncio
import time
from typing import Any

from llm_actor.client.interface import LLMClientWithCircuitBreakerInterface
from llm_actor.core.messages import ActorMessage
from llm_actor.exceptions import ActorFailedError, CircuitBreakerOpenError, OverloadError
from llm_actor.logger import BrokerLogger
from llm_actor.metrics import MetricsCollector
from llm_actor.settings import LLMBrokerSettings

# Таймаут периодического пробуждения в idle-режиме для проверки _running.
_IDLE_POLL_TIMEOUT = 1.0


class ModelActor:
    """Actor that fails fast and relies on CircuitBreaker + Supervisor for recovery"""

    def __init__(
        self,
        client: LLMClientWithCircuitBreakerInterface,
        actor_id: str,
        settings: LLMBrokerSettings,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._client: LLMClientWithCircuitBreakerInterface = client
        self._actor_id = actor_id
        self._settings = settings
        self._metrics = metrics
        self._inbox: asyncio.Queue[ActorMessage[Any] | None] = asyncio.Queue(
            maxsize=settings.LLM_MAX_QUEUE_SIZE
        )
        self._batch_size = settings.LLM_BATCH_SIZE
        self._batch_timeout = settings.LLM_BATCH_TIMEOUT
        self._semaphore = asyncio.Semaphore(settings.LLM_MAX_CONCURRENT)
        self._max_consecutive_failures = settings.LLM_MAX_CONSECUTIVE_FAILURES

        self._pending: list[ActorMessage[Any]] = []
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._consecutive_failures = 0
        self._logger = BrokerLogger.bind_context(actor_id=self._actor_id)

    @property
    def actor_id(self) -> str:
        """Actor identifier for metrics and logging"""
        return self._actor_id

    @property
    def inbox(self) -> asyncio.Queue[ActorMessage[Any] | None]:
        """Actor message inbox queue"""
        return self._inbox

    @property
    def pending(self) -> list[ActorMessage[Any]]:
        """Pending messages batch"""
        return self._pending

    @property
    def is_alive(self) -> bool:
        """True если актор запущен и его задача активна."""
        return self._running and self._task is not None and not self._task.done()

    async def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._run())
            self._logger.info("Actor started")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            await self._inbox.put(None)
            await self._task
            self._logger.info("Actor stopped")

    async def send(self, msg: ActorMessage[Any]) -> None:
        try:
            await asyncio.wait_for(self._inbox.put(msg), timeout=1.0)
        except TimeoutError as err:
            # P-7: asyncio.TimeoutError наследует TimeoutError в Python 3.11+,
            # но на более ранних версиях они могут различаться — ловим оба явно.
            raise OverloadError(f"{self._actor_id} mailbox full") from err

    async def _run(self) -> None:
        """Main actor loop with self-healing"""
        self._logger.info("Main loop started")
        last_metrics_update = 0.0

        while True:
            if not self._running:
                self._logger.warning(
                    f"Stopped due to consecutive failures ({self._consecutive_failures}/{self._max_consecutive_failures})"
                )
                break

            now = time.time()
            if self._metrics and (now - last_metrics_update) >= 1.0:
                self._metrics.inbox_size_gauge.labels(actor_id=self._actor_id).set(
                    self._inbox.qsize()
                )
                last_metrics_update = now

            # P-3: при отсутствии pending используем периодический таймаут вместо None,
            # чтобы цикл мог проверить _running и не блокировать вечно при idle.
            timeout = self._batch_timeout if self._pending else _IDLE_POLL_TIMEOUT

            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
            except TimeoutError:
                if self._pending:
                    self._logger.debug(f"Batch timeout, processing {len(self._pending)} messages")
                    await self._safe_process_batch()
                continue

            if msg is None:
                if self._pending:
                    self._logger.debug(f"Stop signal received, processing {len(self._pending)} pending messages")
                    await self._safe_process_batch()
                self._inbox.task_done()
                break

            self._pending.append(msg)
            self._inbox.task_done()

            if len(self._pending) >= self._batch_size:
                self._logger.debug(f"Batch full ({len(self._pending)}/{self._batch_size}), processing immediately")
                await self._safe_process_batch()

        self._logger.info("Main loop finished")

    async def _safe_process_batch(self) -> None:
        """Process batch with fail-fast error handling"""
        batch = self._pending.copy()
        self._pending.clear()
        batch_size = len(batch)

        start_time = time.time()
        try:
            self._logger.debug(f"Processing batch of {batch_size} messages")
            await self._process_batch(batch)
            self._consecutive_failures = 0
            duration = time.time() - start_time
            self._logger.info(
                f"Batch processed successfully: {batch_size} messages in {duration:.3f}s"
            )
            if self._metrics:
                self._metrics.batches_processed_counter.labels(
                    actor_id=self._actor_id
                ).inc()
                self._metrics.batch_processing_duration_histogram.labels(
                    actor_id=self._actor_id
                ).observe(duration)
        except CircuitBreakerOpenError as e:
            duration = time.time() - start_time
            if self._metrics:
                self._metrics.batches_failed_counter.labels(actor_id=self._actor_id).inc()
                self._metrics.batch_processing_duration_histogram.labels(
                    actor_id=self._actor_id
                ).observe(duration)
            self._logger.warning(f"Circuit breaker open, rejecting batch of {batch_size} messages: {e}")
            self._reject_batch(batch, e)
        except Exception as e:
            duration = time.time() - start_time
            if self._metrics:
                self._metrics.batches_failed_counter.labels(actor_id=self._actor_id).inc()
                self._metrics.batch_processing_duration_histogram.labels(
                    actor_id=self._actor_id
                ).observe(duration)
            self._consecutive_failures += 1
            self._logger.error(
                f"Batch processing failed ({self._consecutive_failures}/{self._max_consecutive_failures}): "
                f"{batch_size} messages rejected after {duration:.3f}s: {e}",
                exc_info=True,
            )
            if self._consecutive_failures >= self._max_consecutive_failures:
                self._logger.critical(
                    f"Exceeded max consecutive failures ({self._max_consecutive_failures}), "
                    "raising ActorFailedError for supervisor restart."
                )
                failed_messages = [*batch, *self._pending]
                self._pending.clear()
                raise ActorFailedError(
                    message=(
                        f"Actor {self._actor_id} failed after "
                        f"{self._consecutive_failures} consecutive errors"
                    ),
                    actor_id=self._actor_id,
                    pending_messages=failed_messages,
                ) from e

            self._reject_batch(batch, e)

    async def _process_batch(self, batch: list[ActorMessage[Any]]) -> None:
        """Process batch of messages"""

        async def limited_ask(msg: ActorMessage[Any]) -> Any:
            async with self._semaphore:
                log = self._logger.bind(request_id=msg.id)
                log.debug("Processing request")
                try:
                    result = await self._client.generate(msg.prompt, msg.response_model)
                    log.debug("Request processed successfully")
                    return result
                except Exception as e:
                    log.error("Request processing failed: {}", e, exc_info=True)
                    raise

        results = await asyncio.gather(
            *[limited_ask(msg) for msg in batch], return_exceptions=True
        )

        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = len(results) - success_count

        if error_count > 0:
            self._logger.warning(
                f"Batch completed with {error_count} errors out of {len(batch)} requests"
            )

        for msg, result in zip(batch, results, strict=True):
            if msg.future and not msg.future.done():
                if isinstance(result, Exception):
                    msg.future.set_exception(result)
                else:
                    msg.future.set_result(result)

    def _reject_batch(self, batch: list[ActorMessage[Any]], error: Exception) -> None:
        """Reject all messages in batch with error"""
        for msg in batch:
            if msg.future and not msg.future.done():
                msg.future.set_exception(error)
