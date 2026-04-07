import asyncio
import contextlib
import time
from typing import Any, Literal

from llm_actor import tracing as otel_tracing
from llm_actor.client.interface import LLMClientWithCircuitBreakerInterface
from llm_actor.core.messages import ActorMessage
from llm_actor.exceptions import ActorFailedError, CircuitBreakerOpenError
from llm_actor.logger import ActorLogger
from llm_actor.metrics import MetricsCollector
from llm_actor.settings import LLMActorSettings

# Idle poll interval to re-check _running.
_IDLE_POLL_TIMEOUT = 1.0

_QueuePollOutcome = Literal["timeout", "stop", "message"]


class ModelActor:
    """Actor that fails fast and relies on CircuitBreaker + Supervisor for recovery"""

    def __init__(
        self,
        client: LLMClientWithCircuitBreakerInterface,
        actor_id: str,
        settings: LLMActorSettings,
        shared_queue: asyncio.PriorityQueue[Any],
        metrics: MetricsCollector | None = None,
        pool_id: str = "default",
    ) -> None:
        self._client: LLMClientWithCircuitBreakerInterface = client
        self._actor_id = actor_id
        self._settings = settings
        self._metrics = metrics
        self._shared_queue = shared_queue
        self._pool_id = pool_id
        self._stop_event = asyncio.Event()
        self._batch_size = settings.LLM_BATCH_SIZE
        self._batch_timeout = settings.LLM_BATCH_TIMEOUT
        self._semaphore = asyncio.Semaphore(settings.LLM_MAX_CONCURRENT)
        self._max_consecutive_failures = settings.LLM_MAX_CONSECUTIVE_FAILURES

        self._pending: list[ActorMessage[Any]] = []
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._consecutive_failures = 0
        self._logger = ActorLogger.bind_context(actor_id=self._actor_id)

    @property
    def actor_id(self) -> str:
        """Actor identifier for metrics and logging"""
        return self._actor_id

    @property
    def pending(self) -> list[ActorMessage[Any]]:
        """Pending messages batch"""
        return self._pending

    @property
    def is_alive(self) -> bool:
        """True when the actor is running and its task is still active."""
        return self._running and self._task is not None and not self._task.done()

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    async def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._run())
            self._logger.info("Actor started")

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except TimeoutError:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
        self._logger.info("Actor stopped")

    async def _abort_pending_queue_get(self, get_fut: asyncio.Future[Any]) -> None:
        if get_fut.done():
            return
        get_fut.cancel()
        try:
            await get_fut
        except (asyncio.CancelledError, Exception):
            pass

    def _append_item_if_get_succeeded(
        self, done: set[asyncio.Future[Any]], get_fut: asyncio.Future[Any]
    ) -> bool:
        if get_fut not in done or get_fut.cancelled():
            return False
        if get_fut.exception() is not None:
            return False
        item = get_fut.result()
        self._shared_queue.task_done()
        message = item.message
        if message.queue_wait_span_closer is not None:
            message.queue_wait_span_closer()
            message.queue_wait_span_closer = None
        self._pending.append(message)
        return True

    async def _poll_shared_queue_or_stop(
        self, stop_fut: asyncio.Future[Any], timeout: float
    ) -> _QueuePollOutcome:
        get_fut = asyncio.ensure_future(self._shared_queue.get())
        done, _ = await asyncio.wait(
            {get_fut, stop_fut},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            await self._abort_pending_queue_get(get_fut)
            return "timeout"
        if stop_fut in done:
            if not self._append_item_if_get_succeeded(done, get_fut):
                await self._abort_pending_queue_get(get_fut)
            return "stop"
        self._append_item_if_get_succeeded(done, get_fut)
        return "message"

    async def _run(self) -> None:
        """Main actor loop with self-healing"""
        self._logger.info("Main loop started")
        last_metrics_update = 0.0
        stop_fut = asyncio.ensure_future(self._stop_event.wait())

        try:
            while True:
                if not self._running:
                    if self._stop_event.is_set():
                        self._logger.debug("Exiting main loop (shutdown)")
                    else:
                        self._logger.warning(
                            "Main loop stopped while running flag cleared "
                            f"({self._consecutive_failures}/{self._max_consecutive_failures} failures)"
                        )
                    break

                now = time.time()
                if self._metrics and (now - last_metrics_update) >= 1.0:
                    self._metrics.inbox_size_gauge.labels(pool_id=self._pool_id).set(
                        self._shared_queue.qsize()
                    )
                    last_metrics_update = now

                timeout = self._batch_timeout if self._pending else _IDLE_POLL_TIMEOUT
                outcome = await self._poll_shared_queue_or_stop(stop_fut, timeout)

                if outcome == "timeout":
                    if self._pending:
                        self._logger.debug(
                            f"Batch timeout, processing {len(self._pending)} messages"
                        )
                        await self._safe_process_batch()
                    continue
                if outcome == "stop":
                    if self._pending:
                        self._logger.debug(
                            f"Stop signal received, processing {len(self._pending)} pending messages"
                        )
                        await self._safe_process_batch()
                    break
                if len(self._pending) >= self._batch_size:
                    self._logger.debug(
                        f"Batch full ({len(self._pending)}/{self._batch_size}), processing immediately"
                    )
                    await self._safe_process_batch()
        finally:
            stop_fut.cancel()

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
                self._metrics.batches_processed_counter.labels(actor_id=self._actor_id).inc()
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
            self._logger.warning(
                f"Circuit breaker open, rejecting batch of {batch_size} messages: {e}"
            )
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
                failed_messages = batch
                raise ActorFailedError(
                    message=(
                        f"Actor {self._actor_id} failed after "
                        f"{self._consecutive_failures} consecutive errors"
                    ),
                    actor_id=self._actor_id,
                    pending_messages=failed_messages,
                ) from e

            self._reject_batch(batch, e)
        except asyncio.CancelledError:
            self._reject_batch(
                batch,
                RuntimeError(f"Actor {self._actor_id} was cancelled during batch processing"),
            )
            raise

    async def _process_batch(self, batch: list[ActorMessage[Any]]) -> None:
        """Process batch of messages"""

        async def limited_ask(msg: ActorMessage[Any]) -> Any:
            async with self._semaphore:
                log = self._logger.bind(request_id=msg.id)
                log.debug("Processing request")
                with otel_tracing.attach_extracted_context(msg.otel_context):
                    tracer = otel_tracing.get_tracer()
                    with tracer.start_as_current_span(
                        "llm_actor.actor_process",
                        attributes={
                            "llm_actor.request_id": msg.id or "",
                            "llm_actor.priority": msg.priority,
                        },
                    ):
                        try:
                            result = await self._client.generate(msg.request, msg.response_model)
                            log.debug("Request processed successfully")
                            return result
                        except Exception as e:
                            log.error("Request processing failed: {}", e, exc_info=True)
                            raise

        results = await asyncio.gather(
            *[
                asyncio.create_task(limited_ask(msg), context=msg.caller_context)
                if msg.caller_context is not None
                else limited_ask(msg)
                for msg in batch
            ],
            return_exceptions=True,
        )

        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = len(results) - success_count

        if error_count > 0:
            self._logger.warning(
                f"Batch completed with {error_count} errors out of {len(batch)} requests"
            )

        for msg, result in zip(batch, results, strict=True):
            if msg.future and not msg.future.done():
                if isinstance(result, asyncio.CancelledError):
                    msg.future.cancel()
                elif isinstance(result, BaseException):
                    msg.future.set_exception(result)
                else:
                    msg.future.set_result(result)

    def _reject_batch(self, batch: list[ActorMessage[Any]], error: Exception) -> None:
        """Reject all messages in batch with error"""
        for msg in batch:
            if msg.future and not msg.future.done():
                msg.future.set_exception(error)
