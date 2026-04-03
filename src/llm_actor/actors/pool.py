import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any, TypeVar, overload
from uuid import uuid4

from opentelemetry.context import get_current

from llm_actor import tracing as otel_tracing
from llm_actor.actors.worker import ModelActor
from llm_actor.client.interface import LLMClientWithCircuitBreakerInterface
from llm_actor.core.messages import ActorMessage
from llm_actor.core.request import LLMRequest
from llm_actor.exceptions import ActorFailedError, OverloadError, PoolShuttingDownError
from llm_actor.logger import ActorLogger
from llm_actor.metrics import MetricsCollector
from llm_actor.settings import LLMActorSettings

T = TypeVar("T", bound=object)


@dataclass(eq=False)
class _PrioritizedMessage:
    priority: int
    sequence: int
    message: ActorMessage[Any]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _PrioritizedMessage):
            return NotImplemented
        return (self.priority, self.sequence) < (other.priority, other.sequence)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, _PrioritizedMessage):
            return NotImplemented
        return (self.priority, self.sequence) <= (other.priority, other.sequence)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, _PrioritizedMessage):
            return NotImplemented
        return (self.priority, self.sequence) > (other.priority, other.sequence)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, _PrioritizedMessage):
            return NotImplemented
        return (self.priority, self.sequence) >= (other.priority, other.sequence)


@dataclass
class HealthStatus:
    status: str
    alive_actors: int
    total_actors: int
    reason: str = ""


class SupervisedActorPool:
    def __init__(
        self,
        client: LLMClientWithCircuitBreakerInterface,
        settings: LLMActorSettings,
        metrics: MetricsCollector | None = None,
        pool_id: str | None = None,
    ) -> None:
        self._client: LLMClientWithCircuitBreakerInterface = client
        self._settings = settings
        self._metrics = metrics
        self._num_actors = settings.LLM_NUM_ACTORS
        self._max_restarts = settings.LLM_MAX_RESTARTS
        self._restart_window = settings.LLM_RESTART_WINDOW
        self._pool_id = pool_id or str(uuid4())

        self._actors: list[ModelActor] = []
        self._actor_tasks: list[asyncio.Task[None]] = []
        self._restart_counts: list[list[float]] = []
        self._supervisor_task: asyncio.Task[None] | None = None
        self._running = False
        self._shared_queue: asyncio.PriorityQueue[_PrioritizedMessage] | None = None
        self._sequence_counter: int = 0
        self._logger = ActorLogger.bind_context(pool_id=self._pool_id)

    @property
    def pool_id(self) -> str:
        """Pool identifier for metrics and logging"""
        return self._pool_id

    async def start(self) -> None:
        """Start pool with supervision"""
        if self._running:
            return
        self._running = True
        self._logger.info(f"Starting pool with {self._num_actors} actors")

        self._shared_queue = asyncio.PriorityQueue(maxsize=self._settings.LLM_MAX_QUEUE_SIZE)

        for i in range(self._num_actors):
            actor = ModelActor(
                client=self._client,
                actor_id=f"actor-{i}",
                settings=self._settings,
                shared_queue=self._shared_queue,
                metrics=self._metrics,
                pool_id=self._pool_id,
            )
            await actor.start()
            task = actor.task
            if task is None:
                raise RuntimeError(f"Actor {actor.actor_id} failed to start: task was not created")
            self._actors.append(actor)
            self._actor_tasks.append(task)
            self._restart_counts.append([])

        self._supervisor_task = asyncio.create_task(self._supervise())
        self._logger.info(f"Pool started successfully with {self._num_actors} actors")

    async def _supervise(self) -> None:
        """Supervisor loop - watches actor health"""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                await self._check_actor_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.critical(
                    f"Supervisor loop failed unexpectedly! Pool is now unmonitored. Error: {e}",
                    exc_info=True,
                )
                # P-5: CancelledError during recovery sleep must still end the supervisor.
                try:
                    await asyncio.sleep(5.0)
                except asyncio.CancelledError:
                    break

    async def _check_actor_tasks(self) -> None:
        for actor_index, (actor, task) in enumerate(
            zip(self._actors, self._actor_tasks, strict=True)
        ):
            if not task.done():
                continue

            pending_messages = self._extract_pending_from_task_exception(actor, task)
            if self._should_restart(actor_index):
                try:
                    await self._restart_actor(actor_index)
                    if pending_messages:
                        await self._requeue_pending_messages(pending_messages)
                except Exception as exc:
                    log = self._logger.bind(actor_id=actor.actor_id)
                    log.error(
                        f"Failed to restart actor, failing {len(pending_messages)} pending messages: {exc}",
                        exc_info=True,
                    )
                    self._fail_pending_messages(pending_messages, exc)
                continue

            log = self._logger.bind(actor_id=actor.actor_id)
            log.critical(
                f"Actor exceeded restart limit ({self._max_restarts} in {self._restart_window}s), "
                "not restarting"
            )
            self._fail_pending_messages(
                pending_messages,
                RuntimeError(
                    f"Actor {actor.actor_id} exceeded restart limit, cannot recover messages"
                ),
            )

    def _extract_pending_from_task_exception(
        self,
        actor: ModelActor,
        task: asyncio.Task[None],
    ) -> list[ActorMessage[Any]]:
        try:
            exception = task.exception()
        except asyncio.CancelledError:
            exception = None

        log = self._logger.bind(actor_id=actor.actor_id)
        if exception is None:
            log.warning("Actor task completed unexpectedly")
            return []

        if isinstance(exception, ActorFailedError):
            log.error(f"Actor failed and requested restart: {exception}", exc_info=exception)
            return list(exception.pending_messages)

        log.error(f"Actor crashed: {exception}", exc_info=exception)
        return []

    def _should_restart(self, actor_index: int) -> bool:
        """Check if actor should be restarted based on restart policy.

        The restart counter increments only after a successful restart in _restart_actor,
        not here, so failed attempts do not consume the budget.
        """
        now = time.time()
        restarts = self._restart_counts[actor_index]
        restarts[:] = [ts for ts in restarts if now - ts < self._restart_window]
        return len(restarts) < self._max_restarts

    async def _restart_actor(self, actor_index: int) -> None:
        """Restart specific actor"""
        if self._shared_queue is None:
            raise RuntimeError("Pool shared queue is not initialized")
        old_actor = self._actors[actor_index]
        log = self._logger.bind(actor_id=old_actor.actor_id, actor_index=actor_index)
        log.info(
            f"Restarting actor (attempt {len(self._restart_counts[actor_index]) + 1}/{self._max_restarts})"
        )

        try:
            await asyncio.wait_for(old_actor.stop(), timeout=5.0)
        except TimeoutError:
            log.warning("Actor stop timeout during restart")

        restart_num = len(self._restart_counts[actor_index]) + 1
        new_actor = ModelActor(
            client=self._client,
            actor_id=f"actor-{actor_index}-restart-{restart_num}",
            settings=self._settings,
            shared_queue=self._shared_queue,
            metrics=self._metrics,
            pool_id=self._pool_id,
        )
        await new_actor.start()

        if new_actor.task is None:
            await new_actor.stop()
            raise RuntimeError(
                f"Actor {new_actor.actor_id} failed to start: task was not created after start()"
            )

        self._actors[actor_index] = new_actor
        self._actor_tasks[actor_index] = new_actor.task

        # P-12: bump restart count only after a successful restart.
        self._restart_counts[actor_index].append(time.time())

        if self._metrics:
            self._metrics.actor_restarts_counter.labels(
                actor_id=new_actor.actor_id, pool_id=self.pool_id
            ).inc()

        log = self._logger.bind(actor_id=new_actor.actor_id, actor_index=actor_index)
        log.info("Actor restarted successfully")

    async def _requeue_pending_messages(self, pending_messages: list[ActorMessage[Any]]) -> None:
        tracer = otel_tracing.get_tracer()
        for message in pending_messages:
            # Clear otel_context: root generate span is closed; the new wait span must
            # attach to the current (supervisor) context, not a finished parent span.
            message.otel_context = None
            message.queue_wait_span_closer = None
            parent_ctx = get_current()
            wait_span = tracer.start_span(
                "llm_pool.wait",
                context=parent_ctx,
                attributes={"llm_actor.priority": message.priority},
            )
            message.queue_wait_span_closer = wait_span.end
            try:
                await self._put_in_queue(message)
            except Exception as exc:
                if message.future and not message.future.done():
                    message.future.set_exception(exc)
                else:
                    self._logger.error(
                        f"Failed to requeue message {message.id} and cannot signal future: {exc}",
                        exc_info=True,
                    )

    def _fail_pending_messages(
        self,
        pending_messages: list[ActorMessage[Any]],
        error: Exception,
    ) -> None:
        for message in pending_messages:
            if message.future and not message.future.done():
                message.future.set_exception(error)

    async def _put_in_queue(self, msg: ActorMessage[Any]) -> None:
        if self._shared_queue is None:
            raise RuntimeError("Pool has not been started; call start() before send()")
        if msg.enqueue_sequence is None:
            msg.enqueue_sequence = self._sequence_counter
            self._sequence_counter += 1
        item = _PrioritizedMessage(
            priority=msg.priority,
            sequence=msg.enqueue_sequence,
            message=msg,
        )
        try:
            await asyncio.wait_for(self._shared_queue.put(item), timeout=1.0)
        except TimeoutError as err:
            raise OverloadError("Shared queue is full") from err

    async def send(self, msg: ActorMessage[Any]) -> None:
        if not self._running:
            raise PoolShuttingDownError("Pool is shutting down, cannot accept new requests")
        if self._shared_queue is None:
            raise RuntimeError("Pool has not been started; call start() before send()")
        tracer = otel_tracing.get_tracer()
        if msg.queue_wait_span_closer is None:
            parent_ctx = (
                otel_tracing.extract_context(msg.otel_context)
                if msg.otel_context
                else get_current()
            )
            wait_span = tracer.start_span(
                "llm_pool.wait",
                context=parent_ctx,
                attributes={"llm_actor.priority": msg.priority},
            )
            msg.queue_wait_span_closer = wait_span.end
        await self._put_in_queue(msg)

    @overload
    async def generate(
        self,
        request: LLMRequest,
        response_model: None = None,
        *,
        priority: int = 10,
    ) -> str: ...

    @overload
    async def generate(
        self,
        request: LLMRequest,
        response_model: type[T],
        *,
        priority: int = 10,
    ) -> T: ...

    async def generate(
        self,
        request: LLMRequest,
        response_model: type[Any] | None = None,
        *,
        priority: int = 10,
    ) -> Any | str:
        """
        High-level interface for sending request to pool.
        If response_model is provided, returns validated Pydantic model instance.
        Otherwise returns string response.
        """
        if not self._running:
            raise PoolShuttingDownError("Pool is shutting down, cannot accept new requests")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        msg = ActorMessage(
            request=request,
            response_model=response_model,
            future=future,
            priority=priority,
        )
        tracer = otel_tracing.get_tracer()
        msg.otel_context = otel_tracing.inject_context()
        wait_span = tracer.start_span(
            "llm_pool.wait",
            attributes={"llm_actor.priority": priority},
        )
        msg.queue_wait_span_closer = wait_span.end
        self._logger.bind(request_id=msg.id).debug(
            f"Received generate request (response_model={'provided' if response_model else 'None'})"
        )

        # P-2: if send() raises (retries exhausted), the future would stay pending —
        # complete it with the same exception so await future does not hang.
        try:
            await self.send(msg)
        except Exception as exc:
            # End wait_span: the message never reaches an actor.
            wait_span.end()
            msg.queue_wait_span_closer = None
            if not future.done():
                future.set_exception(exc)
                # Suppress asyncio "Future exception was never retrieved" warning —
                # the caller receives the exception via `raise` below, not via `await future`.
                future.exception()
            raise

        return await future

    def get_health_status(self) -> HealthStatus:
        """Get current health status of the pool"""
        alive_count = sum(1 for actor in self._actors if actor.is_alive)

        total = len(self._actors)
        alive_ratio = alive_count / total if total > 0 else 0.0

        if alive_ratio == 1.0:
            status = "healthy"
            reason = "All actors operational"
        elif alive_ratio >= 0.5:
            status = "degraded"
            reason = f"{total - alive_count} actors down"
        else:
            status = "critical"
            reason = f"Only {alive_count}/{total} actors operational"

        return HealthStatus(
            status=status, alive_actors=alive_count, total_actors=total, reason=reason
        )

    def get_metrics(self) -> MetricsCollector | None:
        """Get metrics collector instance"""
        return self._metrics

    async def _wait_for_empty_inboxes(self) -> None:
        """Wait until shared queue and actor pending buffers are empty or all actors are dead."""
        while True:
            queue_size = self._shared_queue.qsize() if self._shared_queue else 0
            pending_size = sum(len(actor.pending) for actor in self._actors)
            all_dead = all(task.done() for task in self._actor_tasks)
            if queue_size + pending_size == 0 or all_dead:
                break
            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop pool and supervisor with graceful shutdown"""
        self._running = False
        self._logger.info("Stopping pool: blocking new requests")

        if self._supervisor_task:
            try:
                await asyncio.wait_for(self._supervisor_task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                self._supervisor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._supervisor_task

        self._logger.info("Waiting for actors to drain queues")
        try:
            await asyncio.wait_for(
                self._wait_for_empty_inboxes(),
                timeout=self._settings.LLM_GRACEFUL_SHUTDOWN_TIMEOUT,
            )
        except TimeoutError:
            queue_sz = self._shared_queue.qsize() if self._shared_queue else 0
            total_lost = queue_sz + sum(len(actor.pending) for actor in self._actors)
            self._logger.warning(
                f"Shutdown timeout after {self._settings.LLM_GRACEFUL_SHUTDOWN_TIMEOUT}s, "
                f"~{total_lost} messages may be lost"
            )

        await asyncio.gather(*[actor.stop() for actor in self._actors], return_exceptions=True)
        self._logger.info("Pool stopped successfully")
