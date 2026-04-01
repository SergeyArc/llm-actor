import asyncio
import time
from dataclasses import dataclass
from typing import Any, TypeVar, overload
from uuid import uuid4

from llm_broker.actors.worker import ModelActor
from llm_broker.client.interface import LLMClientWithCircuitBreakerInterface
from llm_broker.core.messages import ActorMessage
from llm_broker.exceptions import PoolShuttingDownError
from llm_broker.logger import BrokerLogger
from llm_broker.metrics import MetricsCollector
from llm_broker.settings import LLMBrokerSettings

T = TypeVar("T", bound=object)


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
        settings: LLMBrokerSettings,
        metrics: MetricsCollector | None = None,
        restart_strategy: str = "one-for-one",
        pool_id: str | None = None,
    ) -> None:
        self._client: LLMClientWithCircuitBreakerInterface = client
        self._settings = settings
        self._metrics = metrics
        self._num_actors = settings.LLM_NUM_ACTORS
        self._restart_strategy = restart_strategy
        self._max_restarts = settings.LLM_MAX_RESTARTS
        self._restart_window = settings.LLM_RESTART_WINDOW
        self._pool_id = pool_id or str(uuid4())

        self._actors: list[ModelActor] = []
        self._actor_tasks: list[asyncio.Task[None]] = []
        self._restart_counts: list[list[float]] = []
        self._supervisor_task: asyncio.Task[None] | None = None
        self._running = False
        self._current_actor_index: int = 0
        self._logger = BrokerLogger.bind_context(pool_id=self._pool_id)

    @property
    def pool_id(self) -> str:
        """Pool identifier for metrics and logging"""
        return self._pool_id

    async def start(self) -> None:
        """Start pool with supervision"""
        self._running = True
        self._logger.info(f"Starting pool with {self._num_actors} actors")

        for i in range(self._num_actors):
            actor = ModelActor(
                client=self._client,
                actor_id=f"actor-{i}",
                settings=self._settings,
                metrics=self._metrics,
            )
            await actor.start()
            self._actors.append(actor)
            if actor._task is not None:
                self._actor_tasks.append(actor._task)
            self._restart_counts.append([])

        self._supervisor_task = asyncio.create_task(self._supervise())
        self._logger.info(f"Pool started successfully with {self._num_actors} actors")

    async def _supervise(self) -> None:
        """Supervisor loop - watches actor health"""
        while self._running:
            try:
                await asyncio.sleep(1.0)

                for i, (actor, task) in enumerate(
                    zip(self._actors, self._actor_tasks, strict=True)
                ):
                    if task.done():
                        try:
                            exception = task.exception()
                        except asyncio.CancelledError:
                            exception = None

                        log = self._logger.bind(actor_id=actor.actor_id)
                        if exception:
                            log.error(f"Actor crashed: {exception}", exc_info=exception)
                        else:
                            log.warning("Actor task completed unexpectedly")

                        if self._should_restart(i):
                            await self._restart_actor(i)
                        else:
                            log.critical(
                                f"Actor exceeded restart limit ({self._max_restarts} in {self._restart_window}s), "
                                f"not restarting"
                            )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.critical(
                    f"Supervisor loop failed unexpectedly! Pool is now unmonitored. Error: {e}",
                    exc_info=True,
                )
                # P-5: CancelledError во время recovery-sleep тоже должен завершить supervisor.
                try:
                    await asyncio.sleep(5.0)
                except asyncio.CancelledError:
                    break

    def _should_restart(self, actor_index: int) -> bool:
        """Check if actor should be restarted based on restart policy.

        Счётчик restart увеличивается только после успешного рестарта в _restart_actor,
        а не здесь — чтобы неудачные попытки не расходовали бюджет.
        """
        now = time.time()
        restarts = self._restart_counts[actor_index]
        restarts[:] = [ts for ts in restarts if now - ts < self._restart_window]
        return len(restarts) < self._max_restarts

    async def _restart_actor(self, actor_index: int) -> None:
        """Restart specific actor"""
        old_actor = self._actors[actor_index]
        log = self._logger.bind(actor_id=old_actor.actor_id, actor_index=actor_index)
        log.info(
            f"Restarting actor (attempt {len(self._restart_counts[actor_index]) + 1}/{self._max_restarts})"
        )

        try:
            await asyncio.wait_for(old_actor.stop(), timeout=5.0)
        except TimeoutError:
            log.warning("Actor stop timeout during restart")

        new_actor = ModelActor(
            client=self._client,
            actor_id=f"actor-{actor_index}-restarted",
            settings=self._settings,
            metrics=self._metrics,
        )
        await new_actor.start()

        # P-12: проверяем, что задача действительно создана, прежде чем принять рестарт.
        if new_actor._task is None:
            log.error("Actor failed to start (no task created); aborting restart")
            return

        self._actors[actor_index] = new_actor
        self._actor_tasks[actor_index] = new_actor._task

        # P-12: счётчик рестартов увеличиваем только после успеха.
        self._restart_counts[actor_index].append(time.time())

        if self._metrics:
            self._metrics.actor_restarts_counter.labels(
                actor_id=new_actor.actor_id, pool_id=self.pool_id
            ).inc()

        log = self._logger.bind(actor_id=new_actor.actor_id, actor_index=actor_index)
        log.info("Actor restarted successfully")

    def _select_next_actor(self) -> ModelActor | None:
        """Select next actor using round-robin strategy, skipping dead actors"""
        if not self._actors:
            return None

        if not any(actor.is_alive for actor in self._actors):
            return None

        start_index = self._current_actor_index % len(self._actors)
        for i in range(len(self._actors)):
            idx = (start_index + i) % len(self._actors)
            actor = self._actors[idx]
            if actor.is_alive:
                self._current_actor_index = (idx + 1) % len(self._actors)
                return actor

        return None

    async def send(self, msg: ActorMessage[Any], retries: int = 2) -> None:
        """Send message to actor using round-robin routing with retry"""
        if not self._running:
            raise PoolShuttingDownError("Pool is shutting down, cannot accept new requests")
        for attempt in range(retries):
            actor = self._select_next_actor()
            if actor is None:
                raise RuntimeError("No alive actors available in pool")
            try:
                await actor.send(msg)
                return
            except Exception as e:
                self._logger.bind(actor_id=actor.actor_id, request_id=msg.id).warning(
                    f"Failed to send message to actor on attempt {attempt + 1}/{retries}: {e}"
                )
                if attempt == retries - 1:
                    raise

    @overload
    async def generate(
        self,
        prompt: str,
        response_model: None = None,
    ) -> str: ...

    @overload
    async def generate(
        self,
        prompt: str,
        response_model: type[T],
    ) -> T: ...

    async def generate(
        self,
        prompt: str,
        response_model: type[Any] | None = None,
    ) -> Any | str:
        """
        High-level interface for sending prompt to pool.
        If response_model is provided, returns validated Pydantic model instance.
        Otherwise returns string response.
        """
        if not self._running:
            raise PoolShuttingDownError("Pool is shutting down, cannot accept new requests")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        msg = ActorMessage(prompt=prompt, response_model=response_model, future=future)
        self._logger.bind(request_id=msg.id).debug(
            f"Received generate request (response_model={'provided' if response_model else 'None'})"
        )

        # P-2: если send() бросит исключение (все retry исчерпаны), future останется
        # неразрешённым — завершаем его с тем же исключением, чтобы await future не завис.
        try:
            await self.send(msg)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
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
        """Wait until all actor inboxes and pending batches are empty or all actors are dead."""
        while True:
            total_pending = 0
            all_dead = True

            for actor, task in zip(self._actors, self._actor_tasks, strict=True):
                if not task.done():
                    all_dead = False
                    # P-4: учитываем как inbox, так и pending-буфер актора,
                    # потому что сообщения могут быть уже извлечены из inbox,
                    # но ещё не обработаны (находятся в _pending).
                    total_pending += actor.inbox.qsize() + len(actor.pending)

            if total_pending == 0 or all_dead:
                break

            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop pool and supervisor with graceful shutdown"""
        self._running = False
        self._logger.info("Stopping pool: blocking new requests")

        if self._supervisor_task:
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except asyncio.CancelledError:
                pass

        self._logger.info("Waiting for actors to drain queues")
        try:
            await asyncio.wait_for(
                self._wait_for_empty_inboxes(),
                timeout=self._settings.LLM_GRACEFUL_SHUTDOWN_TIMEOUT,
            )
        except TimeoutError:
            total_lost = sum(actor.inbox.qsize() + len(actor.pending) for actor in self._actors)
            self._logger.warning(
                f"Shutdown timeout after {self._settings.LLM_GRACEFUL_SHUTDOWN_TIMEOUT}s, "
                f"~{total_lost} messages may be lost"
            )

        await asyncio.gather(*[actor.stop() for actor in self._actors], return_exceptions=True)
        self._logger.info("Pool stopped successfully")
