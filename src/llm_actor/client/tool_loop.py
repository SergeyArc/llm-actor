import asyncio
import inspect
from typing import Any, cast

from opentelemetry.trace import StatusCode

from llm_actor import tracing as otel_tracing
from llm_actor.client.interface import (
    LLMClientWithCircuitBreakerInterface,
    ToolCapableClientInterface,
)
from llm_actor.core.request import LLMRequest
from llm_actor.core.tools import Tool, ToolCall, ToolResult
from llm_actor.exceptions import (
    LLMServiceGeneralError,
    ToolExecutionTimeoutError,
    ToolLoopMaxIterationsError,
)
from llm_actor.logger import BrokerLogger
from llm_actor.settings import LLMBrokerSettings


class ToolCallOrchestratorClient:
    def __init__(
        self,
        cb_client: LLMClientWithCircuitBreakerInterface,
        settings: LLMBrokerSettings,
    ) -> None:
        self._cb_client = cb_client
        self._settings = settings
        self._logger = BrokerLogger.get_logger(name="tool_loop")

    async def generate(self, request: LLMRequest, response_model: type[Any] | None = None) -> Any:
        if request.tools:
            if response_model is not None:
                raise ValueError("Combining tools with response_model is not supported")
            if not isinstance(self._cb_client, ToolCapableClientInterface):
                raise LLMServiceGeneralError(
                    f"Client {type(self._cb_client).__name__} does not implement ToolCapableClientInterface"
                )
            return await self._run_tool_loop(request)
        return await self._cb_client.generate(request, response_model)

    async def _run_tool_loop(self, request: LLMRequest) -> str:
        resolved_tools = cast(list[Tool], request.tools or [])
        tool_map: dict[str, Tool] = {
            tool.name: tool for tool in resolved_tools if tool.name is not None
        }
        conversation: list[dict[str, Any]] = []
        tc_client = cast(ToolCapableClientInterface, self._cb_client)

        for iteration in range(1, self._settings.LLM_TOOL_MAX_ITERATIONS + 1):
            self._logger.debug(
                f"Tool loop iteration {iteration}/{self._settings.LLM_TOOL_MAX_ITERATIONS}"
            )

            response = await tc_client.generate_with_tools_async(request, conversation)

            if not response.has_tool_calls:
                if response.content is None:
                    raise LLMServiceGeneralError("Tool loop completed without content from LLM")
                self._logger.debug(f"Tool loop completed in {iteration} iteration(s)")
                return response.content

            self._logger.debug(f"LLM requested {len(response.tool_calls)} tool call(s)")
            if response.assistant_message is None:
                raise LLMServiceGeneralError(
                    "Client returned tool_calls without assistant_message — cannot build valid conversation"
                )
            conversation.append(response.assistant_message)

            tool_results = await self._execute_tool_calls(
                response.tool_calls, tool_map, request.tool_timeout
            )
            conversation.extend(tc_client.format_tool_results(tool_results))

        raise ToolLoopMaxIterationsError(self._settings.LLM_TOOL_MAX_ITERATIONS)

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        tool_map: dict[str, Tool],
        tool_timeout: float | None,
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in tool_calls:
            result = await self._execute_single(call, tool_map, tool_timeout)
            results.append(result)
        return results

    async def _execute_single(
        self,
        call: ToolCall,
        tool_map: dict[str, Tool],
        tool_timeout: float | None,
    ) -> ToolResult:
        tracer = otel_tracing.get_tracer()
        with tracer.start_as_current_span(
            "llm_actor.tool_call",
            attributes={"llm_actor.tool.name": call.name},
        ) as span:
            tool = tool_map.get(call.name)
            if tool is None:
                self._logger.warning(f"Tool '{call.name}' not found in tool_map")
                span.set_status(StatusCode.ERROR, f"Unknown tool: {call.name}")
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    result=f"Unknown tool: {call.name}",
                    is_error=True,
                )

            effective_timeout = (
                tool_timeout
                if tool_timeout is not None
                else self._settings.LLM_TOOL_EXECUTION_TIMEOUT
            )
            self._logger.debug(f"Executing tool '{call.name}' (timeout={effective_timeout}s)")

            try:
                if inspect.iscoroutinefunction(tool.func):
                    coro = tool.func(**call.arguments)
                else:
                    coro = asyncio.to_thread(tool.func, **call.arguments)

                raw_result = await asyncio.wait_for(coro, timeout=effective_timeout)
                self._logger.debug(f"Tool '{call.name}' executed successfully")
                return ToolResult(tool_call_id=call.id, name=call.name, result=str(raw_result))

            except TimeoutError as exc:
                # Исключение пробрасывается — SDK автоматически вызовет record_exception и set_status.
                raise ToolExecutionTimeoutError(call.name, effective_timeout) from exc
            except Exception as exc:
                # Исключение «глотается» (возвращаем ToolResult), поэтому записываем явно.
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                self._logger.warning(f"Tool '{call.name}' raised exception: {exc}")
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    result=str(exc),
                    is_error=True,
                )
