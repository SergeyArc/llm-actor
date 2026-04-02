import inspect
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

_PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _resolve_json_type(python_type: Any) -> str:
    """Map a Python type annotation to a JSON Schema type string.

    Handles Optional[T] / Union[T, None] / T | None by extracting the non-None arg.
    """
    origin = getattr(python_type, "__origin__", None)
    if origin is typing.Union:
        non_none = [a for a in python_type.__args__ if a is not type(None)]
        if non_none:
            return _PYTHON_TO_JSON_TYPE.get(non_none[0], "string")
        return "string"
    if isinstance(python_type, types.UnionType):
        non_none = [a for a in python_type.__args__ if a is not type(None)]
        if non_none:
            return _PYTHON_TO_JSON_TYPE.get(non_none[0], "string")
        return "string"
    return _PYTHON_TO_JSON_TYPE.get(python_type, "string")


@dataclass
class Tool:
    func: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    schema_override: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            raw = self.func.__name__
            if raw == "<lambda>":
                raise ValueError(
                    "Lambda functions must be given an explicit name: Tool(func=..., name='my_tool')"
                )
            self.name = raw
        if self.description is None:
            self.description = inspect.getdoc(self.func) or ""

    def build_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
                "parameters": self.schema_override
                if self.schema_override is not None
                else self._infer_schema(),
            },
        }

    def build_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description or "",
            "input_schema": self.schema_override
            if self.schema_override is not None
            else self._infer_schema(),
        }

    def _infer_schema(self) -> dict[str, Any]:
        sig = inspect.signature(self.func)
        try:
            hints = get_type_hints(self.func)
        except Exception:
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            python_type = hints.get(param_name)
            json_type = _resolve_json_type(python_type) if python_type else "string"
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    result: str
    is_error: bool = False


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)
