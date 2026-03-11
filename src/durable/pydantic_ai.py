"""Pydantic AI integration for python-durable.

Provides DurableAgent — a wrapper that makes any pydantic-ai Agent durable,
automatically checkpointing model requests and tool calls to the store.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Generic, TypeVar

from .backoff import BackoffStrategy, exponential
from .workflow import Workflow

log = logging.getLogger("durable.pydantic_ai")

AgentDepsT = TypeVar("AgentDepsT")
OutputT = TypeVar("OutputT")


def _serialize_messages(messages: list[Any]) -> list[dict]:
    """Convert pydantic-ai message objects to JSON-serializable dicts."""
    result = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            d = msg.model_dump(mode="json")
            d["__type__"] = type(msg).__name__
            result.append(d)
        elif isinstance(msg, dict):
            result.append(msg)
        else:
            result.append({"__repr__": repr(msg)})
    return result


def _deserialize_messages(data: list[dict]) -> list[Any]:
    """Reconstruct pydantic-ai message objects from serialized dicts.

    Falls back to returning raw dicts if the classes aren't available.
    """
    try:
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
        )

        _type_map: dict[str, type] = {
            "ModelRequest": ModelRequest,
            "ModelResponse": ModelResponse,
        }
    except ImportError:
        _type_map = {}

    result = []
    for item in data:
        type_name = item.pop("__type__", None) if isinstance(item, dict) else None
        if type_name and type_name in _type_map:
            try:
                result.append(_type_map[type_name].model_validate(item))
            except Exception:
                item["__type__"] = type_name
                result.append(item)
        else:
            if type_name:
                item["__type__"] = type_name
            result.append(item)
    return result


def _run_id_for_agent(agent_name: str, prompt: str, run_id: str | None) -> str:
    """Generate a deterministic run ID from the agent name and prompt."""
    if run_id:
        return run_id
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    return f"agent-{agent_name}-{prompt_hash}"


class TaskConfig:
    """Retry/backoff configuration for model requests or tool calls."""

    def __init__(
        self,
        retries: int | None = None,
        backoff: BackoffStrategy | None = None,
    ) -> None:
        self.retries = retries
        self.backoff = backoff


class DurableAgent(Generic[AgentDepsT, OutputT]):
    """Wrap a pydantic-ai Agent for durable execution with python-durable.

    Wraps each ``agent.run()`` call as a ``@wf.workflow`` and checkpoints
    model requests and tool calls as ``@wf.task`` steps.
    """

    def __init__(
        self,
        agent: Any,  # pydantic_ai.Agent[AgentDepsT, OutputT]
        wf: Workflow,
        *,
        name: str | None = None,
        model_task_config: TaskConfig | None = None,
        tool_task_config: TaskConfig | None = None,
    ) -> None:
        self.agent = agent
        self.wf = wf
        self.name = name or getattr(agent, "name", None) or "agent"

        self._model_retries = (
            model_task_config.retries
            if model_task_config and model_task_config.retries is not None
            else 3
        )
        self._model_backoff = (
            model_task_config.backoff
            if model_task_config and model_task_config.backoff
            else exponential(base=2, max=60)
        )
        self._tool_retries = (
            tool_task_config.retries
            if tool_task_config and tool_task_config.retries is not None
            else 2
        )
        self._tool_backoff = (
            tool_task_config.backoff
            if tool_task_config and tool_task_config.backoff
            else exponential(base=2, max=30)
        )

        self._model_request_task = wf.task(
            name=f"{self.name}.model_request",
            retries=self._model_retries,
            backoff=self._model_backoff,
        )(self._do_model_request)

        self._tool_call_task = wf.task(
            name=f"{self.name}.tool_call",
            retries=self._tool_retries,
            backoff=self._tool_backoff,
        )(self._do_tool_call)

    async def run(
        self,
        prompt: str,
        *,
        deps: Any = None,
        message_history: list[Any] | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the agent durably.

        Works like ``Agent.run()`` but the result is checkpointed. If the
        process crashes and you call ``run()`` again with the same ``run_id``
        (or same prompt), the cached result is returned without re-executing.
        """
        rid = _run_id_for_agent(self.name, prompt, run_id)

        @self.wf.workflow(id=rid)
        async def _durable_run() -> Any:
            return await self._execute_agent_loop(
                prompt=prompt,
                deps=deps,
                message_history=message_history,
                **kwargs,
            )

        return await _durable_run.run(rid)

    def run_sync(
        self,
        prompt: str,
        *,
        deps: Any = None,
        message_history: list[Any] | None = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous version of :meth:`run`."""
        import asyncio

        return asyncio.run(
            self.run(
                prompt,
                deps=deps,
                message_history=message_history,
                run_id=run_id,
                **kwargs,
            )
        )

    async def _execute_agent_loop(
        self,
        prompt: str,
        deps: Any = None,
        message_history: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Wrap the full agent.run() as a single checkpointed task."""
        result = await self._model_request_task(
            prompt,
            deps=deps,
            message_history=message_history,
            step_id="agent-run",
            **kwargs,
        )
        return result

    async def _do_model_request(
        self,
        prompt: str,
        deps: Any = None,
        message_history: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the actual agent.run() call."""
        run_kwargs: dict[str, Any] = {}
        if deps is not None:
            run_kwargs["deps"] = deps
        if message_history is not None:
            run_kwargs["message_history"] = message_history
        run_kwargs.update(kwargs)

        result = await self.agent.run(prompt, **run_kwargs)
        return _AgentRunResult(result)

    async def _do_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> Any:
        """Execute a single tool call."""
        tools = getattr(self.agent, "_function_tools", {})
        if tool_name not in tools:
            raise ValueError(
                f"Tool {tool_name!r} not found on agent {self.name!r}. "
                f"Available: {list(tools.keys())}"
            )
        tool = tools[tool_name]
        return await tool.run(tool_args)

    async def tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        step_id: str | None = None,
    ) -> Any:
        """Durably execute a tool call with checkpointing."""
        return await self._tool_call_task(tool_name, tool_args, step_id=step_id)

    async def signal(self, name: str, *, poll: float = 2.0) -> Any:
        """Durably wait for an external signal (e.g., human approval)."""
        return await self.wf.signal(name, poll=poll)

    def __repr__(self) -> str:
        return (
            f"<DurableAgent name={self.name!r} "
            f"model_retries={self._model_retries} "
            f"tool_retries={self._tool_retries}>"
        )


class _AgentRunResult:
    """Wrapper that holds an agent RunResult and handles both live and
    deserialized (dict) results transparently."""

    def __init__(self, result: Any) -> None:
        self._result = result

    @property
    def output(self) -> Any:
        if hasattr(self._result, "output"):
            return self._result.output
        if isinstance(self._result, dict):
            return self._result.get("output")
        return self._result

    @property
    def all_messages(self) -> list[Any]:
        if hasattr(self._result, "all_messages"):
            return self._result.all_messages()
        if isinstance(self._result, dict):
            return self._result.get("all_messages", [])
        return []

    @property
    def usage(self) -> Any:
        if hasattr(self._result, "usage"):
            return self._result.usage()
        if isinstance(self._result, dict):
            return self._result.get("usage")
        return None

    def __repr__(self) -> str:
        return f"<DurableRunResult output={self.output!r}>"


def durable_tool(
    wf: Workflow,
    *,
    name: str | None = None,
    retries: int = 2,
    backoff: BackoffStrategy | None = None,
):
    """Decorator to make a tool function durable (checkpointed as a @wf.task)."""
    _backoff = backoff or exponential(base=2, max=30)

    def decorator(fn):
        return wf.task(
            name=name or getattr(fn, "__name__", "tool"),
            retries=retries,
            backoff=_backoff,
        )(fn)

    return decorator


def durable_pipeline(
    wf: Workflow,
    *,
    id: str,  # noqa: A002
):
    """Syntactic sugar for ``@wf.workflow`` in multi-agent pipelines."""
    return wf.workflow(id=id)
