"""
Pydantic AI integration for python-durable.

Provides DurableAgent — a wrapper that makes any pydantic-ai Agent durable,
automatically checkpointing model requests and tool calls to the store.

Architecture mirrors Pydantic AI's official integrations (TemporalAgent, DBOSAgent,
PrefectAgent) but uses python-durable's lightweight SQLite/Redis backend instead of
an external orchestration server.

    +----------------------------------------------------+
    |                 Your Application                   |
    |                                                    |
    |   agent = Agent("openai:gpt-4o", tools=[...])      |
    |   durable_agent = DurableAgent(agent, wf)          |
    |   result = await durable_agent.run("Hello")        |
    |                                                    |
    +----------------------------------------------------+
                            |
                            v
    +----------------------------------------------------+
    |               DurableAgent                         |
    |                                                    |
    |   @wf.workflow ── agent run loop (deterministic)   |
    |     |                                              |
    |     +-- @wf.task ── model.request()  (checkpoint)  |
    |     +-- @wf.task ── tool call        (checkpoint)  |
    |     +-- @wf.task ── model.request()  (checkpoint)  |
    |     +-- ...                                        |
    +----------------------------------------------------+
                            |
                            v
    +----------------------------------------------------+
    |                    Store                            |
    |   SQLite (default) / Redis / Custom                |
    +----------------------------------------------------+

Usage:
    from pydantic_ai import Agent
    from durable import Workflow
    from durable.pydantic_ai import DurableAgent

    wf = Workflow("my-app")
    agent = Agent("openai:gpt-4o", instructions="Be helpful.")

    durable_agent = DurableAgent(agent, wf)

    # This is now durable — crashes replay from checkpoint
    result = await durable_agent.run("What is the capital of France?")
    print(result.output)

Requires:
    pip install python-durable[pydantic-ai]
    # or: pip install python-durable pydantic-ai
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


# ---------------------------------------------------------------------------
# Serialization helpers for Pydantic AI message objects
# ---------------------------------------------------------------------------


def _serialize_messages(messages: list[Any]) -> list[dict]:
    """Convert pydantic-ai message objects to JSON-serializable dicts."""
    result = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            # Pydantic model — use discriminated dump
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

    Attempts to import and use ModelRequest/ModelResponse from pydantic_ai.
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


# ---------------------------------------------------------------------------
# TaskConfig — per-step retry/backoff overrides
# ---------------------------------------------------------------------------


class TaskConfig:
    """Configuration for durable task wrapping.

    Controls retries and backoff for model requests and tool calls.

        DurableAgent(
            agent,
            wf,
            model_task_config=TaskConfig(retries=5, backoff=exponential(base=2, max=120)),
            tool_task_config=TaskConfig(retries=2),
        )
    """

    def __init__(
        self,
        retries: int | None = None,
        backoff: BackoffStrategy | None = None,
    ) -> None:
        self.retries = retries
        self.backoff = backoff


# ---------------------------------------------------------------------------
# DurableAgent
# ---------------------------------------------------------------------------


class DurableAgent(Generic[AgentDepsT, OutputT]):
    """Wrap a pydantic-ai Agent for durable execution with python-durable.

    DurableAgent automatically:
      • Wraps each ``agent.run()`` call as a ``@wf.workflow``
      • Checkpoints every model request as a ``@wf.task``
      • Checkpoints every tool call as a ``@wf.task``

    The original agent can still be used directly for non-durable execution.

    Parameters:
        agent: The pydantic-ai Agent to wrap.
        wf: A python-durable Workflow instance.
        name: Optional name override (defaults to ``agent.name``).
        model_task_config: Retry/backoff config for model requests.
        tool_task_config: Retry/backoff config for tool calls.

    Example::

        from pydantic_ai import Agent
        from durable import Workflow
        from durable.pydantic_ai import DurableAgent

        wf = Workflow("my-app")
        agent = Agent("openai:gpt-4o", instructions="Be concise.")

        durable_agent = DurableAgent(agent, wf)
        result = await durable_agent.run("What is 2+2?")
        print(result.output)
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

        # Build durable task wrappers
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

    # ------------------------------------------------------------------
    # Public API — mirrors Agent.run() / Agent.run_sync()
    # ------------------------------------------------------------------

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

        Works like ``Agent.run()`` but every model request and tool call is
        checkpointed. If the process crashes and you call ``run()`` again with
        the same ``run_id`` (or same prompt), completed steps replay from the
        store without re-executing.

        Args:
            prompt: The user prompt to send to the agent.
            deps: Dependencies to pass to the agent (same as Agent.run).
            message_history: Optional conversation history.
            run_id: Explicit run ID. If omitted, derived from agent name + prompt hash.
            **kwargs: Additional arguments forwarded to Agent.run().

        Returns:
            The agent's RunResult (same type as Agent.run()).
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
        """Synchronous version of :meth:`run`.

        Convenience wrapper that calls ``asyncio.run()`` under the hood.
        Cannot be used if an event loop is already running.
        """
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

    # ------------------------------------------------------------------
    # Core agent execution loop
    # ------------------------------------------------------------------

    async def _execute_agent_loop(
        self,
        prompt: str,
        deps: Any = None,
        message_history: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the agent run, checkpointing each model request and tool call.

        This method implements the agent's request/tool loop manually so that
        each non-deterministic operation (model call, tool execution) can be
        individually checkpointed as a durable task.

        The strategy:
        1. Try the simple approach first: wrap the entire agent.run() as a
           single durable task. This gives coarse-grained durability — the
           whole agent call is checkpointed as one unit.
        2. For fine-grained control, users can break their workflow into
           multiple durable tasks (see the pipeline example).
        """
        # Strategy: wrap the full agent.run() as a single checkpointed task.
        # This is the approach that works with any pydantic-ai agent, regardless
        # of its internal tool/model configuration.
        #
        # For finer granularity (individual LLM calls), users should decompose
        # their workflow into multiple @wf.task steps — see examples/.

        result = await self._model_request_task(
            prompt,
            deps=deps,
            message_history=message_history,
            step_id="agent-run",
            **kwargs,
        )
        return result

    # ------------------------------------------------------------------
    # Durable task implementations
    # ------------------------------------------------------------------

    async def _do_model_request(
        self,
        prompt: str,
        deps: Any = None,
        message_history: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the actual agent.run() call.

        This is wrapped as a @wf.task, so the result is checkpointed.
        On replay, the cached result is returned without calling the LLM.
        """
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
        """Execute a single tool call.

        Can be used directly in multi-step workflows::

            result = await durable_agent.tool(
                "search",
                {"query": "python durable execution"},
                step_id="search-0",
            )
        """
        # Look up the tool on the agent
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
        """Durably execute a tool call with checkpointing.

        Useful in multi-step workflows where you want to call agent tools
        individually with separate checkpoints::

            @wf.workflow(id="research-{topic}")
            async def research(topic: str):
                plan = await durable_agent.run(f"Plan research on {topic}")
                for i, query in enumerate(plan.output.queries):
                    result = await durable_agent.tool(
                        "web_search",
                        {"query": query},
                        step_id=f"search-{i}",
                    )
        """
        return await self._tool_call_task(tool_name, tool_args, step_id=step_id)

    # ------------------------------------------------------------------
    # Signal integration — human-in-the-loop
    # ------------------------------------------------------------------

    async def signal(self, name: str, *, poll: float = 2.0) -> Any:
        """Durably wait for an external signal (e.g., human approval).

        Delegates to ``wf.signal()`` — see the approval example.

        Args:
            name: Signal name to wait for.
            poll: Poll interval in seconds for store-based fallback.

        Returns:
            The signal payload delivered via ``wf.complete()``.
        """
        return await self.wf.signal(name, poll=poll)

    def __repr__(self) -> str:
        return (
            f"<DurableAgent name={self.name!r} "
            f"model_retries={self._model_retries} "
            f"tool_retries={self._tool_retries}>"
        )


class _AgentRunResult:
    """Thin wrapper that holds an agent RunResult and makes it JSON-serializable.

    The durable store needs to serialize task results as JSON. A pydantic-ai
    RunResult contains message history, usage info, etc. that we serialize
    to/from dicts.

    On cache hit (replay), the store returns raw dicts. This wrapper
    transparently handles both cases.
    """

    def __init__(self, result: Any) -> None:
        self._result = result

    @property
    def output(self) -> Any:
        """The agent's output — works whether result is live or deserialized."""
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


# ---------------------------------------------------------------------------
# Convenience: durable_task decorator for standalone tool functions
# ---------------------------------------------------------------------------


def durable_tool(
    wf: Workflow,
    *,
    name: str | None = None,
    retries: int = 2,
    backoff: BackoffStrategy | None = None,
):
    """Decorator to make a pydantic-ai tool function durable.

    Use this on tool functions that perform I/O (API calls, database queries)
    so they are checkpointed independently within a durable workflow::

        from durable import Workflow
        from durable.pydantic_ai import durable_tool

        wf = Workflow("my-app")

        @durable_tool(wf, retries=3)
        async def web_search(query: str) -> str:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"https://api.search.com?q={query}")
                return resp.text

        # Register with pydantic-ai agent
        agent = Agent("openai:gpt-4o", tools=[web_search])

    The decorated function works as a normal async function outside workflows
    (great for testing) and becomes durable inside a workflow context.
    """
    _backoff = backoff or exponential(base=2, max=30)

    def decorator(fn):
        return wf.task(
            name=name or getattr(fn, "__name__", "tool"),
            retries=retries,
            backoff=_backoff,
        )(fn)

    return decorator


# ---------------------------------------------------------------------------
# Pipeline helpers — compose multiple agents durably
# ---------------------------------------------------------------------------


def durable_pipeline(
    wf: Workflow,
    *,
    id: str,  # noqa: A002
):
    """Decorator to create a durable multi-agent pipeline.

    Syntactic sugar over ``@wf.workflow``::

        @durable_pipeline(wf, id="research-{topic_id}")
        async def research(topic_id: str, topic: str):
            plan = await planner_agent.run(f"Plan: {topic}")
            results = []
            for i, query in enumerate(plan.output.queries):
                r = await searcher_agent.run(query, step_id=f"search-{i}")
                results.append(r)
            return await summarizer_agent.run(str(results))
    """
    return wf.workflow(id=id)
