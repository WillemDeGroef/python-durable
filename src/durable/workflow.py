"""
Core Workflow class — the single object developers interact with.

    wf = Workflow("my-app")          # defaults to SQLite at ./durable.db
    wf = Workflow("my-app", db="sqlite:///jobs.db")
    wf = Workflow("my-app", db=MyPostgresStore())

Then decorate tasks and workflows:

    @wf.task
    async def my_task(x: int) -> str: ...

    @wf.task(retries=5, backoff=exponential(base=2, max=30))
    async def flaky_task(url: str) -> dict: ...

    @wf.workflow(id="job-{job_id}")
    async def my_workflow(job_id: str) -> None:
        result = await my_task(42)
        ...
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import re
from typing import Any, Callable, ParamSpec, TypeVar, overload

from .backoff import BackoffStrategy, exponential
from .context import RunContext, _active_run
from .store import SQLiteStore, Store

log = logging.getLogger("durable")

P = ParamSpec("P")
R = TypeVar("R")

# Matches "{param_name}" in workflow id templates
_TEMPLATE_RE = re.compile(r"\{(\w+)\}")


def _format_run_id(template: str, bound: inspect.BoundArguments) -> str:
    """Replace {param} placeholders with actual argument values."""
    args = {**bound.arguments}
    # Flatten **kwargs into the top-level dict if present
    for k, v in list(args.items()):
        if isinstance(v, dict) and k not in _TEMPLATE_RE.findall(template):
            args.update(v)
    return template.format_map(args)


class _TaskWrapper:
    """
    Wraps an async function with checkpoint + retry logic.

    Behaves like the original coroutine function — fully typed, awaitable,
    introspectable. The durable magic only activates when called inside an
    active workflow run (i.e. a RunContext is set in the ContextVar).

    Outside a workflow, it just executes normally — great for testing.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        step_name: str,
        retries: int,
        backoff: BackoffStrategy,
    ) -> None:
        self._fn = fn
        self._step_name = step_name
        self._retries = retries
        self._backoff = backoff
        # Preserve the original function's metadata for IDE / tooling support
        functools.update_wrapper(self, fn)

    async def __call__(
        self, *args: Any, step_id: str | None = None, **kwargs: Any
    ) -> Any:
        ctx = _active_run.get()

        if ctx is None:
            # Called outside a workflow — run as a plain async function.
            log.debug(
                "[durable] %s called outside workflow context, running directly",
                self._step_name,
            )
            return await self._fn(*args, **kwargs)

        sid = step_id or ctx.next_step_id(self._step_name)
        found, cached = await ctx.store.get_step(ctx.run_id, sid)

        if found:
            log.debug(
                "[durable] ↩  %s (step=%s) — replayed from store", self._step_name, sid
            )
            return cached

        return await self._execute_with_retry(ctx, sid, args, kwargs)

    async def _execute_with_retry(
        self, ctx: RunContext, sid: str, args: tuple, kwargs: dict
    ) -> Any:
        last_exc: Exception | None = None

        for attempt in range(self._retries + 1):
            if attempt > 0:
                wait = self._backoff(attempt - 1)
                log.warning(
                    "[durable] ↻  %s (step=%s) attempt %d/%d — retrying in %.1fs",
                    self._step_name,
                    sid,
                    attempt,
                    self._retries,
                    wait,
                )
                await asyncio.sleep(wait)

            try:
                result = await self._fn(*args, **kwargs)
                await ctx.store.set_step(ctx.run_id, sid, result, attempt + 1)
                log.debug("[durable] ✓  %s (step=%s)", self._step_name, sid)
                return result

            except Exception as exc:
                last_exc = exc
                log.warning(
                    "[durable] ✗  %s (step=%s) attempt %d failed: %s",
                    self._step_name,
                    sid,
                    attempt + 1,
                    exc,
                )

        raise last_exc  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"<DurableTask '{self._step_name}' retries={self._retries}>"


class Workflow:
    """
    The main entry point for the durable library.

    Create one per application (or one per logical domain):

        wf = Workflow("orders", db="sqlite:///orders.db")

    Then use @wf.task and @wf.workflow to make your code durable.
    """

    def __init__(
        self,
        name: str,
        db: str | Store = "durable.db",
        default_retries: int = 3,
        default_backoff: BackoffStrategy = exponential(),
    ) -> None:
        self.name = name
        self._default_retries = default_retries
        self._default_backoff = default_backoff
        self._store = self._build_store(db)
        self._initialized = False

    # ------------------------------------------------------------------
    # @wf.task — can be used bare or with arguments
    # ------------------------------------------------------------------

    @overload
    def task(self, fn: Callable[P, R]) -> _TaskWrapper: ...

    @overload
    def task(
        self,
        fn: None = None,
        *,
        name: str | None = None,
        retries: int | None = None,
        backoff: BackoffStrategy | None = None,
    ) -> Callable[[Callable[P, R]], _TaskWrapper]: ...

    def task(
        self,
        fn: Callable | None = None,
        *,
        name: str | None = None,
        retries: int | None = None,
        backoff: BackoffStrategy | None = None,
    ) -> _TaskWrapper | Callable:
        """
        Decorate an async function to make it a durable task.

        Usage (bare):
            @wf.task
            async def fetch_user(user_id: str) -> User: ...

        Usage (with options):
            @wf.task(retries=5, backoff=exponential(base=2, max=120))
            async def call_api(url: str) -> dict: ...

            @wf.task(name="send-welcome-email")
            async def send_email(user: User) -> None: ...
        """

        def decorator(func: Callable[..., Any]) -> _TaskWrapper:
            fn_name = getattr(func, "__name__", repr(func))
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(
                    f"@wf.task requires an async function, got: {fn_name!r}"
                )
            return _TaskWrapper(
                func,
                step_name=name or fn_name,
                retries=retries if retries is not None else self._default_retries,
                backoff=backoff or self._default_backoff,
            )

        if fn is not None:
            # Used bare: @wf.task
            return decorator(fn)
        # Used with args: @wf.task(retries=5)
        return decorator

    # ------------------------------------------------------------------
    # @wf.workflow — entry point for a durable run
    # ------------------------------------------------------------------

    def workflow(
        self,
        fn: Callable | None = None,
        *,
        id: str | None = None,  # noqa: A002  (shadows builtin intentionally)
    ) -> Callable:
        """
        Decorate an async function as a durable workflow entry point.

        The `id` parameter is a template string resolved from the function's
        arguments at call time:

            @wf.workflow(id="process-order-{order_id}")
            async def process_order(order_id: str) -> None: ...

            await process_order(order_id="ord-99")
            # run_id → "process-order-ord-99"

        If `id` is omitted, the run_id is "{workflow_name}-{fn_name}-{all_args_joined}".

        You can also call `.run(run_id, ...)` to supply an explicit run ID:

            await process_order.run("my-run-123", order_id="ord-99")
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            fn_name = getattr(func, "__name__", repr(func))
            if not asyncio.iscoroutinefunction(func):
                raise TypeError(
                    f"@wf.workflow requires an async function, got: {fn_name!r}"
                )

            id_template = id  # capture for closure
            sig = inspect.signature(func)

            async def _run_with_id(run_id: str, *args: Any, **kwargs: Any) -> Any:
                await self._ensure_initialized()

                # Create a new RunContext and install it in the ContextVar.
                ctx = RunContext(
                    run_id=run_id,
                    workflow_id=f"{self.name}.{fn_name}",
                    store=self._store,
                )

                if isinstance(self._store, SQLiteStore):
                    await self._store.ensure_run(run_id, ctx.workflow_id)

                token = _active_run.set(ctx)
                try:
                    result = await func(*args, **kwargs)
                    await self._store.mark_run_done(run_id)
                    log.info("[durable] ✓✓ workflow %s (%s) completed", fn_name, run_id)
                    return result
                except Exception as exc:
                    await self._store.mark_run_failed(run_id, str(exc))
                    log.error(
                        "[durable] ✗✗ workflow %s (%s) failed: %s",
                        fn_name,
                        run_id,
                        exc,
                    )
                    raise
                finally:
                    _active_run.reset(token)

            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                if id_template:
                    run_id = _format_run_id(id_template, bound)
                else:
                    parts = [self.name, fn_name] + [
                        str(v) for v in bound.arguments.values()
                    ]
                    run_id = "-".join(parts)

                return await _run_with_id(run_id, *args, **kwargs)

            async def run(run_id: str, *args: Any, **kwargs: Any) -> Any:
                """Call with an explicit run ID instead of the template-derived one."""
                return await _run_with_id(run_id, *args, **kwargs)

            wrapper.run = run  # type: ignore[attr-defined]
            return wrapper

        if fn is not None:
            return decorator(fn)
        return decorator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._store.setup()
            self._initialized = True

    @staticmethod
    def _build_store(db: str | Store) -> Store:
        if isinstance(db, Store):
            return db
        # Accept both "sqlite:///path.db" and bare "path.db"
        path = db.removeprefix("sqlite:///")
        return SQLiteStore(path)
