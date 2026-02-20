"""
RunContext holds the state of a single workflow execution.

It's stored in a ContextVar so it propagates implicitly through async call chains —
tasks can read it without the caller threading it through manually (same pattern
as pydantic-ai's RunContext / dependency injection).
"""

from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import Store


@dataclass
class RunContext:
    run_id: str
    workflow_id: str
    store: "Store"

    # Tracks how many times each step name has been used so we can
    # auto-generate unique, deterministic step IDs without the user needing
    # to think about it.  e.g. fetch_user called twice → "fetch_user", "fetch_user#1"
    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def next_step_id(self, name: str) -> str:
        count = self._counters[name]
        self._counters[name] += 1
        return name if count == 0 else f"{name}#{count}"


# The single active run for the current async task / coroutine tree.
# set() returns a Token you can use to reset later (important for nesting).
_active_run: ContextVar[RunContext | None] = ContextVar("_active_run", default=None)
