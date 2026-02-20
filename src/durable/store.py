"""
Storage backends for durable step checkpoints.

Default: SQLiteStore (zero deps beyond aiosqlite).
Override by passing any Store subclass to Workflow(db=...).

Schema is intentionally minimal — two tables:
  runs  → track lifecycle of a workflow execution
  steps → store serialized results keyed by (run_id, step_id)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import aiosqlite

# We wrap every stored value in {"v": ...} so we can distinguish
# "step not found" (None) from "step completed and returned None" ({"v": null}).
_WRAP_KEY = "v"


def _wrap(value: Any) -> str:
    return json.dumps({_WRAP_KEY: value})


def _unwrap(raw: str) -> Any:
    return json.loads(raw)[_WRAP_KEY]


class Store(ABC):
    """Abstract base — implement this to use Postgres, Redis, etc."""

    @abstractmethod
    async def setup(self) -> None:
        """Called once before the store is used. Create tables, connect pools, etc."""

    @abstractmethod
    async def get_step(self, run_id: str, step_id: str) -> tuple[bool, Any]:
        """Return (found, value). found=False means the step hasn't run yet."""

    @abstractmethod
    async def set_step(
        self, run_id: str, step_id: str, result: Any, attempt: int = 1
    ) -> None:
        """Persist a completed step result."""

    @abstractmethod
    async def mark_run_done(self, run_id: str) -> None:
        """Mark the whole workflow run as successfully completed."""

    @abstractmethod
    async def mark_run_failed(self, run_id: str, error: str) -> None:
        """Mark the run as failed with an error message."""


_SENTINEL = object()


class InMemoryStore(Store):
    """
    Dict-backed store — no persistence, no dependencies.

    Useful for testing, short-lived scripts, and development where you don't
    need crash recovery across process restarts.
    """

    def __init__(self) -> None:
        self._steps: dict[tuple[str, str], Any] = {}
        self._runs: dict[str, str] = {}

    async def setup(self) -> None:
        pass

    async def get_step(self, run_id: str, step_id: str) -> tuple[bool, Any]:
        value = self._steps.get((run_id, step_id), _SENTINEL)
        if value is _SENTINEL:
            return False, None
        return True, value

    async def set_step(
        self, run_id: str, step_id: str, result: Any, attempt: int = 1
    ) -> None:
        self._steps[(run_id, step_id)] = result

    async def mark_run_done(self, run_id: str) -> None:
        self._runs[run_id] = "done"

    async def mark_run_failed(self, run_id: str, error: str) -> None:
        self._runs[run_id] = "failed"


class SQLiteStore(Store):
    """
    Default store backed by a local SQLite file via aiosqlite.

    Works great for local dev, single-process services, CLIs, and scripts.
    For production or multi-process workloads, swap in a Postgres/Redis store.
    """

    def __init__(self, path: str = "durable.db") -> None:
        self.path = path
        self._ready = False

    async def setup(self) -> None:
        if self._ready:
            return
        async with aiosqlite.connect(self.path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id      TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'running',
                    error       TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS steps (
                    run_id     TEXT NOT NULL,
                    step_id    TEXT NOT NULL,
                    result     TEXT NOT NULL,
                    attempt    INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, step_id)
                );
            """)
            await db.commit()
        self._ready = True

    async def get_step(self, run_id: str, step_id: str) -> tuple[bool, Any]:
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "SELECT result FROM steps WHERE run_id = ? AND step_id = ?",
                (run_id, step_id),
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return False, None
                return True, _unwrap(row[0])

    async def set_step(
        self, run_id: str, step_id: str, result: Any, attempt: int = 1
    ) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO steps (run_id, step_id, result, attempt)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id, step_id) DO UPDATE
                    SET result = excluded.result, attempt = excluded.attempt
                """,
                (run_id, step_id, _wrap(result), attempt),
            )
            await db.commit()

    async def _upsert_run(
        self, run_id: str, workflow_id: str, status: str, error: str | None = None
    ) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO runs (run_id, workflow_id, status, error)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE
                    SET status = excluded.status,
                        error  = excluded.error,
                        updated_at = CURRENT_TIMESTAMP
                """,
                (run_id, workflow_id, status, error),
            )
            await db.commit()

    async def mark_run_done(self, run_id: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "UPDATE runs SET status = 'done', updated_at = CURRENT_TIMESTAMP WHERE run_id = ?",
                (run_id,),
            )
            await db.commit()

    async def mark_run_failed(self, run_id: str, error: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "UPDATE runs SET status = 'failed', error = ?, updated_at = CURRENT_TIMESTAMP WHERE run_id = ?",
                (error, run_id),
            )
            await db.commit()

    async def ensure_run(self, run_id: str, workflow_id: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO runs (run_id, workflow_id, status) VALUES (?, ?, 'running')",
                (run_id, workflow_id),
            )
            await db.commit()
