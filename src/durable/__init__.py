"""
durable — lightweight workflow durability for Python.

Make any async workflow resumable after crashes with just a decorator.
Backed by SQLite out of the box; swap in any Store subclass for production.

Quick start:

    from durable import Workflow
    from durable.backoff import exponential

    wf = Workflow("my-app")

    @wf.task(retries=3, backoff=exponential(base=2, max=60))
    async def fetch_data(url: str) -> dict:
        async with httpx.AsyncClient() as client:
            return (await client.get(url)).json()

    @wf.task
    async def save_result(data: dict) -> None:
        await db.insert(data)

    @wf.workflow(id="pipeline-{source}")
    async def run_pipeline(source: str) -> None:
        data = await fetch_data(f"https://api.example.com/{source}")
        await save_result(data)

    # First call: runs all steps and checkpoints each one.
    # If it crashes and you call it again with the same args,
    # completed steps are replayed from SQLite instantly.
    await run_pipeline(source="users")
"""

from .backoff import BackoffStrategy, constant, exponential, linear
from .context import RunContext
from .redis_store import RedisStore
from .store import InMemoryStore, SQLiteStore, Store
from .workflow import Workflow

__all__ = [
    "Workflow",
    "Store",
    "SQLiteStore",
    "InMemoryStore",
    "RedisStore",
    "RunContext",
    "BackoffStrategy",
    "exponential",
    "constant",
    "linear",
]

__version__ = "0.1.0"
