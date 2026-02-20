"""
in_memory_example.py — Using InMemoryStore for testing and ephemeral workflows.

No files, no SQLite — just a dict. Checkpoints live for the duration of the
process, so retries and deduplication work within a single run, but nothing
survives a restart.

Run with:
    uv run python examples/in_memory_example.py
"""

import asyncio

from durable import Workflow
from durable.backoff import constant
from durable.store import InMemoryStore

wf = Workflow("demo", db=InMemoryStore())

call_count = 0


@wf.task(retries=3, backoff=constant(0))
async def flaky_fetch(url: str) -> dict:
    """Fails twice, then succeeds — retries are handled automatically."""
    global call_count
    call_count += 1
    if call_count < 3:
        raise ConnectionError(f"attempt {call_count}: connection refused")
    print(f"  [flaky_fetch] succeeded on attempt {call_count}")
    return {"url": url, "status": "ok"}


@wf.task
async def transform(data: dict) -> str:
    print(f"  [transform] processing {data['url']}")
    return f"transformed-{data['status']}"


@wf.workflow(id="pipeline-{job_id}")
async def pipeline(job_id: str) -> str:
    data = await flaky_fetch("https://api.example.com/data")
    return await transform(data)


async def main():
    print("── First run: flaky_fetch retries, then both steps execute ──")
    result = await pipeline(job_id="job-1")
    print(f"  result: {result}")

    print("\n── Second run (same id): both steps replayed from memory ──")
    result = await pipeline(job_id="job-1")
    print(f"  result: {result}")

    print("\n✓ Done. Nothing written to disk.")


if __name__ == "__main__":
    asyncio.run(main())
