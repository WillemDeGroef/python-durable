"""
redis_example.py — Using RedisStore for distributed, TTL-backed workflows.

Checkpoints are stored in Redis with automatic expiration (default: 24 hours).
This is ideal for production deployments where multiple workers share state
or you want old runs to expire automatically.

Prerequisites:
    pip install python-durable[redis]
    # or: uv sync --all-extras

    # Redis must be running locally (default: localhost:6379)
    docker run -d --name redis -p 6379:6379 redis:latest

Run with:
    uv run python examples/redis_example.py
"""

import asyncio

from durable import RedisStore, Workflow
from durable.backoff import constant

store = RedisStore(
    url="redis://localhost:6379/0",
    ttl=3600,  # checkpoints expire after 1 hour
    prefix="example",
)
wf = Workflow("redis-demo", db=store)

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
    await store.setup()
    try:
        print("── First run: flaky_fetch retries, then both steps execute ──")
        result = await pipeline(job_id="job-1")
        print(f"  result: {result}")

        print("\n── Second run (same id): both steps replayed from Redis ──")
        result = await pipeline(job_id="job-1")
        print(f"  result: {result}")

        print("\n✓ Done. Checkpoints stored in Redis (TTL: 1 hour).")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
