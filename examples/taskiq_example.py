"""
taskiq_example.py — Triggering durable workflows from a taskiq task queue.

Pattern: taskiq handles distributed task dispatch and retries, while durable
provides crash recovery within each workflow. If a worker crashes mid-workflow
and taskiq retries the message, durable replays completed steps and continues
from where it left off — no wasted work.

This example uses InMemoryBroker + InMemoryStore (zero infrastructure).
For production, swap in a real broker (e.g. taskiq-redis) and a persistent
store (e.g. durable's RedisStore or SQLiteStore).

Run with:
    uv run python examples/taskiq_example.py
"""

import asyncio

from taskiq import InMemoryBroker

from durable import Workflow
from durable.store import InMemoryStore

broker = InMemoryBroker()
wf = Workflow("taskiq-demo", db=InMemoryStore())


@wf.task
async def fetch_data(url: str) -> dict:
    print(f"  [fetch_data] fetching {url}")
    return {"url": url, "payload": [1, 2, 3]}


@wf.task
async def transform(data: dict) -> dict:
    print(f"  [transform] processing {data['url']}")
    return {"source": data["url"], "total": sum(data["payload"])}


@wf.task
async def save_result(result: dict) -> str:
    print(f"  [save_result] saving total={result['total']}")
    return f"saved:{result['total']}"


@wf.workflow(id="pipeline-{job_id}")
async def pipeline(job_id: str) -> str:
    data = await fetch_data(f"https://api.example.com/{job_id}")
    result = await transform(data)
    return await save_result(result)


@broker.task
async def run_pipeline(job_id: str) -> str:
    return await pipeline(job_id=job_id)


async def main():
    await broker.startup()

    print("── First run: all steps execute ──")
    task = await run_pipeline.kiq(job_id="job-42")
    result = await task.wait_result()
    print(f"  result: {result.return_value}")

    print("\n── Second run (same job_id): all steps replayed from cache ──")
    task = await run_pipeline.kiq(job_id="job-42")
    result = await task.wait_result()
    print(f"  result: {result.return_value}")

    print("\n── Third run (new job_id): all steps execute fresh ──")
    task = await run_pipeline.kiq(job_id="job-99")
    result = await task.wait_result()
    print(f"  result: {result.return_value}")

    await broker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
