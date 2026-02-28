"""
Tests for durable signals (human-in-the-loop / external input).

Covers:
  1. Basic flow: workflow waits, external completes, workflow receives payload
  2. Replay: re-running after delivery returns cached payload instantly
  3. Idempotent complete: second complete() returns False
  4. Signal before wait: complete() before workflow reaches signal()
  5. Multiple signals: workflow waits for two independent signals
"""

import asyncio

from fakeredis.aioredis import FakeRedis

from durable import InMemoryStore, RedisStore, Workflow


# ---------------------------------------------------------------------------
# InMemoryStore tests
# ---------------------------------------------------------------------------


def _mem_wf() -> Workflow:
    return Workflow("test-sig", db=InMemoryStore(), default_retries=0)


async def test_basic_signal_flow():
    wf = _mem_wf()
    received = []

    @wf.workflow(id="basic-sig")
    async def my_workflow() -> dict:
        result = await wf.signal("approval")
        received.append(result)
        return result

    async def deliver():
        await asyncio.sleep(0.05)
        await wf.complete("basic-sig", "approval", {"approved": True})

    asyncio.create_task(deliver())
    result = await my_workflow()
    assert result == {"approved": True}
    assert received == [{"approved": True}]


async def test_signal_replay():
    wf = _mem_wf()
    call_count = 0

    @wf.task
    async def before_signal() -> str:
        nonlocal call_count
        call_count += 1
        return "step-done"

    @wf.workflow(id="replay-sig")
    async def my_workflow() -> dict:
        await before_signal()
        return await wf.signal("approval")

    # Deliver signal, run workflow
    async def deliver():
        await asyncio.sleep(0.05)
        await wf.complete("replay-sig", "approval", {"ok": True})

    asyncio.create_task(deliver())
    result = await my_workflow()
    assert result == {"ok": True}
    assert call_count == 1

    # Re-run: signal is already in the store, no waiting
    call_count = 0
    result = await my_workflow()
    assert result == {"ok": True}
    assert call_count == 0  # task replayed from checkpoint


async def test_idempotent_complete():
    wf = _mem_wf()

    @wf.workflow(id="idem-sig")
    async def my_workflow() -> dict:
        return await wf.signal("approval")

    # Complete before workflow runs
    ok1 = await wf.complete("idem-sig", "approval", {"first": True})
    ok2 = await wf.complete("idem-sig", "approval", {"second": True})
    assert ok1 is True
    assert ok2 is False

    # Workflow gets the first payload
    result = await my_workflow()
    assert result == {"first": True}


async def test_signal_before_wait():
    wf = _mem_wf()

    @wf.workflow(id="pre-sig")
    async def my_workflow() -> dict:
        return await wf.signal("approval")

    # Deliver before workflow starts
    await wf.complete("pre-sig", "approval", {"early": True})

    # Workflow should pick it up immediately (no waiting)
    result = await my_workflow()
    assert result == {"early": True}


async def test_multiple_signals():
    wf = _mem_wf()

    @wf.workflow(id="multi-sig")
    async def my_workflow() -> list:
        a = await wf.signal("step-a")
        b = await wf.signal("step-b")
        return [a, b]

    async def deliver():
        await asyncio.sleep(0.05)
        await wf.complete("multi-sig", "step-a", {"val": "A"})
        await asyncio.sleep(0.05)
        await wf.complete("multi-sig", "step-b", {"val": "B"})

    asyncio.create_task(deliver())
    result = await my_workflow()
    assert result == [{"val": "A"}, {"val": "B"}]


# ---------------------------------------------------------------------------
# RedisStore tests
# ---------------------------------------------------------------------------


def _redis_store(ttl: int = 86_400) -> RedisStore:
    store = RedisStore(ttl=ttl)
    store._redis = FakeRedis(decode_responses=True)
    return store


def _redis_wf(store: RedisStore) -> Workflow:
    return Workflow("test-redis-sig", db=store, default_retries=0)


async def test_redis_basic_signal_flow():
    store = _redis_store()
    wf = _redis_wf(store)

    @wf.workflow(id="redis-basic-sig")
    async def my_workflow() -> dict:
        return await wf.signal("approval")

    async def deliver():
        await asyncio.sleep(0.05)
        await wf.complete("redis-basic-sig", "approval", {"approved": True})

    asyncio.create_task(deliver())
    result = await my_workflow()
    assert result == {"approved": True}


async def test_redis_signal_replay():
    store = _redis_store()
    wf = _redis_wf(store)

    @wf.workflow(id="redis-replay-sig")
    async def my_workflow() -> dict:
        return await wf.signal("approval")

    async def deliver():
        await asyncio.sleep(0.05)
        await wf.complete("redis-replay-sig", "approval", {"ok": True})

    asyncio.create_task(deliver())
    result = await my_workflow()
    assert result == {"ok": True}

    # Re-run: replayed from store
    result = await my_workflow()
    assert result == {"ok": True}


async def test_redis_idempotent_complete():
    store = _redis_store()
    wf = _redis_wf(store)

    ok1 = await wf.complete("redis-idem", "approval", {"first": True})
    ok2 = await wf.complete("redis-idem", "approval", {"second": True})
    assert ok1 is True
    assert ok2 is False

    # Verify first payload persisted
    found, payload = await store.get_signal("redis-idem", "approval")
    assert found is True
    assert payload == {"first": True}


async def test_redis_signal_before_wait():
    store = _redis_store()
    wf = _redis_wf(store)

    @wf.workflow(id="redis-pre-sig")
    async def my_workflow() -> dict:
        return await wf.signal("approval")

    await wf.complete("redis-pre-sig", "approval", {"early": True})
    result = await my_workflow()
    assert result == {"early": True}


async def test_redis_signal_ttl():
    store = _redis_store(ttl=120)
    wf = _redis_wf(store)

    await wf.complete("ttl-test", "approval", {"data": 1})
    keys = [k for k in await store._client().keys("*") if "sig" in k]
    assert len(keys) == 1
    ttl = await store._client().ttl(keys[0])
    assert 0 < ttl <= 120


async def test_redis_multiple_signals():
    store = _redis_store()
    wf = _redis_wf(store)

    @wf.workflow(id="redis-multi-sig")
    async def my_workflow() -> list:
        a = await wf.signal("step-a")
        b = await wf.signal("step-b")
        return [a, b]

    async def deliver():
        await asyncio.sleep(0.05)
        await wf.complete("redis-multi-sig", "step-a", {"val": "A"})
        await asyncio.sleep(0.05)
        await wf.complete("redis-multi-sig", "step-b", {"val": "B"})

    asyncio.create_task(deliver())
    result = await my_workflow()
    assert result == [{"val": "A"}, {"val": "B"}]
