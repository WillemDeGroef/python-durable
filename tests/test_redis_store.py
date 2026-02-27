import json

import pytest
from fakeredis.aioredis import FakeRedis

from durable import RedisStore, Workflow


def _make_store(ttl: int = 86_400) -> RedisStore:
    store = RedisStore(ttl=ttl)
    store._redis = FakeRedis(decode_responses=True)
    return store


def _make_wf(store: RedisStore) -> Workflow:
    return Workflow("test-redis", db=store, default_retries=0)


async def test_get_missing_step_returns_not_found():
    store = _make_store()
    await store.setup()
    found, value = await store.get_step("run-1", "step-1")
    assert found is False
    assert value is None


async def test_set_and_get_step():
    store = _make_store()
    await store.setup()

    await store.set_step("run-1", "step-1", {"count": 42})
    found, value = await store.get_step("run-1", "step-1")
    assert found is True
    assert value == {"count": 42}


async def test_step_returns_none_value():
    store = _make_store()
    await store.setup()

    await store.set_step("run-1", "step-1", None)
    found, value = await store.get_step("run-1", "step-1")
    assert found is True
    assert value is None


async def test_ttl_is_applied():
    store = _make_store(ttl=60)
    await store.setup()

    await store.set_step("run-1", "step-1", "hello")
    key = next(k for k in await store._client().keys("*") if "step" in k)
    ttl = await store._client().ttl(key)
    assert 0 < ttl <= 60


async def test_no_ttl_when_zero():
    store = _make_store(ttl=0)
    await store.setup()

    await store.set_step("run-1", "step-1", "hello")
    key = next(k for k in await store._client().keys("*") if "step" in k)
    ttl = await store._client().ttl(key)
    assert ttl == -1


async def test_mark_run_done():
    store = _make_store()
    await store.setup()

    await store.mark_run_done("run-1")
    key = next(k for k in await store._client().keys("*") if "run" in k)
    assert await store._client().get(key) == "done"


async def test_mark_run_failed():
    store = _make_store()
    await store.setup()

    await store.mark_run_failed("run-1", "kaboom")
    key = next(k for k in await store._client().keys("*") if "run" in k)
    raw = await store._client().get(key)
    data = json.loads(raw)
    assert data["status"] == "failed"
    assert data["error"] == "kaboom"


async def test_overwrite_step():
    store = _make_store()
    await store.setup()

    await store.set_step("run-1", "step-1", "first")
    await store.set_step("run-1", "step-1", "second")
    found, value = await store.get_step("run-1", "step-1")
    assert found is True
    assert value == "second"


async def test_close():
    store = _make_store()
    await store.setup()
    await store.close()
    assert store._redis is None


async def test_workflow_checkpoints_with_redis():
    store = _make_store()
    wf = _make_wf(store)
    call_log: list[str] = []

    @wf.task
    async def step_a() -> str:
        call_log.append("a")
        return "result-a"

    @wf.task
    async def step_b(x: str) -> str:
        call_log.append("b")
        return f"{x}+b"

    @wf.workflow(id="ckpt-test")
    async def pipeline() -> str:
        a = await step_a()
        return await step_b(a)

    result = await pipeline()
    assert result == "result-a+b"
    assert call_log == ["a", "b"]

    call_log.clear()
    result = await pipeline()
    assert result == "result-a+b"
    assert call_log == []


async def test_workflow_resumes_after_failure():
    store = _make_store()
    wf = _make_wf(store)
    call_count = 0

    @wf.task
    async def good_step() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    fail_once = True

    @wf.task
    async def flaky_step(x: str) -> str:
        nonlocal fail_once, call_count
        call_count += 1
        if fail_once:
            fail_once = False
            raise RuntimeError("boom")
        return f"{x}!"

    @wf.workflow(id="resume-test")
    async def pipeline() -> str:
        a = await good_step()
        return await flaky_step(a)

    with pytest.raises(RuntimeError, match="boom"):
        await pipeline()

    assert call_count == 2

    call_count = 0
    result = await pipeline()
    assert result == "ok!"
    assert call_count == 1
