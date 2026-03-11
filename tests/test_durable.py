"""
tests.py — verifies durable's core contract:
  1. Completed steps are never re-executed after a crash.
  2. Failed steps retry with backoff.
  3. Workflows resume from the exact failure point.
  4. Tasks work normally when called outside a workflow.
"""

import tempfile

import pytest

from durable import Workflow
from durable.backoff import constant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_wf(tmp_path: str) -> Workflow:
    return Workflow("test", db=f"{tmp_path}/test.db", default_retries=0)


# ---------------------------------------------------------------------------
# 1. Completed steps are replayed, not re-executed
# ---------------------------------------------------------------------------


async def test_steps_are_checkpointed():
    with tempfile.TemporaryDirectory() as tmp:
        wf = make_wf(tmp)
        call_log = []

        @wf.task
        async def step_a() -> str:
            call_log.append("a")
            return "result-a"

        @wf.task
        async def step_b(prev: str) -> str:
            call_log.append("b")
            return f"result-b-{prev}"

        @wf.workflow(id="test-checkpoint")
        async def my_workflow() -> str:
            a = await step_a()
            b = await step_b(a)
            return b

        # First run: both steps execute
        result = await my_workflow()
        assert result == "result-b-result-a"
        assert call_log == ["a", "b"]

        # Second run: neither step re-executes
        call_log.clear()
        result = await my_workflow()
        assert result == "result-b-result-a"
        assert call_log == [], "Steps were re-executed but should have been replayed!"

    print("  ✓ Checkpoint replay works")


# ---------------------------------------------------------------------------
# 2. Crash mid-workflow → resume from failure point only
# ---------------------------------------------------------------------------


async def test_resume_after_crash():
    with tempfile.TemporaryDirectory() as tmp:
        wf = make_wf(tmp)
        call_log = []

        @wf.task
        async def step_fetch() -> dict:
            call_log.append("fetch")
            return {"id": "u1", "name": "Bob"}

        @wf.task
        async def step_process(user: dict) -> str:
            call_log.append("process")
            return f"processed-{user['id']}"

        @wf.task
        async def step_save(result: str) -> None:
            call_log.append("save")

        crashed = False

        @wf.workflow(id="test-crash-resume")
        async def pipeline() -> None:
            user = await step_fetch()
            result = await step_process(user)
            if not crashed:
                raise RuntimeError("simulated crash before save!")
            await step_save(result)

        # First run: crashes after step_process
        with pytest.raises(RuntimeError, match="simulated crash"):
            await pipeline()

        assert call_log == ["fetch", "process"]

        # "Restart" — only step_save should run
        call_log.clear()
        crashed = True  # let it through this time
        await pipeline()

        assert call_log == ["save"], (
            f"Expected only ['save'] but got {call_log} — "
            "fetch and process should have been replayed from cache"
        )

    print("  ✓ Crash + resume from exact failure point works")


# ---------------------------------------------------------------------------
# 3. Retry with backoff on failure
# ---------------------------------------------------------------------------


async def test_retry_on_failure():
    with tempfile.TemporaryDirectory() as tmp:
        wf = Workflow(
            "test", db=f"{tmp}/test.db", default_retries=3, default_backoff=constant(0)
        )
        attempts = []
        should_fail_until = 3

        @wf.task(retries=4, backoff=constant(0))
        async def flaky() -> str:
            attempts.append(1)
            if len(attempts) < should_fail_until:
                raise ValueError(f"failing on attempt {len(attempts)}")
            return "finally succeeded"

        @wf.workflow(id="test-retry")
        async def retrying_workflow() -> str:
            return await flaky()

        result = await retrying_workflow()
        assert result == "finally succeeded"
        assert len(attempts) == should_fail_until

    print("  ✓ Retry with backoff works")


# ---------------------------------------------------------------------------
# 4. Tasks work as plain async functions outside a workflow
# ---------------------------------------------------------------------------


async def test_task_outside_workflow():
    with tempfile.TemporaryDirectory() as tmp:
        wf = make_wf(tmp)

        @wf.task
        async def standalone(x: int) -> int:
            return x * 2

        # No workflow context → runs directly, no checkpoint logic
        result = await standalone(21)
        assert result == 42

    print("  ✓ Task works normally outside workflow context")


# ---------------------------------------------------------------------------
# 5. Loop steps with explicit step_id are each checkpointed independently
# ---------------------------------------------------------------------------


async def test_loop_with_step_id():
    with tempfile.TemporaryDirectory() as tmp:
        wf = make_wf(tmp)
        processed = []

        @wf.task
        async def process_item(item: int) -> int:
            processed.append(item)
            return item * 10

        @wf.workflow(id="test-loop")
        async def loop_workflow() -> list:
            results = []
            for i in range(5):
                r = await process_item(i, step_id=f"item-{i}")
                results.append(r)
            return results

        results = await loop_workflow()
        assert results == [0, 10, 20, 30, 40]
        assert processed == [0, 1, 2, 3, 4]

        # Rerun — nothing should execute
        processed.clear()
        results = await loop_workflow()
        assert results == [0, 10, 20, 30, 40]
        assert processed == []

    print("  ✓ Loop steps with step_id are individually checkpointed")


# ---------------------------------------------------------------------------
# 6. Same task reused across different workflows stays isolated
# ---------------------------------------------------------------------------


async def test_task_reuse_across_workflows():
    with tempfile.TemporaryDirectory() as tmp:
        wf = make_wf(tmp)
        call_log = []

        @wf.task
        async def shared_task(x: str) -> str:
            call_log.append(x)
            return f"done-{x}"

        @wf.workflow(id="wf-a")
        async def workflow_a() -> str:
            return await shared_task("a")

        @wf.workflow(id="wf-b")
        async def workflow_b() -> str:
            return await shared_task("b")

        r_a = await workflow_a()
        r_b = await workflow_b()

        assert r_a == "done-a"
        assert r_b == "done-b"
        assert call_log == ["a", "b"]

        # Rerun both — neither should fire shared_task again
        call_log.clear()
        await workflow_a()
        await workflow_b()
        assert call_log == []

    print("  ✓ Shared tasks are isolated per workflow run")


# ---------------------------------------------------------------------------
# 7. Pydantic model serialization/deserialization in @wf.task
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class UserModel(BaseModel):
    id: int
    name: str
    email: str


async def test_pydantic_model_serializes_from_task():
    """Pydantic model returned from @wf.task serializes without error on first run."""
    with tempfile.TemporaryDirectory() as tmp:
        wf = make_wf(tmp)

        @wf.task
        async def fetch_user() -> UserModel:
            return UserModel(id=1, name="Alice", email="alice@example.com")

        @wf.workflow(id="test-pydantic-serialize")
        async def my_workflow() -> UserModel:
            return await fetch_user()

        result = await my_workflow()
        assert isinstance(result, UserModel)
        assert result.id == 1
        assert result.name == "Alice"


async def test_pydantic_model_rehydrated_on_replay():
    """Pydantic model is correctly rehydrated on replay (not returned as dict)."""
    with tempfile.TemporaryDirectory() as tmp:
        wf = make_wf(tmp)
        call_log = []

        @wf.task
        async def fetch_user() -> UserModel:
            call_log.append("fetch")
            return UserModel(id=2, name="Bob", email="bob@example.com")

        @wf.workflow(id="test-pydantic-replay")
        async def my_workflow() -> UserModel:
            return await fetch_user()

        # First run
        result = await my_workflow()
        assert isinstance(result, UserModel)
        assert call_log == ["fetch"]

        # Second run — replayed from store
        call_log.clear()
        result = await my_workflow()
        assert call_log == [], "Task was re-executed but should have been replayed!"
        assert isinstance(result, UserModel), f"Expected UserModel, got {type(result)}"
        assert result.id == 2
        assert result.name == "Bob"


async def test_plain_types_still_work():
    """Plain dict/string/int returns still work (no regression)."""
    with tempfile.TemporaryDirectory() as tmp:
        wf = make_wf(tmp)

        @wf.task
        async def return_dict() -> dict:
            return {"key": "value"}

        @wf.task
        async def return_str() -> str:
            return "hello"

        @wf.task
        async def return_int() -> int:
            return 42

        @wf.workflow(id="test-plain-types")
        async def my_workflow():
            d = await return_dict()
            s = await return_str()
            i = await return_int()
            return d, s, i

        d, s, i = await my_workflow()
        assert d == {"key": "value"}
        assert s == "hello"
        assert i == 42

        # Replay
        d, s, i = await my_workflow()
        assert d == {"key": "value"}
        assert s == "hello"
        assert i == 42


async def test_pydantic_without_return_type_hint():
    """Task without return type hint still serializes Pydantic models (but no rehydration)."""
    with tempfile.TemporaryDirectory() as tmp:
        wf = make_wf(tmp)
        call_log = []

        @wf.task
        async def fetch_user():
            call_log.append("fetch")
            return UserModel(id=3, name="Charlie", email="charlie@example.com")

        @wf.workflow(id="test-pydantic-no-hint")
        async def my_workflow():
            return await fetch_user()

        # First run — should serialize without error
        result = await my_workflow()
        assert result.id == 3
        assert call_log == ["fetch"]

        # Replay — no type hint, so it comes back as dict (no rehydration)
        call_log.clear()
        result = await my_workflow()
        assert call_log == []
        assert isinstance(result, dict), f"Expected dict without type hint, got {type(result)}"
        assert result["id"] == 3
