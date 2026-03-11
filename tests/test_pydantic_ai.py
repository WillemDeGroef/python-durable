"""Tests for durable.pydantic_ai integration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from durable import InMemoryStore, Workflow
from durable.backoff import exponential
from durable.pydantic_ai import (
    DurableAgent,
    TaskConfig,
    _AgentRunResult,
    _run_id_for_agent,
    _serialize_messages,
    durable_pipeline,
    durable_tool,
)


@pytest.fixture
def wf():
    return Workflow("test-app", db=InMemoryStore())


def _mock_agent(name: str = "test-agent", output: str = "mocked output"):
    agent = MagicMock()
    agent.name = name

    run_result = MagicMock()
    run_result.output = output
    run_result.all_messages.return_value = []
    run_result.usage.return_value = {"tokens": 42}

    agent.run = AsyncMock(return_value=run_result)
    return agent, run_result


class TestDurableAgent:
    async def test_basic_run(self, wf):
        agent, _ = _mock_agent(output="Paris")
        durable = DurableAgent(agent, wf)

        result = await durable.run("What is the capital of France?", run_id="test-1")

        assert result.output == "Paris"
        agent.run.assert_called_once()

    async def test_replay_from_cache(self, wf):
        agent, _ = _mock_agent(output="Berlin")
        durable = DurableAgent(agent, wf)

        await durable.run("Capital of Germany?", run_id="test-replay")
        assert agent.run.call_count == 1

        await durable.run("Capital of Germany?", run_id="test-replay")
        assert agent.run.call_count == 1

    async def test_different_run_ids_execute_separately(self, wf):
        agent, _ = _mock_agent()
        durable = DurableAgent(agent, wf)

        await durable.run("Question 1", run_id="run-a")
        await durable.run("Question 2", run_id="run-b")

        assert agent.run.call_count == 2

    async def test_auto_generated_run_id(self, wf):
        agent, _ = _mock_agent(output="Tokyo")
        durable = DurableAgent(agent, wf)

        result = await durable.run("Capital of Japan?")
        assert result.output == "Tokyo"

    async def test_custom_name(self, wf):
        agent, _ = _mock_agent(name="original")
        durable = DurableAgent(agent, wf, name="custom-name")
        assert durable.name == "custom-name"

    async def test_deps_forwarded(self, wf):
        agent, _ = _mock_agent()
        durable = DurableAgent(agent, wf)

        deps = {"db": "connection"}
        await durable.run("Query", run_id="deps-test", deps=deps)

        call_kwargs = agent.run.call_args
        assert call_kwargs[1].get("deps") == deps


class TestTaskConfig:
    async def test_custom_retries(self, wf):
        agent, _ = _mock_agent()
        durable = DurableAgent(
            agent,
            wf,
            model_task_config=TaskConfig(retries=7),
        )
        assert durable._model_retries == 7

    async def test_custom_backoff(self, wf):
        custom_backoff = exponential(base=3, max=90)
        agent, _ = _mock_agent()
        durable = DurableAgent(
            agent,
            wf,
            model_task_config=TaskConfig(backoff=custom_backoff),
        )
        assert durable._model_backoff is custom_backoff

    async def test_default_config(self, wf):
        agent, _ = _mock_agent()
        durable = DurableAgent(agent, wf)
        assert durable._model_retries == 3
        assert durable._tool_retries == 2


class TestDurableAgentSignal:
    async def test_signal_integration(self, wf):
        agent, _ = _mock_agent(output="reviewed")
        durable = DurableAgent(agent, wf)

        @wf.workflow(id="signal-test")
        async def workflow_with_signal():
            result = await durable.run("Review this", run_id="signal-inner")

            await wf.complete("signal-test", "approval", {"ok": True})
            approval = await durable.signal("approval")
            return {"result": result.output, "approval": approval}

        out = await workflow_with_signal.run("signal-test")
        assert out["approval"] == {"ok": True}


class TestDurableTool:
    async def test_durable_tool_decorator(self, wf):
        @durable_tool(wf, retries=3)
        async def my_tool(query: str) -> str:
            return f"result for {query}"

        @wf.workflow(id="tool-test")
        async def test_wf():
            return await my_tool("test query")

        result = await test_wf.run("tool-test")
        assert result == "result for test query"

    async def test_durable_tool_caches(self, wf):
        call_count = 0

        @durable_tool(wf, retries=1)
        async def counting_tool(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        @wf.workflow(id="tool-cache-test")
        async def test_wf():
            return await counting_tool(5)

        r1 = await test_wf.run("tool-cache-test")
        assert r1 == 10
        assert call_count == 1

        r2 = await test_wf.run("tool-cache-test")
        assert r2 == 10
        assert call_count == 1

    async def test_durable_tool_outside_workflow(self, wf):
        @durable_tool(wf)
        async def plain_tool(x: int) -> int:
            return x + 1

        result = await plain_tool(10)
        assert result == 11


class TestDurablePipeline:
    async def test_pipeline_decorator(self, wf):
        @wf.task
        async def step_a(x: int) -> int:
            return x + 1

        @wf.task
        async def step_b(x: int) -> int:
            return x * 2

        @durable_pipeline(wf, id="pipe-{n}")
        async def my_pipeline(n: int) -> int:
            a = await step_a(n)
            return await step_b(a)

        result = await my_pipeline(n=5)
        assert result == 12  # (5+1)*2

    async def test_pipeline_with_loop(self, wf):
        @wf.task
        async def process_item(item: str) -> str:
            return f"processed-{item}"

        @durable_pipeline(wf, id="loop-pipe-{batch}")
        async def batch_pipeline(batch: str) -> list:
            items = ["a", "b", "c"]
            results = []
            for i, item in enumerate(items):
                r = await process_item(item, step_id=f"item-{i}")
                results.append(r)
            return results

        result = await batch_pipeline(batch="test")
        assert result == ["processed-a", "processed-b", "processed-c"]


class TestHelpers:
    def test_run_id_with_explicit_id(self):
        rid = _run_id_for_agent("geo", "prompt", "my-id")
        assert rid == "my-id"

    def test_run_id_auto_generated(self):
        rid = _run_id_for_agent("geo", "What is the capital?", None)
        assert rid.startswith("agent-geo-")
        assert len(rid) > len("agent-geo-")

    def test_run_id_deterministic(self):
        rid1 = _run_id_for_agent("geo", "same prompt", None)
        rid2 = _run_id_for_agent("geo", "same prompt", None)
        assert rid1 == rid2

    def test_run_id_different_for_different_prompts(self):
        rid1 = _run_id_for_agent("geo", "prompt A", None)
        rid2 = _run_id_for_agent("geo", "prompt B", None)
        assert rid1 != rid2


class TestAgentRunResult:
    def test_output_from_live_result(self):
        mock = MagicMock()
        mock.output = "hello"
        wrapper = _AgentRunResult(mock)
        assert wrapper.output == "hello"

    def test_output_from_dict(self):
        wrapper = _AgentRunResult({"output": "from dict"})
        assert wrapper.output == "from dict"

    def test_output_from_plain_value(self):
        wrapper = _AgentRunResult("plain")
        assert wrapper.output == "plain"

    def test_repr(self):
        mock = MagicMock()
        mock.output = "test"
        wrapper = _AgentRunResult(mock)
        assert "test" in repr(wrapper)


class TestSerializeMessages:
    def test_serialize_pydantic_model(self):
        mock = MagicMock()
        mock.model_dump.return_value = {"role": "user", "content": "hi"}
        type(mock).__name__ = "ModelRequest"

        result = _serialize_messages([mock])
        assert result[0]["role"] == "user"
        assert result[0]["__type__"] == "ModelRequest"

    def test_serialize_dict(self):
        result = _serialize_messages([{"key": "value"}])
        assert result[0] == {"key": "value"}

    def test_serialize_other(self):
        result = _serialize_messages([42])
        assert "__repr__" in result[0]


class TestIntegration:
    async def test_full_multi_agent_pipeline(self, wf):
        execution_log = []

        @wf.task
        async def plan(topic: str) -> dict:
            execution_log.append("plan")
            return {"queries": ["q1", "q2", "q3"]}

        @wf.task(retries=2)
        async def search(query: str) -> str:
            execution_log.append(f"search-{query}")
            return f"result-{query}"

        @wf.task
        async def summarize(findings: list) -> str:
            execution_log.append("summarize")
            return f"Summary of {len(findings)} findings"

        @wf.workflow(id="integration-{tid}")
        async def pipeline(tid: str) -> str:
            p = await plan("test topic")
            findings = []
            for i, q in enumerate(p["queries"]):
                r = await search(q, step_id=f"search-{i}")
                findings.append(r)
            return await summarize(findings)

        result = await pipeline(tid="test-1")
        assert result == "Summary of 3 findings"
        assert execution_log == [
            "plan",
            "search-q1",
            "search-q2",
            "search-q3",
            "summarize",
        ]

        execution_log.clear()
        result = await pipeline(tid="test-1")
        assert result == "Summary of 3 findings"
        assert execution_log == []

    async def test_durable_agent_with_mock_agent(self, wf):
        agent, _ = _mock_agent(output="42")
        durable = DurableAgent(agent, wf, name="calculator")

        r1 = await durable.run("What is 6*7?", run_id="calc-1")
        assert r1.output == "42"
        assert agent.run.call_count == 1

        r2 = await durable.run("What is 6*7?", run_id="calc-1")
        assert r2.output == "42"
        assert agent.run.call_count == 1
