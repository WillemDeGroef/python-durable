"""
pydantic_ai_example.py — Using pydantic-ai agents with durable workflows.

Same idea as the Temporal integration (https://ai.pydantic.dev/durable_execution/temporal/)
but without infrastructure: no server, no workers — just decorators and SQLite.

Requires:
    uv sync --extra examples
    export OPENAI_API_KEY=sk-...

Run with:
    uv run python examples/pydantic_ai_example.py
"""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

from durable import Workflow
from durable.backoff import exponential

wf = Workflow("ai-agent", db="ai_agent.db")


# ---------------------------------------------------------------------------
# Example 1 — Simple durable agent (direct equivalent of the Temporal example)
#
# Temporal version needs: server, worker, PydanticAIPlugin, TemporalAgent,
# PydanticAIWorkflow subclass, task queue, and client.connect().
#
# Durable version: two decorators.
# ---------------------------------------------------------------------------

geography_agent = Agent(
    "openai:gpt-4o",
    instructions="You're an expert in geography. Give concise answers.",
)


@wf.task(retries=2, backoff=exponential(base=2, max=30))
async def ask_geography(prompt: str) -> str:
    """Run the geography agent — checkpointed so a crash won't repeat the LLM call."""
    result = await geography_agent.run(prompt)
    return result.output


@wf.workflow(id="geo-{question_id}")
async def geography_question(question_id: str, prompt: str) -> str:
    return await ask_geography(prompt)


# ---------------------------------------------------------------------------
# Example 2 — Multi-step research pipeline with structured output
#
# Shows the real value of durable execution with AI agents:
#   1. Planner agent produces a structured research plan (list of queries)
#   2. Each query executes as an individually checkpointed task
#   3. Summarizer agent synthesizes the findings
#
# If the workflow crashes after the plan and 3/5 queries, on restart:
#   - plan is replayed instantly from SQLite
#   - 3 completed queries are replayed
#   - remaining 2 queries execute
#   - summarizer runs on all 5 results
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    queries: list[str]


planner_agent = Agent(
    "openai:gpt-4o",
    output_type=ResearchPlan,
    instructions=(
        "Given a research topic, produce 3-5 specific search queries "
        "that would help investigate the topic thoroughly."
    ),
)

summarizer_agent = Agent(
    "openai:gpt-4o",
    instructions="Synthesize research findings into a clear, concise summary.",
)


@wf.task(retries=2, backoff=exponential(base=2, max=30))
async def plan_research(topic: str) -> dict:
    """Agent produces a structured plan — .model_dump() for JSON checkpoint."""
    result = await planner_agent.run(topic)
    return result.output.model_dump()


@wf.task(retries=3, backoff=exponential(base=2, max=60))
async def execute_query(query: str) -> str:
    """
    Execute a single research query.

    In production this would call a search API (e.g. Tavily, Brave, SerpAPI).
    Simulated here so the example runs without extra API keys.
    """
    print(f"  [execute_query] searching: {query}")
    await asyncio.sleep(0.1)  # simulate I/O
    return f"Results for '{query}': [simulated search results would go here]"


@wf.task(retries=2, backoff=exponential(base=2, max=30))
async def summarize_findings(topic: str, findings: list[str]) -> str:
    """Agent synthesizes all findings into a final answer."""
    prompt = f"Topic: {topic}\n\nFindings:\n" + "\n".join(f"- {f}" for f in findings)
    result = await summarizer_agent.run(prompt)
    return result.output


@wf.workflow(id="research-{topic_id}")
async def research_pipeline(topic_id: str, topic: str) -> str:
    # Step 1: Agent plans the research
    plan = await plan_research(topic)
    print(f"  [plan] {len(plan['queries'])} queries: {plan['queries']}")

    # Step 2: Execute each query — individually checkpointed via step_id
    findings = []
    for i, query in enumerate(plan["queries"]):
        result = await execute_query(query, step_id=f"query-{i}")
        findings.append(result)

    # Step 3: Agent synthesizes the results
    return await summarize_findings(topic, findings)


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


async def main():
    print("\n── Example 1: Simple durable agent (geography) ──")
    answer = await geography_question(
        question_id="q1",
        prompt="What is the capital of Mexico?",
    )
    print(f"  answer: {answer}")
    print("  (run again → replayed from cache, no LLM call)")
    answer = await geography_question(
        question_id="q1",
        prompt="What is the capital of Mexico?",
    )
    print(f"  answer: {answer}")

    print("\n── Example 2: Multi-step research pipeline ──")
    summary = await research_pipeline(
        topic_id="ai-safety",
        topic="Recent developments in AI safety research",
    )
    print(f"  summary: {summary[:200]}...")

    print("\n✓ All examples complete. Check ai_agent.db to see the checkpoint store.")


if __name__ == "__main__":
    asyncio.run(main())
