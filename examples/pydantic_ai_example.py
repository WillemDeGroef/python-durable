"""Durable pydantic-ai agents with python-durable.

Requires:
    uv sync --all-extras
    export OPENAI_API_KEY=sk-...

Run:
    uv run python examples/pydantic_ai_example.py
"""

import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

from durable import Workflow
from durable.backoff import exponential
from durable.pydantic_ai import DurableAgent, TaskConfig, durable_pipeline, durable_tool


wf = Workflow("ai-agents", db="ai_agents.db")


# Example 1 — DurableAgent

geography_agent = Agent(
    "openai:gpt-5.2",
    instructions="You're an expert in geography. Give concise answers.",
    name="geography",
)

durable_geo = DurableAgent(geography_agent, wf)


async def example_1():
    print("\n── Example 1: DurableAgent (simple wrapper) ──")

    result = await durable_geo.run(
        "What is the capital of Mexico?",
        run_id="geo-mexico-capital",
    )
    print(f"  answer: {result.output}")

    result = await durable_geo.run(
        "What is the capital of Mexico?",
        run_id="geo-mexico-capital",
    )
    print(f"  replayed: {result.output}")


# Example 2 — DurableAgent with custom retry config

coding_agent = Agent(
    "openai:gpt-5.2",
    instructions="You're an expert Python developer. Be concise.",
    name="coder",
)

durable_coder = DurableAgent(
    coding_agent,
    wf,
    model_task_config=TaskConfig(retries=5, backoff=exponential(base=2, max=120)),
)


async def example_2():
    print("\n── Example 2: DurableAgent with retry config ──")
    result = await durable_coder.run(
        "Write a one-liner to flatten a nested list in Python.",
        run_id="code-flatten",
    )
    print(f"  answer: {result.output}")


# Example 3 — Multi-agent pipeline with fine-grained checkpointing


class ResearchPlan(BaseModel):
    queries: list[str]


planner_agent = Agent(
    "openai:gpt-5.2",
    output_type=ResearchPlan,
    instructions=(
        "Given a research topic, produce 3-5 specific search queries "
        "that would help investigate the topic thoroughly."
    ),
    name="planner",
)

summarizer_agent = Agent(
    "openai:gpt-5.2",
    instructions="Synthesize research findings into a clear, concise summary.",
    name="summarizer",
)


@wf.task(retries=2, backoff=exponential(base=2, max=30))
async def plan_research(topic: str) -> dict:
    result = await planner_agent.run(topic)
    return result.output.model_dump()


@wf.task(retries=3, backoff=exponential(base=2, max=60))
async def execute_query(query: str) -> str:
    print(f"    [search] {query}")
    await asyncio.sleep(0.1)
    return f"Results for '{query}': [simulated results]"


@wf.task(retries=2, backoff=exponential(base=2, max=30))
async def summarize_findings(topic: str, findings: list[str]) -> str:
    prompt = f"Topic: {topic}\n\nFindings:\n" + "\n".join(f"- {f}" for f in findings)
    result = await summarizer_agent.run(prompt)
    return result.output


@durable_pipeline(wf, id="research-{topic_id}")
async def research_pipeline(topic_id: str, topic: str) -> str:
    plan = await plan_research(topic)
    print(f"  [plan] {len(plan['queries'])} queries")

    findings = []
    for i, query in enumerate(plan["queries"]):
        result = await execute_query(query, step_id=f"query-{i}")
        findings.append(result)

    return await summarize_findings(topic, findings)


async def example_3():
    print("\n── Example 3: Multi-agent research pipeline ──")
    summary = await research_pipeline(
        topic_id="ai-safety",
        topic="Recent developments in AI safety research",
    )
    print(f"  summary: {summary[:200]}...")


# Example 4 — Human-in-the-loop with AI review + signals

review_agent = Agent(
    "openai:gpt-5.2",
    instructions=(
        "You review invoice data for compliance. "
        "Return a brief assessment and risk level (low/medium/high)."
    ),
    name="reviewer",
)


@wf.task(retries=2)
async def ai_review_invoice(invoice: dict) -> dict:
    prompt = f"Review this invoice for compliance:\n{invoice}"
    result = await review_agent.run(prompt)
    return {"assessment": result.output, "invoice_id": invoice["id"]}


@wf.task
async def process_approved_invoice(invoice: dict, approval: dict) -> dict:
    print(f"    [process] Invoice {invoice['id']} approved by {approval.get('approver')}")
    return {"status": "processed", "invoice_id": invoice["id"]}


@wf.workflow(id="invoice-review-{invoice_id}")
async def invoice_review_workflow(invoice_id: str) -> dict:
    invoice = {"id": invoice_id, "vendor": "ACME Corp", "amount": 15_000}

    review = await ai_review_invoice(invoice)
    print(f"    [review] {review['assessment'][:100]}...")

    print("    [waiting] Waiting for manager approval...")
    approval = await wf.signal("manager-approval")

    if not approval.get("approved"):
        return {"status": "rejected", "invoice_id": invoice_id}

    return await process_approved_invoice(invoice, approval)


async def example_4():
    print("\n── Example 4: AI review + human approval ──")

    async def deliver_approval():
        await asyncio.sleep(0.5)
        await wf.complete(
            "invoice-review-inv-42",
            "manager-approval",
            {"approved": True, "approver": "alice@example.com"},
        )

    result, _ = await asyncio.gather(
        invoice_review_workflow(invoice_id="inv-42"),
        deliver_approval(),
    )
    print(f"  result: {result}")


# Example 5 — @durable_tool for I/O-heavy tool functions


@durable_tool(wf, retries=3, backoff=exponential(base=2, max=60))
async def fetch_weather(city: str) -> str:
    print(f"    [weather] fetching for {city}")
    await asyncio.sleep(0.1)
    return f"Weather in {city}: 22C, partly cloudy"


@durable_tool(wf, retries=3)
async def fetch_news(topic: str) -> str:
    print(f"    [news] fetching for {topic}")
    await asyncio.sleep(0.1)
    return f"Latest news on {topic}: [simulated headlines]"


@wf.workflow(id="daily-briefing-{user_id}")
async def daily_briefing(user_id: str) -> str:
    weather = await fetch_weather("Brussels", step_id="weather")
    news = await fetch_news("e-invoicing Belgium", step_id="news")

    agent = Agent(
        "openai:gpt-5.2",
        instructions="Create a brief daily briefing from the provided data.",
    )
    result = await agent.run(f"Weather: {weather}\nNews: {news}")
    return result.output


async def example_5():
    print("\n── Example 5: @durable_tool for individual tool checkpointing ──")
    briefing = await daily_briefing(user_id="willem")
    print(f"  briefing: {briefing[:200]}...")


async def main():
    print("=" * 60)
    print("  python-durable x pydantic-ai integration examples")
    print("  (set OPENAI_API_KEY to run all examples)")
    print("=" * 60)

    try:
        await example_1()
        await example_2()
        await example_3()
        await example_4()
        await example_5()
    except Exception as e:
        if "API key" in str(e) or "OPENAI_API_KEY" in str(e):
            print(f"\n  Skipped (no API key): {e}")
        else:
            raise

    print("\n  All examples complete. Check ai_agents.db for checkpoint data.")


if __name__ == "__main__":
    asyncio.run(main())
