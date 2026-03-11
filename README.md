# durable

Lightweight workflow durability for Python. Make any async workflow resumable after crashes with just a decorator.

Backed by SQLite out of the box; swap in Redis or any `Store` subclass for production.

## Install

```bash
pip install python-durable

# With Redis support
pip install python-durable[redis]

# With Pydantic AI integration
pip install python-durable[pydantic-ai]
```

## Quick start

```python
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
```

## How it works

1. **`@wf.task`** wraps an async function with checkpoint + retry logic. When called inside a workflow, results are persisted to the store. On re-run, completed steps return their cached result without re-executing.

2. **`@wf.workflow`** marks the entry point of a durable run. It manages a `RunContext` (via `ContextVar`) so tasks automatically know which run they belong to. The `id` parameter is a template string resolved from function arguments at call time.

3. **`Store`** is the persistence backend. `SQLiteStore` is the default (zero config, backed by aiosqlite). `RedisStore` is available for distributed setups. Subclass `Store` to use Postgres or anything else.

## Features

- **Crash recovery** — completed steps are never re-executed after a restart
- **Automatic retries** — configurable per-task with `exponential`, `linear`, or `constant` backoff
- **Signals** — durably wait for external input (approvals, webhooks, human-in-the-loop)
- **Loop support** — use `step_id` to checkpoint each iteration independently
- **Zero magic outside workflows** — tasks work as plain async functions when called without a workflow context
- **Pluggable storage** — SQLite by default, Redis built-in, or bring your own `Store`

## Signals

Workflows can pause and wait for external input using signals:

```python
@wf.workflow(id="order-{order_id}")
async def process_order(order_id: str) -> None:
    await prepare_order(order_id)
    approval = await wf.signal("manager-approval")  # pauses here
    if approval["approved"]:
        await ship_order(order_id)

# From the outside (e.g. a web handler):
await wf.complete("order-42", "manager-approval", {"approved": True})
```

Signals are durable — if the workflow crashes and restarts, a previously delivered signal replays instantly.

## Redis store

For distributed or multi-process setups, use `RedisStore` instead of the default SQLite:

```python
from durable import Workflow, RedisStore

store = RedisStore(url="redis://localhost:6379/0", ttl=86400)
wf = Workflow("my-app", db=store)
```

Keys auto-expire based on `ttl` (default: 24 hours).

## Backoff strategies

```python
from durable.backoff import exponential, linear, constant

@wf.task(retries=5, backoff=exponential(base=2, max=60))  # 2s, 4s, 8s, 16s, 32s
async def exp_task(): ...

@wf.task(retries=3, backoff=linear(start=2, step=3))      # 2s, 5s, 8s
async def linear_task(): ...

@wf.task(retries=3, backoff=constant(5))                   # 5s, 5s, 5s
async def const_task(): ...
```

## Loops with step_id

When calling the same task in a loop, pass `step_id` so each iteration is checkpointed independently:

```python
@wf.workflow(id="batch-{batch_id}")
async def process_batch(batch_id: str) -> None:
    for i, item in enumerate(items):
        await process_item(item, step_id=f"item-{i}")
```

If the workflow crashes mid-loop, only the remaining items are processed on restart.

## Pydantic AI integration

Make any [pydantic-ai](https://ai.pydantic.dev) agent durable with **zero infrastructure** — no Temporal server, no Prefect cloud, no Postgres. Just decorators and a SQLite file.

Pydantic AI natively supports three durable execution backends: **Temporal**, **DBOS**, and **Prefect**. All three require external infrastructure. `python-durable` is a fourth option that trades scale for simplicity:

| Feature | Temporal | DBOS | Prefect | **python-durable** |
|---------|----------|------|---------|-------------------|
| Infrastructure | Server + Worker | Postgres | Cloud/Server | **SQLite file** |
| Setup | Complex | Moderate | Moderate | **`pip install`** |
| Lines to wrap an agent | ~20 | ~10 | ~10 | **1** |
| Crash recovery | Yes | Yes | Yes | Yes |
| Retries + backoff | Yes | Yes | Yes | Yes |
| Human-in-the-loop signals | Yes | No | No | Yes |
| Multi-process / distributed | Yes | Yes | Yes | No (single process) |
| Production scale | Enterprise | Production | Production | **Dev / SME / CLI** |

**Best for:** prototyping, CLI tools, single-process services, SME deployments, and any situation where you want durable agents without ops overhead.

### DurableAgent

```python
from pydantic_ai import Agent
from durable import Workflow
from durable.pydantic_ai import DurableAgent, TaskConfig
from durable.backoff import exponential

wf = Workflow("my-app")
agent = Agent("openai:gpt-4o", instructions="Be helpful.", name="assistant")

durable_agent = DurableAgent(agent, wf)

result = await durable_agent.run("What is the capital of France?")
print(result.output)  # Paris

# Same run_id after crash → replayed from SQLite, no LLM call
result = await durable_agent.run("What is the capital of France?", run_id="same-id")
```

With custom retry config:

```python
durable_agent = DurableAgent(
    agent,
    wf,
    model_task_config=TaskConfig(retries=5, backoff=exponential(base=2, max=120)),
    tool_task_config=TaskConfig(retries=3),
)
```

### @durable_tool

Make individual tool functions durable:

```python
from durable.pydantic_ai import durable_tool

@durable_tool(wf, retries=3, backoff=exponential(base=2, max=60))
async def web_search(query: str) -> str:
    async with httpx.AsyncClient() as client:
        return (await client.get(f"https://api.search.com?q={query}")).text
```

### @durable_pipeline

Multi-agent workflows with per-step checkpointing. On crash, completed steps replay from the store and only remaining work executes:

```python
from durable.pydantic_ai import durable_pipeline

@durable_pipeline(wf, id="research-{topic_id}")
async def research(topic_id: str, topic: str) -> str:
    plan = await plan_research(topic)
    findings = []
    for i, query in enumerate(plan["queries"]):
        r = await search(query, step_id=f"q-{i}")
        findings.append(r)
    return await summarize(findings)
```

### Comparison with Temporal

```python
# Temporal — requires server + worker + plugin
from temporalio import workflow
from pydantic_ai.durable_exec.temporal import TemporalAgent

temporal_agent = TemporalAgent(agent)

@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self, prompt: str):
        return await temporal_agent.run(prompt)

# python-durable
from durable import Workflow
from durable.pydantic_ai import DurableAgent

wf = Workflow("my-app")
durable_agent = DurableAgent(agent, wf)
result = await durable_agent.run("Hello")
```

### Caveats

- **Tool functions** registered on the pydantic-ai agent are NOT automatically wrapped. If they perform I/O, decorate them with `@durable_tool(wf)` or `@wf.task`.
- **Streaming** (`agent.run_stream()`) is not supported in durable mode (same limitation as DBOS). Use `agent.run()`.
- **Single process** — unlike Temporal/DBOS, python-durable runs in-process. For distributed workloads, use the Redis store.

See [`examples/pydantic_ai_example.py`](examples/pydantic_ai_example.py) for five complete patterns.

## Important: JSON serialization

Task return values must be JSON-serializable (dicts, lists, strings, numbers, booleans, `None`). The store uses `json.dumps` internally.

For Pydantic models, return `.model_dump()` from tasks and reconstruct with `.model_validate()` downstream:

```python
@wf.task
async def validate_invoice(draft: InvoiceDraft) -> dict:
    validated = ValidatedInvoice(...)
    return validated.model_dump()

@wf.task
async def book_invoice(data: dict) -> dict:
    invoice = ValidatedInvoice.model_validate(data)
    ...
```

## License

MIT
