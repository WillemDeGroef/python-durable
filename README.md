# durable

Lightweight workflow durability for Python. Make any async workflow resumable after crashes with just a decorator.

Backed by SQLite out of the box; swap in any `Store` subclass for production.

## Install

```bash
pip install python-durable
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

3. **`Store`** is the persistence backend. `SQLiteStore` is the default (zero config, backed by aiosqlite). Subclass `Store` to use Postgres, Redis, or anything else.

## Features

- **Crash recovery** — completed steps are never re-executed after a restart
- **Automatic retries** — configurable per-task with `exponential`, `linear`, or `constant` backoff
- **Loop support** — use `step_id` to checkpoint each iteration independently
- **Zero magic outside workflows** — tasks work as plain async functions when called without a workflow context
- **Pluggable storage** — SQLite by default, bring your own `Store` for production

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
