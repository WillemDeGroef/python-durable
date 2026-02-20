"""
examples.py — durable library usage examples.

Run with: python examples.py
"""

import asyncio
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from durable import Workflow
from durable.backoff import constant, exponential, linear

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

wf = Workflow("examples", db="sqlite:///examples.db")


# ---------------------------------------------------------------------------
# Example 1 — Basic pipeline
# ---------------------------------------------------------------------------


@dataclass
class User:
    id: str
    email: str
    name: str


@dataclass
class Invoice:
    user_id: str
    amount: float
    items: list[str]


@wf.task
async def fetch_user(user_id: str) -> dict:
    print(f"  [fetch_user] fetching user {user_id}...")
    await asyncio.sleep(0.1)  # simulate I/O
    return {"id": user_id, "email": f"{user_id}@example.com", "name": "Alice"}


@wf.task
async def build_invoice(user: dict) -> dict:
    print(f"  [build_invoice] building for {user['name']}...")
    await asyncio.sleep(0.1)
    return {"user_id": user["id"], "amount": 49.99, "items": ["Pro Plan"]}


@wf.task
async def send_email(user: dict, invoice: dict) -> None:
    print(f"  [send_email] sending to {user['email']} for ${invoice['amount']}")
    await asyncio.sleep(0.1)


@wf.workflow(id="process-order-{order_id}")
async def process_order(order_id: str) -> None:
    user = await fetch_user(order_id)
    invoice = await build_invoice(user)
    await send_email(user, invoice)


# ---------------------------------------------------------------------------
# Example 2 — Flaky API with aggressive retries
# ---------------------------------------------------------------------------

_call_count = 0


@wf.task(retries=5, backoff=exponential(base=2, max=30))
async def call_flaky_api(endpoint: str) -> dict:
    global _call_count
    _call_count += 1
    if _call_count < 3:
        raise ConnectionError(f"API down (attempt {_call_count})")
    print(f"  [call_flaky_api] succeeded on attempt {_call_count}")
    return {"status": "ok", "data": [1, 2, 3]}


@wf.workflow(id="flaky-{job_id}")
async def flaky_pipeline(job_id: str) -> dict:
    return await call_flaky_api("/data")


# ---------------------------------------------------------------------------
# Example 3 — Loop with explicit step_id
# ---------------------------------------------------------------------------


@wf.task(retries=2, backoff=constant(1))
async def push_record(record: dict) -> bool:
    print(f"  [push_record] pushing {record['id']}...")
    await asyncio.sleep(0.05)
    return True


@wf.task
async def post_summary(count: int) -> None:
    print(f"  [post_summary] all done — pushed {count} records")


@wf.workflow(id="crm-sync-{batch_id}")
async def sync_to_crm(batch_id: str) -> None:
    # Records loaded with plain Python — not every function needs to be a task
    records = [{"id": f"rec-{i}", "value": i * 10} for i in range(5)]

    for i, record in enumerate(records):
        # step_id disambiguates repeated calls to the same task in a loop.
        # If the workflow crashes mid-loop, only the remaining records are pushed.
        await push_record(record, step_id=f"push-record-{i}")

    await post_summary(len(records))


# ---------------------------------------------------------------------------
# Example 4 — Multiple backoff strategies side by side
# ---------------------------------------------------------------------------


@wf.task(retries=3, backoff=exponential(base=2, max=60))  # 2s, 4s, 8s
async def sync_to_warehouse(row: dict) -> None:
    print(f"  [sync_to_warehouse] writing row {row['id']}")


@wf.task(retries=3, backoff=linear(start=2, step=3))  # 2s, 5s, 8s
async def notify_slack(message: str) -> None:
    print(f"  [notify_slack] {message}")


@wf.task(retries=5, backoff=constant(5))  # always 5s
async def send_sms(number: str, body: str) -> None:
    print(f"  [send_sms] → {number}: {body}")


# ---------------------------------------------------------------------------
# Example 5 — Explicit run ID via .run()
# ---------------------------------------------------------------------------


@wf.workflow(id="report-{date}")
async def generate_report(date: str) -> dict:
    data = await fetch_user("analyst-1")  # reusing tasks across workflows!
    return {"generated_for": date, "by": data["name"]}


# ---------------------------------------------------------------------------
# Example 6 — Pydantic models as task parameters
#
# The checkpoint store serializes results with json.dumps / json.loads, so
# task return values must be JSON-serializable (dicts, lists, primitives).
#
# Pattern: tasks return model.model_dump(), downstream tasks reconstruct
# with Model.model_validate(). This keeps checkpoints portable and lets
# workflows resume cleanly after a crash.
# ---------------------------------------------------------------------------


class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float


class Currency(StrEnum):
    EUR = "EUR"
    USD = "USD"


class InvoiceDraft(BaseModel):
    customer_name: str
    currency: Currency
    lines: list[LineItem]


class ValidatedInvoice(BaseModel):
    customer_name: str
    currency: Currency
    lines: list[LineItem]
    total: float
    reference: str


@wf.task
async def validate_invoice(draft: InvoiceDraft) -> dict:
    """Accept a Pydantic model, return a dict for checkpoint storage."""
    print(
        f"  [validate_invoice] validating {len(draft.lines)} line(s) for {draft.customer_name}"
    )
    total = sum(line.quantity * line.unit_price for line in draft.lines)
    validated = ValidatedInvoice(
        customer_name=draft.customer_name,
        currency=draft.currency,
        lines=draft.lines,
        total=round(total, 2),
        reference=f"INV-{hash(draft.customer_name) % 10000:04d}",
    )
    # .model_dump() → plain dict that json.dumps can serialize
    return validated.model_dump()


@wf.task
async def book_invoice(invoice_data: dict) -> dict:
    """Reconstruct the Pydantic model from the checkpointed dict."""
    invoice = ValidatedInvoice.model_validate(invoice_data)
    print(
        f"  [book_invoice] booking {invoice.reference}: "
        f"{invoice.currency} {invoice.total:.2f} for {invoice.customer_name}"
    )
    return {"reference": invoice.reference, "status": "booked"}


@wf.workflow(id="invoice-{customer}")
async def process_invoice(customer: str, draft: InvoiceDraft) -> dict:
    validated = await validate_invoice(draft)
    return await book_invoice(validated)


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


async def main():
    print("\n── Example 1: Basic order pipeline ──")
    await process_order(order_id="ord-42")
    print("  (run again → all steps replayed from cache)")
    await process_order(order_id="ord-42")

    print("\n── Example 2: Flaky API with retries ──")
    result = await flaky_pipeline(job_id="job-1")
    print(f"  result: {result}")

    print("\n── Example 3: Loop with step_id ──")
    await sync_to_crm(batch_id="batch-2024-01")
    print("  (run again → all records skipped, already pushed)")
    await sync_to_crm(batch_id="batch-2024-01")

    print("\n── Example 4: Explicit run ID ──")
    report = await generate_report.run("weekly-report-custom-id", date="2024-01-15")
    print(f"  report: {report}")

    print("\n── Example 6: Pydantic models as task params ──")
    draft = InvoiceDraft(
        customer_name="Acme Corp",
        currency=Currency.EUR,
        lines=[
            LineItem(description="Consulting", quantity=10, unit_price=150.0),
            LineItem(description="Travel expenses", quantity=1, unit_price=340.50),
        ],
    )
    booking = await process_invoice(customer="acme", draft=draft)
    print(f"  result: {booking}")
    print("  (run again → all steps replayed from cache)")
    booking = await process_invoice(customer="acme", draft=draft)
    print(f"  result: {booking}")

    print("\n✓ All examples complete. Check examples.db to see the checkpoint store.")


if __name__ == "__main__":
    asyncio.run(main())
