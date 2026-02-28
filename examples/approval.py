"""
approval.py — Human-in-the-loop approval workflow using durable signals.

Demonstrates how a workflow can durably wait for external input (e.g. a
human approval via a web endpoint), survive crashes, and resume cleanly.

Run with:
    uvicorn examples.approval:app --reload

Then:
    1. POST /orders/ord-100        → starts the workflow, blocks at approval
    2. POST /approve/process-order-ord-100/manager-approval
       body: {"approved": true, "approver": "alice@example.com"}
                                    → delivers the signal, workflow continues
    3. POST /orders/ord-100        → re-run: everything replays from cache
"""

import asyncio
import logging

from fastapi import FastAPI

from durable import Workflow

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Durable Approval Example")
wf = Workflow("approvals", db="sqlite:///approval.db")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@wf.task
async def validate_order(order_id: str) -> dict:
    print(f"  [validate_order] validating {order_id}...")
    await asyncio.sleep(0.1)
    return {"order_id": order_id, "total": 12_500.00, "items": 3}


@wf.task
async def fulfill_order(order: dict, approval: dict) -> dict:
    print(
        f"  [fulfill_order] fulfilling {order['order_id']}, approved by {approval.get('approver')}"
    )
    await asyncio.sleep(0.1)
    return {"status": "fulfilled", "order_id": order["order_id"]}


# ---------------------------------------------------------------------------
# Workflow — blocks at wf.signal() until a human approves
# ---------------------------------------------------------------------------


@wf.workflow(id="process-order-{order_id}")
async def process_order(order_id: str) -> dict:
    order = await validate_order(order_id)
    print("  [process_order] order validated, waiting for manager approval...")

    # This durably blocks until someone calls wf.complete()
    approval = await wf.signal("manager-approval")

    if not approval.get("approved"):
        return {"status": "rejected", "order_id": order_id}

    return await fulfill_order(order, approval)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@app.post("/orders/{order_id}")
async def start_order(order_id: str):
    """Start (or re-run) the order workflow. Blocks until approval signal arrives."""
    result = await process_order(order_id=order_id)
    return result


@app.post("/approve/{run_id}/{signal_name}")
async def approve(run_id: str, signal_name: str, payload: dict):
    """
    Deliver a signal to a waiting workflow.

    Example:
        POST /approve/process-order-ord-100/manager-approval
        {"approved": true, "approver": "alice@example.com"}
    """
    ok = await wf.complete(run_id, signal_name, payload)
    if not ok:
        return {"status": "already_completed", "run_id": run_id, "signal": signal_name}
    return {"status": "delivered", "run_id": run_id, "signal": signal_name}
