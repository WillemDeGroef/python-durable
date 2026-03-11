"""
Redis-backed store with automatic key expiration.

Requires the ``redis`` extra: ``uv sync --extra redis``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from redis.asyncio import Redis

from .store import Store

_WRAP_KEY = "v"


def _wrap(value: Any) -> str:
    return json.dumps({_WRAP_KEY: value})


def _unwrap(raw: str | bytes) -> Any:
    return json.loads(raw)[_WRAP_KEY]


def _step_key(prefix: str, run_id: str, step_id: str) -> str:
    tag = hashlib.sha256(f"{run_id}:{step_id}".encode()).hexdigest()[:16]
    return f"{prefix}:step:{tag}"


def _run_key(prefix: str, run_id: str) -> str:
    return f"{prefix}:run:{run_id}"


def _sig_key(prefix: str, run_id: str, name: str) -> str:
    tag = hashlib.sha256(f"{run_id}:{name}".encode()).hexdigest()[:16]
    return f"{prefix}:sig:{tag}"


class RedisStore(Store):
    """Async Redis store with TTL-based auto-expiration."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        ttl: int = 86_400,
        prefix: str = "durable",
    ) -> None:
        self._url = url
        self._ttl = ttl
        self._prefix = prefix
        self._redis: Redis | None = None

    async def setup(self) -> None:
        if self._redis is None:
            self._redis = Redis.from_url(self._url, decode_responses=True)

    def _client(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("RedisStore.setup() must be called before use")
        return self._redis

    async def get_step(self, run_id: str, step_id: str) -> tuple[bool, Any]:
        raw = await self._client().get(_step_key(self._prefix, run_id, step_id))
        if raw is None:
            return False, None
        return True, _unwrap(raw)

    async def set_step(
        self, run_id: str, step_id: str, result: Any, attempt: int = 1
    ) -> None:
        payload = json.dumps({"v": result, "attempt": attempt})
        key = _step_key(self._prefix, run_id, step_id)
        client = self._client()
        if self._ttl > 0:
            await client.setex(key, self._ttl, payload)
        else:
            await client.set(key, payload)

    async def mark_run_done(self, run_id: str) -> None:
        key = _run_key(self._prefix, run_id)
        client = self._client()
        if self._ttl > 0:
            await client.setex(key, self._ttl, "done")
        else:
            await client.set(key, "done")

    async def mark_run_failed(self, run_id: str, error: str) -> None:
        key = _run_key(self._prefix, run_id)
        payload = json.dumps({"status": "failed", "error": error})
        client = self._client()
        if self._ttl > 0:
            await client.setex(key, self._ttl, payload)
        else:
            await client.set(key, payload)

    async def get_signal(self, run_id: str, name: str) -> tuple[bool, Any]:
        raw = await self._client().get(_sig_key(self._prefix, run_id, name))
        if raw is None:
            return False, None
        return True, json.loads(raw)

    async def set_signal(self, run_id: str, name: str, payload: Any) -> bool:
        key = _sig_key(self._prefix, run_id, name)
        client = self._client()
        created = await client.set(key, json.dumps(payload), nx=True)
        if created and self._ttl > 0:
            await client.expire(key, self._ttl)
        return bool(created)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
