"""Live field data from Members 2 and 3, over HTTP (CLAUDE.md: never import another member's code).

Calls go through httpx's ASGI transport against this same FastAPI app (as Member 3 does), so they use the real
routes without a network hop. Results are cached briefly (15-30 s; the field and season for 10 minutes).
"""

from __future__ import annotations

import time
from typing import Protocol

import httpx


class LiveError(RuntimeError):
    """A Member 2 / 3 route failed or is unavailable."""


class Live(Protocol):
    async def kpis(self) -> dict: ...
    async def alerts(self) -> dict: ...
    async def recommendations(self) -> dict: ...
    async def field(self) -> dict: ...
    async def season(self) -> dict: ...


class LiveHttp:
    # Short for the live data: Members 2 and 3 cache their own results per readings version, so a repeat call is
    # cheap, and a change in the readings (e.g. the demo scenario) reaches the chat within seconds.
    TTL = {"/api/m3/kpis": 15, "/api/m3/alerts": 15, "/api/m3/field": 600, "/api/m2/recommendations": 30,
           "/api/m2/season": 600}

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self._cache: dict[str, tuple[float, dict]] = {}

    @classmethod
    def for_app(cls, app, timeout_s: float = 120.0) -> "LiveHttp":
        return cls(httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://soilmind.internal",
                                     timeout=timeout_s))

    async def _get(self, path: str, **params) -> dict:
        hit = self._cache.get(path)
        if hit and time.monotonic() - hit[0] < self.TTL.get(path, 60):
            return hit[1]
        try:
            r = await self.client.get(path, params=params or None)
        except Exception as exc:  # noqa: BLE001 - any transport failure means "unavailable"
            raise LiveError(f"{path}: {exc.__class__.__name__}: {exc}") from exc
        if r.status_code != 200:
            raise LiveError(f"{path}: HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        self._cache[path] = (time.monotonic(), data)
        return data

    async def kpis(self) -> dict:
        return await self._get("/api/m3/kpis")

    async def alerts(self) -> dict:
        return await self._get("/api/m3/alerts")

    async def field(self) -> dict:
        return await self._get("/api/m3/field")

    async def season(self) -> dict:
        return await self._get("/api/m2/season")

    async def recommendations(self) -> dict:
        return await self._get("/api/m2/recommendations", node="plot")
