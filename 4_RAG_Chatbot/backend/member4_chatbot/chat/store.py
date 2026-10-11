"""Chat sessions (last 3 turns) and escalations (questions forwarded to an agriculture officer).

Escalations are stored in MongoDB's `escalations` collection; if the database is down they are kept in memory
(and still returned by GET /api/m4/escalations), so a refusal never fails.
"""

from __future__ import annotations

import logging
import secrets
from collections import defaultdict, deque
from datetime import UTC, datetime

log = logging.getLogger("soilmind.m4")
HISTORY_TURNS = 3
ESCALATIONS = "escalations"


class Sessions:
    def __init__(self):
        self._turns: dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_TURNS))

    def history(self, session_id: str) -> list[dict]:
        return list(self._turns[session_id])

    def add(self, session_id: str, turn: dict) -> None:
        self._turns[session_id].append(turn)


class Escalations:
    def __init__(self, db_getter=None):
        self._db_getter = db_getter
        self._memory: list[dict] = []

    def _db(self):
        if self._db_getter is None:
            return None
        try:
            return self._db_getter()
        except Exception:  # noqa: BLE001
            return None

    async def add(self, record: dict) -> str:
        ref = "ESC-" + secrets.token_hex(3).upper()
        doc = {"ref": ref, "created_at": datetime.now(UTC), "status": "open", **record}
        db = self._db()
        if db is not None:
            try:
                await db[ESCALATIONS].insert_one(dict(doc))
                return ref
            except Exception as exc:  # noqa: BLE001 - fall back to memory, never fail the chat
                log.warning("escalation not stored in MongoDB (%s); kept in memory", exc)
        self._memory.append(doc)
        return ref

    async def recent(self, limit: int = 50) -> list[dict]:
        out = list(self._memory)
        db = self._db()
        if db is not None:
            try:
                out += await db[ESCALATIONS].find({}, {"_id": 0}).sort("created_at", -1).to_list(limit)
            except Exception:  # noqa: BLE001
                pass
        out.sort(key=lambda d: d["created_at"], reverse=True)
        return out[:limit]
