"""Structured step log (JSON lines). This is the permanent history System 2 replans from."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


class StepLog:
    def __init__(self, path: str | Path | None = None, keep: int = 200):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recent: deque[dict] = deque(maxlen=keep)

    def write(self, record: dict[str, Any]) -> None:
        self._recent.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=_default) + "\n")

    def tail(self, n: int = 20) -> list[dict]:
        return list(self._recent)[-n:]

    def history_text(self, n: int = 20) -> str:
        """Compact history for System 2: one line per step."""
        lines = []
        for record in self.tail(n):
            lines.append(
                f"step {record.get('step')}: route={record.get('route')} "
                f"nav={record.get('navigation')} action={record.get('action')} -> {record.get('outcome')}"
            )
        return "\n".join(lines) if lines else "(no steps yet)"

    def clear_recent(self) -> None:
        self._recent.clear()


def _default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)
