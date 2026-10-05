"""Protected core: completion checks, latency, executor limits, change check, think rule, goal age.

Read-only to the LLM. Adapters never report latency or judge completion.

Completion conditions are small JSON objects written by think into each subtask:

    {"type": "env_success"}
    {"type": "text_visible", "text": "Welcome"}
    {"type": "text_gone", "text": "Submit"}
    {"type": "value_equals", "target": "Username", "value": "alice"}
    {"type": "all", "of": [...]}   {"type": "any", "of": [...]}

Unknown or malformed conditions evaluate to False, so a subtask can never be completed by a bad condition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .schema import BBox, Target


@dataclass
class CheckState:
    targets: list[Target] = field(default_factory=list)
    env_success: Optional[bool] = None


def changed(before: Optional[np.ndarray], after: Optional[np.ndarray], threshold: float = 0.5,
            watch: Optional[list[BBox]] = None) -> bool:
    """Mean absolute pixel difference above threshold, over the watched regions or the whole frame."""
    if before is None or after is None or before.shape != after.shape:
        return True
    diff = np.abs(before.astype(np.int16) - after.astype(np.int16))
    if not watch:
        return float(diff.mean()) > threshold
    h, w = diff.shape[:2]
    for x0, y0, x1, y1 in watch:
        region = diff[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1), int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)]
        if region.size and float(region.mean()) > threshold:
            return True
    return False


class Core:
    def __init__(self, allow_irreversible: bool = False, allowed_verbs: tuple[str, ...] = (),
                 change_threshold: float = 0.5, think_below: float = 0.35, goal_max_age: int = 5):
        self.allow_irreversible = bool(allow_irreversible)
        self.allowed_verbs = tuple(allowed_verbs)
        self.change_threshold = float(change_threshold)
        self.think_below = float(think_below)
        self.goal_max_age = int(goal_max_age)
        self._start: float | None = None

    # --- change check ----------------------------------------------------
    def window_changed(self, before: Optional[np.ndarray], after: Optional[np.ndarray],
                       watch: Optional[list[BBox]] = None) -> bool:
        return changed(before, after, self.change_threshold, watch)

    # --- hard think rule --------------------------------------------------
    def force_think(self, top_probability: float) -> bool:
        return top_probability < self.think_below

    # --- goal age ---------------------------------------------------------
    def goal_valid(self, goal_tick: int, now_tick: int, window_changed_since: bool) -> bool:
        return not window_changed_since and (now_tick - goal_tick) <= self.goal_max_age

    # --- latency ---------------------------------------------------------
    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = (time.perf_counter() - self._start) * 1000.0
        self._start = None
        return elapsed

    # --- executor limits -------------------------------------------------
    def allow(self, verb: str, target: Target | None) -> tuple[bool, str]:
        if self.allowed_verbs and verb not in self.allowed_verbs:
            return False, f"verb {verb!r} is not permitted"
        if target is not None and target.reversible is False and not self.allow_irreversible:
            return False, f"{target.label!r} is irreversible and irreversible actions are disabled"
        return True, ""

    # --- completion checks -----------------------------------------------
    def check(self, condition: dict | None, state: CheckState) -> bool:
        if not isinstance(condition, dict):
            return False
        kind = condition.get("type")
        try:
            if kind == "env_success":
                return bool(state.env_success)
            if kind == "text_visible":
                return _any_label(state.targets, condition["text"])
            if kind == "text_gone":
                return not _any_label(state.targets, condition["text"])
            if kind == "value_equals":
                wanted = str(condition["value"]).strip().lower()
                return any(
                    _contains(t.label, condition["target"]) and t.value.strip().lower() == wanted for t in state.targets
                )
            if kind == "all":
                parts = condition.get("of") or []
                return bool(parts) and all(self.check(part, state) for part in parts)
            if kind == "any":
                return any(self.check(part, state) for part in condition.get("of") or [])
        except (KeyError, TypeError):
            return False
        return False


def _contains(label: str, text: str) -> bool:
    return str(text).strip().lower() in (label or "").lower()


def _any_label(targets: list[Target], text: str) -> bool:
    return any(_contains(t.label, text) or _contains(t.value, text) for t in targets)
