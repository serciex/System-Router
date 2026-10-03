"""The goal context: one per goal, holding the goal, System 2's task list and the current subtask.

Ordered so the stable part (goal, plan, current subtask) is a fixed prefix System 1 can cache across
steps. The subtask is swapped when its condition passes; the whole context is wiped when the goal
completes. System 2 replans from the step log, never from this context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Subtask:
    text: str
    condition: Optional[dict] = None
    arg: Optional[str] = None  # free-form input, e.g. text to type, written by System 2


@dataclass
class GoalContext:
    goal: str = ""
    subtasks: list[Subtask] = field(default_factory=list)
    index: int = 0
    steps_on_subtask: int = 0
    needs_replan: bool = False

    @property
    def has_plan(self) -> bool:
        return bool(self.subtasks)

    @property
    def current(self) -> Optional[Subtask]:
        return self.subtasks[self.index] if self.index < len(self.subtasks) else None

    @property
    def done(self) -> bool:
        return self.has_plan and self.index >= len(self.subtasks)

    def set_plan(self, subtasks: list[Subtask]) -> None:
        self.subtasks = list(subtasks)
        self.index = 0
        self.steps_on_subtask = 0
        self.needs_replan = False

    def advance(self) -> None:
        self.index += 1
        self.steps_on_subtask = 0

    def tick(self, stall_steps: int) -> None:
        self.steps_on_subtask += 1
        if self.steps_on_subtask > stall_steps:
            self.needs_replan = True

    def wipe(self, goal: str = "") -> None:
        self.goal = goal
        self.subtasks = []
        self.index = 0
        self.steps_on_subtask = 0
        self.needs_replan = False

    def stable_text(self) -> str:
        """The cached prefix: goal, plan, current subtask. Changes only on replan or subtask swap."""
        lines = [f"Goal: {self.goal}"]
        if not self.has_plan:
            lines.append("Plan: (none yet)")
            return "\n".join(lines)
        lines.append("Plan:")
        lines += [f"{i + 1}. {s.text}" for i, s in enumerate(self.subtasks)]
        current = self.current
        if current is not None:
            lines.append(f"Current subtask ({self.index + 1}/{len(self.subtasks)}): {current.text}")
            if current.arg:
                lines.append(f'Text to use for this subtask: "{current.arg}"')
        return "\n".join(lines)
