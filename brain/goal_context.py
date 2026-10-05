"""The goal context: the goal, think's task list, the current subtask and the latest word goal.

Rendered into the prompt's goals section. The subtask is swapped when its condition passes; the word
goal is dropped when the window changes or it gets too old; everything is wiped when the goal ends.
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
    word_goal: Optional[str] = None  # from the last think
    word_goal_tick: int = 0

    def set_word_goal(self, text: Optional[str], tick: int) -> None:
        self.word_goal, self.word_goal_tick = text, tick

    def drop_word_goal(self) -> None:
        self.word_goal = None

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
        self.word_goal = None

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
