"""Think mode: token reasoning with the screen. Plans, word goals, picks, positions and code.

Outputs are JSON so they can be validated: chosen keys must be offered options, positions are clipped to
the window, and every subtask carries a completion condition the protected core can check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from body.prompt import Labeled, PromptParts, question_text
from body.schema import HOLD, KEEP, NONE, Point
from body.vocab import CORE_ACTIONS

from .goal_context import GoalContext, Subtask
from .llm import LLM, generate, parse_json

CONDITIONS = """Completion conditions (one per subtask, checked by the body):
  {"type": "text_visible", "text": "..."}        some option's label or value contains the text
  {"type": "text_gone", "text": "..."}           no option contains the text any more
  {"type": "value_equals", "target": "...", "value": "..."}
  {"type": "env_success"}                        the environment reports the task solved
  {"type": "all", "of": [...]} or {"type": "any", "of": [...]}"""


@dataclass
class ThinkResult:
    navigation: dict[str, str]
    action: str
    arg: Optional[str]
    point: Optional[Point]
    goal: Optional[str]
    replan: Optional[list[Subtask]]
    invented: bool  # the reasoning wanted something the options did not offer
    text: str


class Think:
    def __init__(self, llm: LLM, max_new_tokens: int = 768, temperature: float = 0.0):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _ask(self, prompt: str, image=None) -> tuple[Optional[dict], str]:
        text = generate(self.llm, prompt, image=image, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        return parse_json(text), text

    def plan(self, goal: str, parts: PromptParts) -> list[Subtask]:
        prompt = (
            "Plan for an agent operating a body. Break the goal into small subtasks, each one target and one action. "
            "Put any text that must be typed or written in `arg`.\n"
            f"{CONDITIONS}\n\nGoal: {goal}\n\n{parts.text()}\n\n"
            'Answer with JSON only: {"subtasks": [{"text": "...", "condition": {...}, "arg": null}]}'
        )
        data, _ = self._ask(prompt, parts.images[0] if parts.images else None)
        return _subtasks((data or {}).get("subtasks")) or [Subtask(text=goal, condition={"type": "env_success"})]

    def decide(self, context: GoalContext, parts: PromptParts, labeled: dict[str, Labeled]) -> ThinkResult:
        keys = {name: {o.key: label for label, _, o in entries} for name, entries in labeled.items()}
        by_label = {name: {label: o.key for label, _, o in entries} for name, entries in labeled.items()}
        prompt = (
            "You are the slow, careful mode of an agent. Reason briefly, then choose its next step.\n"
            f"{context.stable_text()}\n\n{parts.text()}\n"
            "Answer each question with an option label from the list above. If the target is a surface, also give "
            '`point` as [x, y] between 0 and 1. Give a short word `goal` for the next steps. If the plan is wrong, '
            "give `replan` (a list of subtasks), else null. If nothing listed fits, say so in `missing`.\n"
            f"{CONDITIONS}\n\nAnswer with JSON only: "
            '{"reasoning": "...", "answers": {' + ", ".join(f'"{n}": "<label>"' for n in labeled)
            + '}, "arg": null, "point": null, "goal": "...", "replan": null, "missing": null}'
        )
        data, text = self._ask(prompt, parts.images[0] if parts.images else None)
        data = data or {}
        answers = data.get("answers") if isinstance(data.get("answers"), dict) else {}
        chosen: dict[str, str] = {}
        for name in labeled:
            value = str(answers.get(name, "")).strip()
            chosen[name] = by_label[name].get(value) or (value if value in keys[name] else _default(name))
        action = chosen.pop("action", NONE)
        point = data.get("point")
        point = (min(max(float(point[0]), 0.0), 1.0), min(max(float(point[1]), 0.0), 1.0)) \
            if isinstance(point, list) and len(point) == 2 else None
        arg = data.get("arg")
        return ThinkResult(navigation=chosen, action=action, arg=str(arg) if arg not in (None, "") else None,
                           point=point, goal=(str(data["goal"]) if data.get("goal") else None),
                           replan=_subtasks(data.get("replan")), invented=bool(data.get("missing")), text=text)


def _default(name: str) -> str:
    if name == "action":
        return NONE
    return KEEP if name == "destination" else HOLD


def _subtasks(raw) -> Optional[list[Subtask]]:
    if not isinstance(raw, list) or not raw:
        return None
    out = []
    for item in raw:
        if isinstance(item, dict) and item.get("text"):
            condition = item.get("condition") if isinstance(item.get("condition"), dict) else None
            arg = item.get("arg")
            out.append(Subtask(str(item["text"]), condition, str(arg) if arg not in (None, "") else None))
    return out or None


__all__ = ["Think", "ThinkResult", "CORE_ACTIONS", "question_text"]
