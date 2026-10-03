"""System 2: deliberate reasoning with the image. Plans the task list, or picks when System 1 is unsure.

Outputs are JSON so they can be validated: every chosen key must be one of the offered options, and
every subtask carries a checkable completion condition the protected core can evaluate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from body.schema import HOLD, NONE, Observation

from .goal_context import GoalContext, Subtask
from .llm import LLM, generate, parse_json

CONDITIONS = """Completion conditions (pick one per subtask, the body checks them):
  {"type": "text_visible", "text": "..."}        some target's label or value contains the text
  {"type": "text_gone", "text": "..."}           no target contains the text any more
  {"type": "value_equals", "target": "...", "value": "..."}  a target whose label contains `target` holds `value`
  {"type": "env_success"}                        the environment reports the task solved
  {"type": "all", "of": [...]} or {"type": "any", "of": [...]}"""


@dataclass
class S2Decision:
    navigation: dict[str, str]
    action: str
    arg: Optional[str]
    replan: Optional[list[Subtask]]


class System2:
    def __init__(self, llm: LLM, max_new_tokens: int = 768, temperature: float = 0.0):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _ask(self, prompt: str, image=None) -> Optional[dict]:
        return parse_json(generate(self.llm, prompt, image=image, max_new_tokens=self.max_new_tokens,
                                   temperature=self.temperature))

    def plan(self, goal: str, state_text: str, history: str, image=None) -> list[Subtask]:
        prompt = (
            "You plan for a computer-use agent. Break the goal into small subtasks, each one target and one "
            "action (for example 'click the Submit button' or 'type the username into the Username field'). "
            "Put any text that must be typed in `arg`.\n"
            f"{CONDITIONS}\n\n"
            f"Goal: {goal}\n\nCurrent screen:\n{state_text}\n\nHistory:\n{history}\n\n"
            'Answer with JSON only: {"subtasks": [{"text": "...", "condition": {...}, "arg": null}]}'
        )
        data = self._ask(prompt, image) or {}
        return _subtasks(data.get("subtasks")) or [Subtask(text=goal, condition={"type": "env_success"})]

    def decide(self, context: GoalContext, observation: Observation, state_text: str, history: str,
               image=None) -> S2Decision:
        nav_listing = "\n".join(
            f"{question}:\n" + "\n".join(f"  {o.key}: {o.text}" for o in options)
            for question, options in observation.navigation.items()
        )
        action_listing = "\n".join(f"  {o.key}: {o.text}" for o in observation.action)
        prompt = (
            "You control a computer-use agent and the fast system was unsure. Choose its next step.\n"
            f"{context.stable_text()}\n\nState:\n{state_text}\n\nHistory:\n{history}\n\n"
            f"Navigation questions and options:\n{nav_listing}\n\nAction options:\n{action_listing}\n\n"
            "If the plan is wrong, give a new list of subtasks in `replan` (same format as planning), else null.\n"
            f"{CONDITIONS}\n\n"
            'Answer with JSON only: {"navigation": {"<question>": "<option key>"}, "action": "<option key>", '
            '"arg": null, "replan": null}'
        )
        data = self._ask(prompt, image) or {}
        valid_nav = {q: {o.key for o in opts} for q, opts in observation.navigation.items()}
        chosen = data.get("navigation") if isinstance(data.get("navigation"), dict) else {}
        navigation = {q: (chosen.get(q) if chosen.get(q) in keys else _default_nav(q)) for q, keys in valid_nav.items()}
        action_keys = {o.key for o in observation.action}
        action = data.get("action") if data.get("action") in action_keys else NONE
        arg = data.get("arg")
        return S2Decision(navigation=navigation, action=action, arg=str(arg) if arg not in (None, "") else None,
                          replan=_subtasks(data.get("replan")))


def _default_nav(question: str) -> str:
    return "keep" if question == "destination" else HOLD


def _subtasks(raw) -> Optional[list[Subtask]]:
    if not isinstance(raw, list) or not raw:
        return None
    subtasks = []
    for item in raw:
        if isinstance(item, dict) and item.get("text"):
            condition = item.get("condition") if isinstance(item.get("condition"), dict) else None
            arg = item.get("arg")
            subtasks.append(Subtask(text=str(item["text"]), condition=condition,
                                    arg=str(arg) if arg not in (None, "") else None))
    return subtasks or None


def dump_subtasks(subtasks: list[Subtask]) -> str:
    return json.dumps([s.__dict__ for s in subtasks])
