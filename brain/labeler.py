"""Labeler: the frozen VLM with privileged access to the environment's source. Training only.

For each screen and subtask it returns the targets that really exist, the important ones (on any valid
path from this screen to the goal) and the level each important target needs (1 if moving in its
direction is enough, 3 if the specific element is needed), plus the verbs that make progress on it.
One label is cached per screen and subtask, and the same label serves both rewards.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from body.schema import HOLD, NONE, Target

from .llm import LLM, generate, parse_json

MAX_SOURCE_CHARS = 12000


@dataclass
class Label:
    valid: set[str] = field(default_factory=set)
    important: dict[str, dict] = field(default_factory=dict)  # id -> {"level": int, "verbs": [str]}

    def to_json(self) -> dict:
        return {"valid": sorted(self.valid), "important": self.important}

    @classmethod
    def from_json(cls, data: dict) -> "Label":
        return cls(valid=set(data.get("valid", [])), important=dict(data.get("important", {})))


class Labeler:
    def __init__(self, llm: LLM, cache_dir: Optional[str | Path] = None, max_new_tokens: int = 512):
        self.llm = llm
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self._memory: dict[str, Label] = {}

    def label(self, source: str, goal: str, subtask: str, targets: list[Target]) -> Label:
        signature = "\n".join(f"{t.id}|{t.label}" for t in targets)
        key = hashlib.sha1(f"{source}\n{goal}\n{subtask}\n{signature}".encode()).hexdigest()
        if key in self._memory:
            return self._memory[key]
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            label = Label.from_json(json.loads(path.read_text(encoding="utf-8")))
        else:
            label = self._query(source, goal, subtask, targets)
            if path:
                path.write_text(json.dumps(label.to_json()), encoding="utf-8")
        self._memory[key] = label
        return label

    def _query(self, source: str, goal: str, subtask: str, targets: list[Target]) -> Label:
        listing = "\n".join(f"{t.id}: {t.label} (verbs: {', '.join(t.verbs) or 'none'})" for t in targets)
        prompt = (
            "You label a screen for training a computer-use agent. You can read the page source, which the "
            "agent cannot.\n"
            f"Goal: {goal}\nCurrent subtask: {subtask}\n\n"
            f"Page source (truncated):\n{source[:MAX_SOURCE_CHARS]}\n\n"
            f"Targets the agent can see:\n{listing}\n\n"
            "1. `valid`: ids of targets that really exist and can be acted on.\n"
            "2. `important`: targets on ANY valid path from this screen to completing the goal, with `level` 1 "
            "if moving in its direction is enough or 3 if the specific element is needed, and the `verbs` that "
            "make progress on it.\n"
            'Answer with JSON only: {"valid": ["t1"], "important": [{"id": "t1", "level": 3, "verbs": ["click"]}]}'
        )
        data = parse_json(generate(self.llm, prompt, max_new_tokens=self.max_new_tokens)) or {}
        known = {t.id for t in targets}
        valid = {i for i in data.get("valid", []) if i in known}
        important = {}
        for item in data.get("important", []) or []:
            if isinstance(item, dict) and item.get("id") in known:
                level = 1 if int(item.get("level", 3)) == 1 else 3
                important[item["id"]] = {"level": level, "verbs": [str(v) for v in item.get("verbs", [])]}
        return Label(valid=valid | set(important), important=important)

    @staticmethod
    def verdict(label: Label, nav_choice: Optional[str], action_key: Optional[str], at_id: Optional[str]) -> bool:
        """Per-step correctness: the navigation and action both make progress on an important target."""
        at_important = at_id is not None and at_id in label.important
        acting = bool(action_key) and action_key != NONE
        if acting:
            verb, _, target_id = action_key.partition("@")
            act_ok = target_id in label.important and (not label.important[target_id]["verbs"]
                                                       or verb in label.important[target_id]["verbs"])
        else:
            act_ok = True
        if nav_choice in (None, HOLD, "keep"):
            nav_ok = acting and act_ok and at_important
        else:
            nav_ok = nav_choice in label.important
        return bool(nav_ok and act_ok)
