"""The v3 step loop (spec v3, 3.1): one LLM, act or think, plus the core actions find and wait.

    observe -> auto-find on window change -> plan if needed -> prefill prompt -> two parallel picks
    -> think (picked, or forced by the hard rule) | find | wait | act (narrowing within the step)
    -> protected check -> history and log

Silent steps (stage 6) slot into the goals section once trained; they are off here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from body.contract import Body
from body.factory import make_body
from body.log import StepLog
from body.prompt import Labeled, PromptParts, build
from body.schema import HOLD, NONE, Observation, Outcome, Point, Target
from body.vocab import FIND, THINK, WAIT
from common.config import Config

from .goal_context import GoalContext
from .grounding import point_native
from .llm import LLM, load_llm
from .system1 import Answer, System1
from .think import Think, ThinkResult


@dataclass
class StepReport:
    mode: str  # act | think | find | wait
    forced_think: bool
    forced_plan: bool
    navigation: dict[str, str]
    action: str
    outcome: Outcome
    top_prob: Optional[float]
    narrowing_depth: int = 0
    done: bool = False
    success: Optional[bool] = None
    record: dict = field(default_factory=dict)


class Brain:
    def __init__(self, cfg: Config, integration, llm: Optional[LLM] = None, log_path: Optional[str | Path] = None,
                 seed: int = 0, think_always: bool = False, grounding: str = "narrowing", shadow_think: float = 0.0):
        self.cfg = cfg
        self.integration = integration
        self.llm = llm or load_llm(cfg)
        self.system1 = System1(self.llm, max_options=int(cfg.system1.max_options))
        self.think = Think(self.llm, max_new_tokens=int(cfg.llm.s2_max_new_tokens), temperature=float(cfg.llm.s2_temperature))
        self.goal = GoalContext()
        self.log = StepLog(log_path)
        self.rng = random.Random(seed)
        self.think_always = think_always
        self.grounding = grounding
        self.shadow_think = float(shadow_think)  # share of act steps where think also answers (data, not executed)
        self._prepared = False
        self.body: Optional[Body] = None
        self.history: list[str] = []
        self.steps = 0

    # ------------------------------------------------------------------ episodes
    def reset(self, seed: Optional[int] = None, adapter: Optional[str] = None) -> Observation:
        if self.body is not None:
            self.body.close()
        self.body = make_body(self.cfg, self.integration, adapter or self.cfg.body.adapter)
        obs = self.body.reset(seed)
        self.goal.wipe(goal=self.integration.goal)
        self.history, self.steps = [], 0
        return obs

    # ------------------------------------------------------------------ prompt
    def _goal_lines(self) -> list[str]:
        lines = self.goal.stable_text().splitlines()
        if self.goal.word_goal:
            lines.append(f"Current aim: {self.goal.word_goal}")
        return lines

    def _status(self, refreshed: bool) -> list[str]:
        body = self.body
        return [f"window changed since last find: {'yes' if body.window_changed else 'no'}",
                f"options refreshed this step: {'yes' if refreshed else 'no'}",
                f"currently at: {body.at.describe() if body.at else 'nothing selected'}"]

    def _prepare(self, obs: Observation, refreshed: bool) -> tuple[PromptParts, dict[str, Labeled]]:
        navigation, action = self.body.questions()
        labeled = self.system1.labels(navigation, action)
        parts = build(self.history[-int(self.cfg.loop.history_lines):], obs, self._goal_lines(), self._status(refreshed),
                      labeled)
        return parts, labeled

    # ------------------------------------------------------------------ positions
    def _locate(self, target: Target, parts: PromptParts, purpose: str) -> tuple[Point, int]:
        """A point on a surface: narrowing within this step, or native grounding."""
        if self.grounding == "native" and parts.images:
            point = point_native(self.llm, parts.images[0], f"{purpose} on {target.label}", target.bbox)
            if point is not None:
                return point, 0
        if not self._prepared:
            self.system1.prepare(parts)
            self._prepared = True
        narrowing = self.body.narrowing(target)
        for _ in range(narrowing.max_depth + 1):
            answer = self.system1.ask_extra(f"Where on {target.label}? {purpose}", narrowing.options())
            point = narrowing.pick(answer.key)
            if point is not None:
                return point, narrowing.depth
        return target.center, narrowing.depth

    # ------------------------------------------------------------------ one step
    def _plan_if_needed(self, parts: PromptParts) -> bool:
        if self.goal.has_plan and not self.goal.needs_replan:
            return False
        self.goal.set_plan(self.think.plan(self.goal.goal, parts))
        return True

    def _apply_think(self, result: ThinkResult, tick: int) -> None:
        if result.replan:
            self.goal.set_plan(result.replan)
        if result.goal:
            self.goal.set_word_goal(result.goal, tick)

    def step(self) -> StepReport:
        body, cfg = self.body, self.cfg
        body.begin_step()
        obs = body.observe()
        refreshed = body.auto_find()
        if refreshed:
            self.goal.drop_word_goal()
        elif self.goal.word_goal and body.tick - self.goal.word_goal_tick > int(cfg.loop.goal_max_age):
            self.goal.drop_word_goal()
        parts, labeled = self._prepare(obs, refreshed)
        forced_plan = self._plan_if_needed(parts)
        if forced_plan:
            parts, labeled = self._prepare(obs, refreshed)

        answers: dict[str, Answer] = {}
        top_prob = None
        self._prepared = False
        if not self.think_always:
            self.system1.prepare(parts)
            self._prepared = True
            answers = self.system1.answer_all(labeled)
            top_prob = min(a.top_prob for a in answers.values())
        navigation = {q: a.key for q, a in answers.items() if q != "action"}
        action = answers["action"].key if answers else THINK
        forced_think = bool(answers) and action != THINK and body.core.force_think(top_prob)
        mode = THINK if (action == THINK or forced_think or self.think_always) else action if action in (FIND, WAIT) else "act"

        arg, point = None, None
        thought: Optional[ThinkResult] = None
        if mode == THINK:
            thought = self.think.decide(self.goal, parts, labeled)
            self._apply_think(thought, body.tick)
            navigation, action, arg, point = thought.navigation, thought.action, thought.arg, thought.point
            mode = action if action in (FIND, WAIT) else THINK
        shadow: Optional[ThinkResult] = None
        if thought is None and self.shadow_think and self.rng.random() < self.shadow_think:
            shadow = self.think.decide(self.goal, parts, labeled)

        depth = 0
        if mode == FIND:
            choice = navigation.get("navigate", HOLD)
            scope = choice if (body._lookup(choice) is not None and body._lookup(choice).kind == "group") else None
            outcome = body.find_outcome(scope)
        elif mode == WAIT:
            outcome = body.wait()
        else:
            if action != NONE and action.startswith("type@") and arg is None and self.goal.current is not None:
                arg = self.goal.current.arg
            nav_target = body._lookup(navigation.get("navigate", HOLD))
            if nav_target is not None and nav_target.kind == "surface" and point is None:
                point, depth = self._locate(nav_target, parts, "Where should the pointer go?")
            drag_to = None
            verb, _, target_id = action.partition("@")
            if verb == "drag" and body.at is not None and body.at.id == target_id and body.at.kind == "surface":
                drag_to, extra = self._locate(body.at, parts, "Where should the drag end?")
                depth += extra
            outcome = body.act(navigation, action, arg, point, drag_to)

        self.goal.tick(int(cfg.goal.stall_steps))
        if self.goal.current is not None and body.check(self.goal.current.condition):
            self.goal.advance()
        self.steps += 1
        done = bool(outcome.env_done or self.goal.done or self.steps >= int(cfg.env.max_steps))
        self.history.append(f"step {self.steps}: {mode}, where={navigation.get('navigate', navigation)}, "
                            f"what={action} -> {outcome.text()}")
        record = {
            "step": self.steps, "mode": mode, "forced_think": forced_think, "forced_plan": forced_plan,
            "navigation": navigation, "action": action, "arg": arg, "point": point, "top_prob": top_prob,
            "picks": {q: {"key": a.key, "probs": a.probs} for q, a in answers.items()},
            "think": thought.text if thought else None, "invented": thought.invented if thought else None,
            "shadow_think": ({"navigation": shadow.navigation, "action": shadow.action, "point": shadow.point,
                              "text": shadow.text} if shadow else None),
            "prompt": parts.text(), "outcome": outcome.text(), "latency_ms": outcome.latency_ms,
            "narrowing_depth": depth, "subtask": self.goal.current.text if self.goal.current else None,
            "success": outcome.env_success, **body.last_exec,
        }
        self.log.write(record)
        return StepReport(mode=mode, forced_think=forced_think, forced_plan=forced_plan, navigation=navigation,
                          action=action, outcome=outcome, top_prob=top_prob, narrowing_depth=depth, done=done,
                          success=outcome.env_success, record=record)
