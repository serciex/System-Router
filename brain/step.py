"""One step of the whole system (spec v2, "One step").

`Brain` owns the frozen-LLM roles and runs steps 4-7 for a decision (route, level, cells) that comes
from either the stage-1 rules or the world model:

    observe -> (plan if needed) -> System 1, escalate if unsure -> System 2 -> act -> check -> rewards -> log

and prepares the world model's inputs (vision cells, text feature, vector) for the next decision.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from body.contract import KEEP, Body
from body.factory import make_body, pick_adapters
from body.log import StepLog
from body.schema import HOLD, NONE, Observation, Outcome
from common.config import Config, resolve
from wm.codec import S2, Decision
from wm.rewards import dec_reward, seg_reward

from .features import FeatureExtractor
from .goal_context import GoalContext
from .labeler import Label, Labeler
from .llm import LLM, load_llm
from .system1 import System1
from .system2 import System2

VEC_SIZE = 10


@dataclass
class StepReport:
    decision: Decision
    route_used: str  # s1 | s2
    escalated: bool
    forced_plan: bool
    navigation: dict[str, str]
    action: str
    outcome: Outcome
    s1_confidence: Optional[float]
    correct: Optional[bool]
    r_seg: float
    r_dec: float
    seg_info: dict = field(default_factory=dict)
    done: bool = False
    success: Optional[bool] = None


class Brain:
    def __init__(self, cfg: Config, environment, llm: Optional[LLM] = None, use_labeler: Optional[bool] = None,
                 log_path: Optional[str | Path] = None, seed: int = 0):
        self.cfg = cfg
        self.environment = environment
        self.llm = llm or load_llm(cfg)
        self.system1 = System1(self.llm, labels=cfg.system1.labels, max_options=int(cfg.system1.max_options),
                               text_layer=int(cfg.llm.text_feature_layer))
        self.system2 = System2(self.llm, max_new_tokens=int(cfg.llm.s2_max_new_tokens),
                               temperature=float(cfg.llm.s2_temperature))
        self.features = FeatureExtractor(self.llm, vision_grid=cfg.features.vision_grid, cell_dim=cfg.features.cell_dim,
                                         text_dim=cfg.features.text_dim, seed=cfg.features.seed)
        enabled = cfg.labeler.enabled if use_labeler is None else use_labeler
        self.labeler = Labeler(self.llm, cache_dir=resolve(cfg.paths.label_cache)) if enabled else None
        self.goal = GoalContext()
        self.log = StepLog(log_path)
        self.rng = random.Random(seed)
        self.body: Optional[Body] = None
        self.adapter_names: list[str] = []
        self.steps = 0
        self.last_outcome_text = "Episode started."
        self.last_confidence = 1.0
        self.last_route = 0
        self.last_nav_status = "held"
        self.last_action_failed = False

    # ------------------------------------------------------------------ episodes
    def reset(self, seed: Optional[int] = None, adapters: Optional[list[str]] = None, evaluate: bool = False,
              features: bool = True) -> Optional[dict]:
        """Start an episode. Returns the world model's first inputs, or None with features=False (stage 1)."""
        if self.body is not None:
            self.body.close()
        self.adapter_names = adapters or pick_adapters(self.cfg, self.rng, evaluate=evaluate)
        self.body = make_body(self.cfg, self.environment, self.adapter_names, seed=seed or 0)
        frame = self.body.reset(seed)
        self.goal.wipe(goal=frame.text)
        self.system1.reset()
        self.log.clear_recent()
        self.steps = 0
        self.last_outcome_text = "Episode started."
        self.last_confidence, self.last_route = 1.0, 0
        self.last_nav_status, self.last_action_failed = "held", False
        if not features:
            self.advance_context()
            return None
        return self.wm_inputs()

    def advance_context(self) -> np.ndarray:
        """Extend the goal context with the last outcome (System 1 needs this every step). Returns the hidden state."""
        self.system1.set_stable(self.goal.stable_text())
        return self.system1.begin_step(self.last_outcome_text)

    def wm_inputs(self) -> dict:
        """The world model's observation for the next decision: vision cells, text feature, vector."""
        hidden = self.advance_context()
        summary = self.body.summary()
        vec = np.array([
            self.last_confidence,
            float(self.last_route),
            min(summary["n_targets"] / 50.0, 1.0),
            summary["mean_confidence"],
            float(summary["has_at"]),
            float(self.goal.has_plan),
            min(self.goal.steps_on_subtask / max(int(self.cfg.goal.stall_steps), 1), 1.0),
            float(self.last_nav_status == "reached"),
            float(self.last_action_failed),
            min(self.steps / max(int(self.cfg.env.max_steps), 1), 1.0),
        ], dtype=np.float32)
        return {
            "vis": self.features.vision(self.body.frame.image),
            "txt": self.features.text(hidden),
            "vec": vec,
        }

    # ------------------------------------------------------------------ step parts
    def _begin(self, decision: Decision) -> tuple[Observation, str, np.ndarray, bool]:
        """Start timing, narrow the view, and plan first if there is no task list (or it stalled)."""
        self.body.begin_step()
        observation = self.body.observe(decision.level, decision.cells)
        state_text = self.state_text(observation)
        forced_plan = False
        if not self.goal.has_plan or self.goal.needs_replan:
            self.goal.set_plan(self.system2.plan(self.goal.goal, state_text, self.log.history_text(), observation.image))
            self.system1.set_stable(self.goal.stable_text())
            self.system1.begin_step(self.last_outcome_text)
            forced_plan = True
        return observation, state_text, observation.image, forced_plan

    def _ask_system1(self, observation: Observation, state_text: str) -> tuple[dict, str, float]:
        answers = self.system1.ask(state_text, self.questions(observation))
        navigation = {q: a.key for q, a in answers.items() if q != "action"}
        return navigation, answers["action"].key, min(a.confidence for a in answers.values())

    def _ask_system2(self, observation: Observation, state_text: str, image) -> tuple[dict, str, Optional[str]]:
        s2 = self.system2.decide(self.goal, observation, state_text, self.log.history_text(), image)
        if s2.replan:
            self.goal.set_plan(s2.replan)
        return s2.navigation, s2.action, s2.arg

    def _execute(self, navigation: dict, action: str, arg: Optional[str]):
        """Act, then let the protected core check the subtask. Returns (outcome, at_before, acted_target, done)."""
        body = self.body
        if action != NONE and action.startswith("type@") and arg is None and self.goal.current is not None:
            arg = self.goal.current.arg
        at_before = body.at.id if body.at is not None else None
        acted_target = body._find(action.partition("@")[2]) if action != NONE else None
        outcome = body.act(navigation, action, arg)
        self.goal.tick(int(self.cfg.goal.stall_steps))
        if self.goal.current is not None and body.check(self.goal.current.condition):
            self.goal.advance()
        self.steps += 1
        done = bool(outcome.env_done or self.goal.done or self.steps >= int(self.cfg.env.max_steps))
        return outcome, at_before, acted_target, done

    def _remember(self, outcome: Outcome, confidence: Optional[float], route_used: str) -> None:
        self.last_outcome_text = outcome.text()
        self.last_confidence = confidence if confidence is not None else 0.0
        self.last_route = 1 if route_used == "s2" else 0
        self.last_nav_status = outcome.navigation.status
        self.last_action_failed = outcome.action.status == "failed"

    # ------------------------------------------------------------------ one step (deployment and stage 1)
    def step(self, decision: Decision) -> StepReport:
        body, cfg = self.body, self.cfg
        observation, state_text, image, forced_plan = self._begin(decision)

        # Label the screen the decision was made on (optional, cached per screen and subtask).
        label: Optional[Label] = None
        if self.labeler is not None and self.goal.current is not None:
            label = self.labeler.label(body.describe_source(), self.goal.goal, self.goal.current.text, body.targets)

        route_used = "s2" if (decision.route == S2 or forced_plan) else "s1"
        escalated, confidence = False, None
        navigation, action, arg = {}, NONE, None
        if route_used == "s1":
            navigation, action, confidence = self._ask_system1(observation, state_text)
            if confidence < float(cfg.system1.escalate_below):
                escalated, route_used = True, "s2"  # hard rule: the router's mistakes are caught here
        if route_used == "s2":
            navigation, action, arg = self._ask_system2(observation, state_text, image)

        outcome, at_before, acted_target, done = self._execute(navigation, action, arg)

        # Rewards.
        r_seg, seg_info, correct = 0.0, {}, None
        if label is not None:
            r_seg, seg_info = seg_reward(label.important, observation, body.targets, body.grid,
                                         body.frame.anchor if body.frame else (0.5, 0.5), **dict(cfg.rewards.seg))
            nav_choice = navigation.get("navigate", navigation.get("destination"))
            correct = Labeler.verdict(label, nav_choice, action, at_before)
        elif outcome.env_done:
            correct = outcome.env_success
        irreversible = acted_target is not None and acted_target.reversible is False
        r_dec = 0.0 if forced_plan else dec_reward(correct, outcome.latency_ms, route_used == "s2", irreversible,
                                                   float(cfg.rewards.expected_latency_ms), **dict(cfg.rewards.dec))

        self.log.write({
            "step": self.steps, "adapters": self.adapter_names, "route": route_used, "router_choice": decision.route,
            "escalated": escalated, "forced_plan": forced_plan, "level": decision.level, "n_cells": len(decision.cells),
            "n_targets_returned": len(observation.targets), "navigation": navigation, "action": action, "arg": arg,
            "s1_confidence": confidence, "outcome": outcome.text(), "latency_ms": outcome.latency_ms,
            "subtask": self.goal.current.text if self.goal.current else None, "correct": correct,
            "r_seg": r_seg, "r_dec": r_dec, **body.last_exec,
        })

        self._remember(outcome, confidence, route_used)
        return StepReport(decision=decision, route_used=route_used, escalated=escalated, forced_plan=forced_plan,
                          navigation=navigation, action=action, outcome=outcome, s1_confidence=confidence,
                          correct=correct, r_seg=r_seg, r_dec=r_dec, seg_info=seg_info, done=done,
                          success=outcome.env_success)

    # ------------------------------------------------------------------ one step (data collection)
    def collect_step(self, decision: Decision, execute: str = "s2") -> tuple[StepReport, dict]:
        """Both systems answer every step; one of them is executed. Returns the report and a record for hindsight labels.

        execute: "s2" (System 2's decision runs, most successes), "s1" (System 1 runs with the usual
        escalation rule), or "mix" (System 1 or System 2 at random).
        """
        body = self.body
        observation, state_text, image, forced_plan = self._begin(decision)
        anchor = body.frame.anchor
        positions = {t.id: list(t.center) for t in body.targets}  # where every target was when deciding

        s1_nav, s1_action, confidence = self._ask_system1(observation, state_text)
        s2_nav, s2_action, s2_arg = self._ask_system2(observation, state_text, image)
        if execute == "mix":
            execute = "s1" if self.rng.random() < 0.5 else "s2"
        if execute == "s1" and not forced_plan and confidence >= float(self.cfg.system1.escalate_below):
            route_used, navigation, action, arg = "s1", s1_nav, s1_action, None
        else:
            route_used, navigation, action, arg = "s2", s2_nav, s2_action, s2_arg

        outcome, at_before, acted_target, done = self._execute(navigation, action, arg)
        nav_choice = navigation.get("navigate", navigation.get("destination"))
        record = {
            "step": self.steps,
            "adapters": self.adapter_names,
            "anchor": list(anchor),
            "positions": positions,
            "s1": {"navigation": s1_nav, "action": s1_action, "confidence": confidence},
            "s2": {"navigation": s2_nav, "action": s2_action},
            "executed": route_used,
            "forced_plan": forced_plan,
            "nav_target": nav_choice if nav_choice in positions else None,
            "acted_target": acted_target.id if acted_target is not None and acted_target.kind == "element" else None,
            "action_status": outcome.action.status,
            "nav_status": outcome.navigation.status,
            "latency_ms": outcome.latency_ms,
            "done": done,
            "success": outcome.env_success,
        }
        self.log.write({**record, "positions": len(positions), "outcome": outcome.text(), **body.last_exec})
        self._remember(outcome, confidence, route_used)
        report = StepReport(decision=decision, route_used=route_used, escalated=False, forced_plan=forced_plan,
                            navigation=navigation, action=action, outcome=outcome, s1_confidence=confidence,
                            correct=None, r_seg=0.0, r_dec=0.0, done=done, success=outcome.env_success)
        return report, record

    # ------------------------------------------------------------------ prompts
    def state_text(self, observation: Observation) -> str:
        lines = [f"Currently at: {observation.at.describe() if observation.at else 'no specific target'}"]
        if observation.destination is not None:
            lines.append(f"Destination: {observation.destination.describe()}")
        lines.append(f"Targets in view: {len(observation.targets)}")
        return "\n".join(lines)

    def questions(self, observation: Observation) -> dict:
        questions = {}
        for name, options in observation.navigation.items():
            if name == "navigate":
                text = "Which target should the pointer move to next? Choose 'stay where you are' if you are already at the right target."
            elif name == "destination":
                text = "Which target should you head toward? Keep the current destination unless the subtask needs another."
            else:
                text = f"Which {name} input moves you toward the destination?"
            questions[name] = (text, options)
        questions["action"] = ("Which action should you take on the current target now? Choose 'do nothing' if you still need to move.",
                               observation.action)
        return questions


__all__ = ["Brain", "StepReport", "HOLD", "KEEP"]
