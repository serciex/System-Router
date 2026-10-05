"""The body (CONTRACT.md v0.3): sensory channel from integration, interaction channel from the adapter.

`Body` keeps the body slot (latest find), stable brain-facing ids and canonical labels, builds the options
of the two questions, runs narrowing on surfaces, executes picks (action first, then navigation), and
runs wait. The core actions think and find are dispatched by the brain; find executes here.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Optional

from .adapters.base import Adapter
from .core import CheckState, Core
from .narrowing import Narrowing
from .schema import (
    CONTRACT_VERSION,
    HOLD,
    KEEP,
    NONE,
    ActionOutcome,
    Item,
    Manifest,
    NavOutcome,
    Observation,
    Option,
    Outcome,
    Point,
    Slot,
    Target,
)
from .vocab import CORE_ACTIONS, label


class Body:
    def __init__(self, integration, adapter: Adapter, core: Core, body_id: str = "body", version: str = "1",
                 wait_cap_s: float = 2.0, wait_poll_s: float = 0.1, narrow_min_px: int = 12, narrow_max_depth: int = 6):
        self.integration = integration
        self.adapter = adapter
        self.core = core
        self.body_id, self.version = body_id, version
        self.wait_cap_s, self.wait_poll_s = float(wait_cap_s), float(wait_poll_s)
        self.narrow_min_px, self.narrow_max_depth = int(narrow_min_px), int(narrow_max_depth)
        self.interaction = adapter.manifest()
        if self.interaction.contract_version != CONTRACT_VERSION:
            raise ValueError(f"{adapter.name} targets contract {self.interaction.contract_version}, body is {CONTRACT_VERSION}")
        self._reset_state()

    # ------------------------------------------------------------------ lifecycle
    def _reset_state(self) -> None:
        self._ids: dict[Any, str] = {}
        self._next_id = 1
        self.tick = 0
        self.step = 0
        self.last_obs: Optional[Observation] = None
        self.slot = Slot(targets=[])
        self._find_image = None
        self.at: Optional[Target] = None
        self.destination: Optional[Target] = None
        self.env_success: Optional[bool] = None
        self.env_done = False
        self.last_exec: dict = {}

    def reset(self, seed: Optional[int] = None) -> Observation:
        self._reset_state()
        self.integration.reset(seed)
        self.adapter.reset(seed)
        obs = self.observe()
        self.find()
        return obs

    def close(self) -> None:
        self.adapter.close()

    @property
    def manifest(self) -> Manifest:
        return Manifest(self.body_id, self.version, self.integration.sensory_manifest(), self.interaction)

    # ------------------------------------------------------------------ sensory channel
    def observe(self) -> Observation:
        obs = self.integration.observe()
        obs.tick = self.tick
        self.last_obs = obs
        return obs

    @property
    def window_changed(self) -> bool:
        if self.last_obs is None:
            return True
        return self.core.window_changed(self._find_image, self.last_obs.image)

    # ------------------------------------------------------------------ find and the body slot
    def find(self, scope: Optional[str] = None) -> Slot:
        scope_target = self._lookup(scope) if scope else None
        items = self.adapter.find(scope_target.handle if scope_target else None)
        targets = [self._to_target(item) for item in items]
        if self.interaction.pointer_mode == "relative":
            for target in targets:
                self._set_bearing(target)
        self.slot = Slot(targets=targets, scope=scope, tick=self.tick)
        self._find_image = self.last_obs.image.copy() if self.last_obs is not None else None
        if self.at is not None and self.at.kind != "surface":
            self.at = next((t for t in targets if t.id == self.at.id), None)
        return self.slot

    def auto_find(self) -> bool:
        """Core rule: refresh the slot when the window changed since the last find."""
        if self.window_changed:
            self.find()
            return True
        return False

    def _key(self, handle: Any) -> Any:
        try:
            hash(handle)
            return handle
        except TypeError:
            return repr(handle)

    def _to_target(self, item: Item) -> Target:
        key = self._key(item.handle)
        if key not in self._ids:
            self._ids[key] = f"t{self._next_id}"
            self._next_id += 1
        return Target(id=self._ids[key], label=label(item.role, item.name), kind=item.kind, verbs=tuple(item.verbs),
                      bbox=item.bbox, handle=item.handle, dims=item.dims, collapsed=item.collapsed, value=item.value,
                      reversible=item.reversible, confidence=item.confidence)

    def _lookup(self, target_id: Optional[str]) -> Optional[Target]:
        if target_id == "self":
            return self.self_target()
        return next((t for t in self.slot.targets if t.id == target_id), None)

    def _set_bearing(self, target: Target) -> None:
        anchor = (self.last_obs.pointer if self.last_obs and self.last_obs.pointer else (0.5, 0.5))
        target.bearing = round((target.center[0] - anchor[0]) * self.interaction.fov_deg, 1)
        height = target.bbox[3] - target.bbox[1]
        target.distance = "near" if height > 0.3 else ("mid" if height > 0.1 else "far")

    def self_target(self) -> Target:
        x, y = (self.last_obs.pointer if self.last_obs and self.last_obs.pointer else (0.5, 0.5))
        return Target(id="self", label="self: body", kind="self", verbs=self.interaction.self_verbs, bbox=(x, y, x, y))

    # ------------------------------------------------------------------ the two questions
    def questions(self) -> tuple[dict[str, list[Option]], list[Option]]:
        """Navigation options per question, and action options (core actions only here)."""
        targets = [Option(t.id, t.describe()) for t in self.slot.targets]
        if self.interaction.pointer_mode == "relative":
            navigation = {axis: [Option(HOLD, "hold")] + [Option(i, i) for i in inputs if i != HOLD]
                          for axis, inputs in self.interaction.movement_axes.items()}
            navigation["destination"] = [Option(KEEP, "keep the current destination")] + targets
        else:
            navigation = {"navigate": [Option(HOLD, "stay where you are")] + targets}
        action = [Option(NONE, "do nothing")]
        if self.at is not None:
            action += [Option(f"{verb}@{self.at.id}", f"{verb} {self.at.label}") for verb in self.at.verbs]
        action += [Option(f"{verb}@self", verb) for verb in self.interaction.self_verbs]
        action += [Option(name, _CORE_TEXT[name]) for name in CORE_ACTIONS]
        return navigation, action

    def narrowing(self, target: Target) -> Narrowing:
        size = None
        if self.last_obs is not None:
            h, w = self.last_obs.image.shape[:2]
            size = (w, h)
        return Narrowing(target.bbox, target.dims or 2, size, self.narrow_min_px, self.narrow_max_depth)

    # ------------------------------------------------------------------ acting
    def begin_step(self) -> None:
        self.core.start()
        self.tick += 1

    def act(self, navigation: dict[str, str], action: str = NONE, arg: Optional[str] = None,
            point: Optional[Point] = None, drag_to: Optional[Point] = None) -> Outcome:
        """Action on the current target first, then navigation. `point` is the narrowing result for a surface pick."""
        results: list[dict] = []
        native: dict = {}
        action_outcome = self._do_action(action, arg, drag_to, results, native)
        nav_outcome = self._do_navigation(navigation, point, results, native)
        before = self.last_obs.image if self.last_obs is not None else None
        obs = self.observe()
        changed = self.core.window_changed(before, obs.image) or any(r.get("reward") for r in results)
        if action_outcome.status == "done":
            action_outcome.changed = changed
        if self.interaction.pointer_mode == "relative" and nav_outcome.status == "moving":
            nav_outcome = self._relative_status(nav_outcome)
        return self._finish(nav_outcome, action_outcome, results, native)

    def _finish(self, nav: NavOutcome, act: ActionOutcome, results: list[dict], native: dict) -> Outcome:
        for r in results:
            self.env_done = self.env_done or bool(r.get("done"))
            if r.get("success") is not None:
                self.env_success = bool(r["success"])
        self.step += 1
        self.last_exec = {"native": native, "pointer": self.last_obs.pointer if self.last_obs else None}
        return Outcome(navigation=nav, action=act, step=self.step, latency_ms=self.core.stop(),
                       env_reward=float(sum(r.get("reward", 0.0) for r in results)), env_done=self.env_done,
                       env_success=self.env_success)

    def _revalidate(self, target_id: str) -> Optional[Target]:
        target = self._lookup(target_id)
        if target is None or target.kind == "self":
            return target
        if self.window_changed:
            self.find(self.slot.scope)
            fresh = self._lookup(target_id)
            if fresh is None:
                return None
            if target.point is not None:
                fresh = replace(fresh, point=target.point)
            target = fresh
        return target

    def _do_action(self, action: str, arg: Optional[str], drag_to: Optional[Point], results: list[dict],
                   native: dict) -> ActionOutcome:
        if not action or action == NONE:
            return ActionOutcome("none", False, "no action")
        if action in CORE_ACTIONS:
            return ActionOutcome("none", False, f"{action} is handled by the brain")
        verb, _, target_id = action.partition("@")
        target = self._revalidate(target_id)
        if target is not None and self.at is not None and self.at.id == target_id and self.at.point is not None:
            target = replace(target, point=self.at.point)
        if target is None:
            return ActionOutcome("unavailable", False, f"{target_id} is no longer available")
        if target.kind != "self" and verb not in target.verbs:
            return ActionOutcome("failed", False, f"{target.label} does not support {verb}")
        allowed, reason = self.core.allow(verb, target)
        if not allowed:
            return ActionOutcome("failed", False, reason)
        if target.kind == "surface":
            start = target.point or target.center
            points = [start, drag_to] if verb == "drag" and drag_to is not None else [start]
            res = self.adapter.act_at(target.handle, points, verb)
        else:
            res = self.adapter.invoke(None if target.kind == "self" else target.handle, verb, arg)
        results.append(res)
        native["action"] = {"verb": verb, "target": target.id, "arg": arg}
        if not res.get("ok"):
            return ActionOutcome("failed", False, res.get("text") or f"{verb} failed")
        return ActionOutcome("done", True, res.get("text") or f"{verb} {target.label}")

    def _do_navigation(self, navigation: dict[str, str], point: Optional[Point], results: list[dict],
                       native: dict) -> NavOutcome:
        navigation = navigation or {}
        if self.interaction.pointer_mode == "relative":
            return self._move_relative(navigation, results, native)
        choice = navigation.get("navigate", HOLD)
        if choice == HOLD:
            return NavOutcome("held", "stayed in place")
        target = self._revalidate(choice)
        if target is None:
            return NavOutcome("blocked", f"{choice} is no longer available")
        if target.kind == "group":
            self.find(scope=target.id)
            native["navigation"] = {"expand": target.id}
            return NavOutcome("reached", f"opened {target.label}")
        if target.kind == "surface":
            target = replace(target, point=point or target.center)
            if "hover" in self.interaction.verbs:
                results.append(self.adapter.act_at(target.handle, [target.point], "hover"))
            self.at = target
            native["navigation"] = {"surface": target.id, "point": list(target.point)}
            return NavOutcome("reached", f"at {target.label}, point {target.point[0]:.2f}, {target.point[1]:.2f}")
        if "hover" in target.verbs:
            results.append(self.adapter.invoke(target.handle, "hover"))
        self.at = target
        native["navigation"] = {"target": target.id}
        return NavOutcome("reached", f"at {target.label}")

    def _move_relative(self, navigation: dict[str, str], results: list[dict], native: dict) -> NavOutcome:
        choice = navigation.get("destination", KEEP)
        if choice != KEEP:
            self.destination = self._lookup(choice)
        inputs = {axis: value for axis, value in navigation.items() if axis != "destination" and value != HOLD}
        if not inputs:
            return NavOutcome("held", "held position")
        res = self.adapter.move(inputs)
        results.append(res)
        native["navigation"] = {"move": inputs}
        if res.get("blocked") or not res.get("ok"):
            return NavOutcome("blocked", res.get("text") or "movement blocked")
        return NavOutcome("moving", res.get("text") or ", ".join(f"{k} {v}" for k, v in inputs.items()))

    def _relative_status(self, outcome: NavOutcome) -> NavOutcome:
        if self.destination is None:
            return outcome
        self.find(self.slot.scope)
        target = self._lookup(self.destination.id)
        if target is None:
            return outcome
        if target.distance == "near" and abs(target.bearing or 0) < 10:
            self.at = target
            return NavOutcome("reached", f"reached {target.label}")
        return NavOutcome("moving", f"{target.describe()}")

    # ------------------------------------------------------------------ wait
    def wait(self, watch: Optional[list] = None, cap_s: Optional[float] = None) -> Outcome:
        """No inference: poll the sensory channel until a meaningful change or the cap."""
        cap = self.wait_cap_s if cap_s is None else float(cap_s)
        start_image = self.last_obs.image if self.last_obs is not None else None
        started = time.monotonic()
        seen = False
        while time.monotonic() - started < cap:
            time.sleep(self.wait_poll_s)
            obs = self.observe()
            if self.core.window_changed(start_image, obs.image, watch):
                seen = True
                break
        status = "done" if seen else "none"
        text = "change seen" if seen else f"no change within {cap:.1f} s"
        return self._finish(NavOutcome("held", "waited"), ActionOutcome(status, seen, text), [], {"wait": cap})

    def find_outcome(self, scope: Optional[str] = None) -> Outcome:
        """A model-called find as a step outcome."""
        slot = self.find(scope)
        text = f"found {len(slot.targets)} items" + (f" in {scope}" if scope else "")
        return self._finish(NavOutcome("held", "stayed in place"), ActionOutcome("done", False, text), [], {"find": scope})

    # ------------------------------------------------------------------ checks
    def check(self, condition: dict | None) -> bool:
        if self.window_changed:
            self.find(self.slot.scope)
        return self.core.check(condition, CheckState(targets=self.slot.targets, env_success=self.env_success))


_CORE_TEXT = {
    "think": "think: reason in words before acting",
    "find": "find: look again at what is available here",
    "wait": "wait: do nothing until something changes",
}
