"""The contract, built once and shared by every environment (CONTRACT.md v0.2).

`Body` merges adapter output into targets with stable IDs, filters them by the world model's level and
cells, builds the options of the two decision models (navigation and action), and executes their choices
in the same step. Adapters only translate; everything here is environment-independent.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .adapters.base import Adapter
from .core import CheckState, Core
from .grid import Grid, iou
from .schema import (
    HOLD,
    NONE,
    ActionOutcome,
    Capabilities,
    Frame,
    NativeElement,
    NavOutcome,
    Observation,
    Option,
    Outcome,
    Target,
)

KEEP = "keep"  # relative mode: keep the current destination


class Body:
    def __init__(self, adapters: list[Adapter], grid: Grid, core: Core, self_verbs: tuple[str, ...] = ()):
        if not adapters:
            raise ValueError("A body needs at least one adapter")
        self.adapters = list(adapters)
        self.primary = self.adapters[0]
        self.caps: Capabilities = self.primary.capabilities()
        for adapter in self.adapters:
            caps = adapter.capabilities()
            if caps.contract_version != self.caps.contract_version:
                raise ValueError(f"{adapter.name} targets contract {caps.contract_version}, body is {self.caps.contract_version}")
        self.grid = grid
        self.core = core
        self.self_verbs = tuple(self_verbs)
        self._by_name = {adapter.name: adapter for adapter in self.adapters}
        self._reset_state()

    # ------------------------------------------------------------------ lifecycle
    def _reset_state(self) -> None:
        self._ids: dict[tuple[str, Any], str] = {}
        self._next_id = 1
        self.step = 0
        self.frame: Optional[Frame] = None
        self.targets: list[Target] = []
        self.at: Optional[Target] = None
        self.destination: Optional[Target] = None
        self.observation: Optional[Observation] = None
        self.env_success: Optional[bool] = None
        self.env_done = False
        self.last_exec: dict = {}

    def reset(self, seed: Optional[int] = None) -> Frame:
        self._reset_state()
        self.primary.reset(seed)
        return self.sense()

    def close(self) -> None:
        for adapter in self.adapters:
            adapter.close()

    # ------------------------------------------------------------------ sensing
    def sense(self) -> Frame:
        """Read every adapter, merge, and assign stable IDs. Called at the start of each step and after acting."""
        frames = [(adapter, adapter.read()) for adapter in self.adapters]
        self.frame = frames[0][1]
        previous = {t.id: t for t in self.targets}
        self.targets = self._merge_and_track(frames, previous)
        by_id = {t.id: t for t in self.targets}
        self.at = by_id.get(self.at.id) if self.at is not None else None
        if self.destination is not None and self.destination.kind == "element":
            self.destination = by_id.get(self.destination.id)
        if self.caps.pointer == "relative":
            for target in self.targets:
                self._set_bearing(target)
        return self.frame

    def summary(self) -> dict:
        """Cheap numbers for the world model's vector input."""
        confidences = [t.confidence for t in self.targets] or [0.0]
        return {
            "n_targets": len(self.targets),
            "mean_confidence": float(np.mean(confidences)),
            "anchor": self.frame.anchor if self.frame else (0.5, 0.5),
            "has_at": self.at is not None,
        }

    def _merge_and_track(self, frames: list[tuple[Adapter, Frame]], previous: dict[str, Target]) -> list[Target]:
        kept: list[Target] = []
        for adapter, frame in frames:
            stable = adapter.capabilities().stable_ids
            for element in frame.elements:
                candidate = self._to_target(adapter.name, element)
                if any(iou(candidate.bbox, other.bbox) > 0.6 for other in kept):
                    continue  # structured sources come first and win
                candidate.id = self._assign_id(adapter.name, element, candidate, stable, previous)
                kept.append(candidate)
        return kept

    def _to_target(self, source: str, element: NativeElement) -> Target:
        return Target(
            id="",
            label=element.label,
            verbs=tuple(element.native_actions),
            bbox=element.bbox,
            reversible=element.reversible,
            confidence=element.confidence,
            value=element.value,
            source=source,
            handle=element.handle,
        )

    def _assign_id(self, source: str, element: NativeElement, candidate: Target, stable: bool,
                   previous: dict[str, Target]) -> str:
        key = (source, element.handle)
        if stable and key in self._ids:
            return self._ids[key]
        if not stable:
            best, best_iou = None, 0.5
            for old in previous.values():
                if old.source == source and old.label == candidate.label:
                    overlap = iou(old.bbox, candidate.bbox)
                    if overlap > best_iou:
                        best, best_iou = old, overlap
            if best is not None:
                return best.id
        new_id = f"t{self._next_id}"
        self._next_id += 1
        if stable:
            self._ids[key] = new_id
        return new_id

    def _set_bearing(self, target: Target) -> None:
        anchor = self.frame.anchor
        cx, _ = target.center
        target.bearing = round((cx - anchor[0]) * self.caps.fov_deg, 1)
        height = target.bbox[3] - target.bbox[1]
        target.distance = "near" if height > 0.3 else ("mid" if height > 0.1 else "far")

    # ------------------------------------------------------------------ observing
    def self_target(self) -> Target:
        x, y = self.frame.anchor if self.frame else (0.5, 0.5)
        return Target(id="self", label="yourself", verbs=self.self_verbs, bbox=(x, y, x, y),
                      source=self.primary.name, kind="self")

    def observe(self, level: int, cells: tuple[int, ...] | list[int]) -> Observation:
        """Targets inside the selected cells at that level, plus the options of both decision models."""
        if self.frame is None:
            raise RuntimeError("sense() or reset() must be called before observe()")
        if level not in self.grid.levels:
            raise ValueError(f"Level {level} is not active")
        cells = tuple(sorted(set(int(c) for c in cells)))
        anchor = self.frame.anchor
        selected = set(cells)
        returned: list[Target] = []

        if level == 1:
            for index in cells:
                name = self.grid.direction(index)
                if name == "here":
                    returned += [t for t in self.targets if self.grid.cell_of(1, *t.center, anchor) == index]
                    continue
                returned.append(Target(id=f"dir:{name}", label=f"{name} of you", verbs=(), source="contract",
                                       bbox=self.grid.cell_rect(1, index, anchor), kind="direction"))
        else:
            occupied: set[int] = set()
            for target in self.targets:
                index = self.grid.cell_of(level, *target.center, anchor)
                if index in selected:
                    returned.append(target)
                    occupied.add(index)
            for index in cells:
                if index not in occupied:
                    returned.append(Target(id=f"cell:{level}:{index}", label=f"empty area, {self.grid.cell_name(level, index)}",
                                           verbs=(), source="contract", bbox=self.grid.cell_rect(level, index, anchor),
                                           kind="positional"))

        edges: list[Target] = []
        if self.caps.pointer == "absolute":
            edges = [
                Target(id="edge:top", label="top edge (scroll up)", verbs=(), bbox=(0.0, 0.0, 1.0, 0.02), source="contract", kind="edge"),
                Target(id="edge:bottom", label="bottom edge (scroll down)", verbs=(), bbox=(0.0, 0.98, 1.0, 1.0), source="contract", kind="edge"),
            ]

        navigation = self._navigation_options(returned + edges)
        action = self._action_options()
        self.observation = Observation(
            image=self.frame.image, level=level, cells=cells, targets=returned, at=self.at,
            navigation=navigation, action=action, destination=self.destination, step=self.step,
        )
        return self.observation

    def _navigation_options(self, targets: list[Target]) -> dict[str, list[Option]]:
        target_options = [Option(t.id, t.describe()) for t in targets]
        if self.caps.pointer == "absolute":
            return {"navigate": [Option(HOLD, "stay where you are")] + target_options}
        questions = {
            axis: [Option(HOLD, "hold")] + [Option(name, name) for name in inputs if name != HOLD]
            for axis, inputs in self.caps.movement_axes.items()
        }
        questions["destination"] = [Option(KEEP, "keep the current destination")] + target_options
        return questions

    def _action_options(self) -> list[Option]:
        options = [Option(NONE, "do nothing")]
        if self.at is not None:
            options += [Option(f"{verb}@{self.at.id}", f"{verb} {self.at.label}") for verb in self.at.verbs]
        options += [Option(f"{verb}@self", verb) for verb in self.self_verbs]
        return options

    # ------------------------------------------------------------------ acting
    def begin_step(self) -> None:
        """Start the protected core's timer. The brain calls this before deciding."""
        self.core.start()

    def act(self, navigation: dict[str, str], action: str = NONE, arg: Optional[str] = None) -> Outcome:
        """Execute both decision models' choices in the same step (action first, then navigation)."""
        if self.observation is None:
            raise RuntimeError("observe() must be called before act()")
        anchor_before = self.frame.anchor
        results: list[dict] = []
        native: dict[str, Any] = {}

        action_outcome = self._do_action(action, arg, results, native)
        nav_outcome = self._do_navigation(navigation, results, native)

        image_before = self.frame.image
        self.sense()
        changed = _changed(image_before, self.frame.image) or any(r.get("reward") for r in results)
        if action_outcome.status == "done":
            action_outcome.changed = changed
        if self.caps.pointer == "relative" and nav_outcome.status == "moving" and self.destination is not None:
            nav_outcome = self._relative_status(nav_outcome)

        for r in results:
            self.env_done = self.env_done or bool(r.get("done"))
            if r.get("success") is not None:
                self.env_success = bool(r["success"])
        self.step += 1
        self.last_exec = {"anchor_before": anchor_before, "anchor_after": self.frame.anchor, "native": native}
        return Outcome(
            navigation=nav_outcome,
            action=action_outcome,
            step=self.step,
            latency_ms=self.core.stop(),
            env_reward=float(sum(r.get("reward", 0.0) for r in results)),
            env_done=self.env_done,
            env_success=self.env_success,
        )

    def _find(self, target_id: str) -> Optional[Target]:
        if target_id == "self":
            return self.self_target()
        for target in self.targets:
            if target.id == target_id:
                return target
        if self.observation is not None:
            for target in self.observation.targets:
                if target.id == target_id:
                    return target
        return None

    def _do_action(self, action: str, arg: Optional[str], results: list[dict], native: dict) -> ActionOutcome:
        if not action or action == NONE:
            return ActionOutcome("none", False, "no action")
        verb, _, target_id = action.partition("@")
        target = self._find(target_id)  # revalidation: the target must still exist
        if target is None:
            return ActionOutcome("failed", False, f"target {target_id} no longer exists")
        if target.kind == "element" and verb not in target.verbs:
            return ActionOutcome("failed", False, f"{target.label!r} does not support {verb}")
        allowed, reason = self.core.allow(verb, target)
        if not allowed:
            return ActionOutcome("failed", False, reason)
        adapter = self._by_name.get(target.source, self.primary)
        res = adapter.invoke(target.handle, verb, arg)
        results.append(res)
        native["action"] = {"adapter": adapter.name, "verb": verb, "target": target.id, "arg": arg}
        if not res.get("ok"):
            return ActionOutcome("failed", False, res.get("text") or f"{verb} failed")
        return ActionOutcome("done", True, res.get("text") or f"{verb} {target.label}")

    def _do_navigation(self, navigation: dict[str, str], results: list[dict], native: dict) -> NavOutcome:
        navigation = navigation or {}
        if self.caps.pointer == "absolute":
            choice = navigation.get("navigate", HOLD)
            if choice == HOLD:
                return NavOutcome("held", "stayed in place")
            target = self._find(choice)
            if target is None:
                return NavOutcome("blocked", f"target {choice} no longer exists")
            if target.kind == "edge":
                verb = "scroll_up" if target.id == "edge:top" else "scroll_down"
                res = self.primary.invoke(None, verb, None)
                results.append(res)
                native["navigation"] = {"scroll": verb}
                return NavOutcome("reached" if res.get("ok") else "blocked", res.get("text") or verb.replace("_", " "))
            x, y = target.center
            res = self.primary.point(x, y)
            results.append(res)
            native["navigation"] = {"point": [x, y]}
            if not res.get("ok"):
                return NavOutcome("blocked", res.get("text") or "pointer could not move")
            self.at = target if target.kind == "element" else None
            return NavOutcome("reached", f"pointer at {target.label}")

        # relative mode: movement inputs per axis plus an optional destination change
        choice = navigation.get("destination", KEEP)
        if choice != KEEP:
            self.destination = self._find(choice)
        inputs = {axis: value for axis, value in navigation.items() if axis != "destination" and value != HOLD}
        if not inputs:
            return NavOutcome("held", "held position")
        res = self.primary.move(inputs)
        results.append(res)
        native["navigation"] = {"move": inputs}
        if res.get("blocked") or not res.get("ok"):
            return NavOutcome("blocked", res.get("text") or "movement blocked")
        return NavOutcome("moving", res.get("text") or ", ".join(f"{k} {v}" for k, v in inputs.items()))

    def _relative_status(self, outcome: NavOutcome) -> NavOutcome:
        target = self._find(self.destination.id) if self.destination else None
        if target is None:
            return outcome
        self._set_bearing(target)
        if target.distance == "near" and abs(target.bearing or 0) < 10:
            self.at = target
            return NavOutcome("reached", f"reached {target.label}")
        return NavOutcome("moving", f"{target.label} now {target.describe()}")

    # ------------------------------------------------------------------ training and checks
    def describe_source(self) -> str:
        """Raw source for the labeler (training only)."""
        for adapter in self.adapters:
            if adapter.capabilities().has_source:
                text = adapter.source()
                if text:
                    return text
        return "\n".join(f"{t.id}: {t.label} ({', '.join(t.verbs)})" for t in self.targets)

    def check(self, condition: dict | None) -> bool:
        return self.core.check(condition, CheckState(targets=self.targets, env_success=self.env_success))


def _changed(before: np.ndarray, after: np.ndarray) -> bool:
    if before is None or after is None or before.shape != after.shape:
        return True
    return float(np.mean(np.abs(before.astype(np.int16) - after.astype(np.int16)))) > 0.5
