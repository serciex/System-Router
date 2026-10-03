"""Conformance checks every adapter must pass before the world model uses it (CONTRACT.md).

Each check returns (passed, detail). `run_conformance` runs the environment-independent ones;
`check_faithful` needs a known test element and an expected state change, supplied per environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .adapters.base import Adapter
from .schema import CONTRACT_VERSION, Frame


@dataclass
class Report:
    adapter: str
    results: dict[str, tuple[bool, str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.results.values())

    def __str__(self) -> str:
        lines = [f"Conformance for {self.adapter}: {'PASS' if self.passed else 'FAIL'}"]
        lines += [f"  [{'ok' if ok else 'XX'}] {name}: {detail}" for name, (ok, detail) in self.results.items()]
        return "\n".join(lines)


def check_version(adapter: Adapter) -> tuple[bool, str]:
    version = adapter.capabilities().contract_version
    return version == CONTRACT_VERSION, f"adapter {version}, contract {CONTRACT_VERSION}"


def check_schema(frame: Frame) -> tuple[bool, str]:
    if frame.image is None or frame.image.ndim != 3 or frame.image.shape[2] != 3:
        return False, "frame image must be (H, W, 3)"
    ax, ay = frame.anchor
    if not (0.0 <= ax <= 1.0 and 0.0 <= ay <= 1.0):
        return False, f"anchor {frame.anchor} is not normalized"
    for element in frame.elements:
        x0, y0, x1, y1 = element.bbox
        if not (0.0 <= x0 <= x1 <= 1.0 and 0.0 <= y0 <= y1 <= 1.0):
            return False, f"element {element.label!r} has a box outside the view: {element.bbox}"
        if not 0.0 <= element.confidence <= 1.0:
            return False, f"element {element.label!r} has confidence {element.confidence}"
    return True, f"{len(frame.elements)} elements"


def check_stable_ids(adapter: Adapter) -> tuple[bool, str]:
    if not adapter.capabilities().stable_ids:
        return True, "adapter declares unstable ids; the contract tracks them by position"
    first = [(e.handle, e.label) for e in adapter.read().elements]
    second = [(e.handle, e.label) for e in adapter.read().elements]
    return first == second, "handles identical across two reads" if first == second else "handles changed with no action"


def check_reset(adapter: Adapter, seed: int = 0) -> tuple[bool, str]:
    first = adapter.reset(seed)
    labels_a = sorted(e.label for e in first.elements)
    second = adapter.reset(seed)
    labels_b = sorted(e.label for e in second.elements)
    return labels_a == labels_b, "same first observation for the same seed" if labels_a == labels_b else "reset is not reproducible"


def check_movement(adapter: Adapter) -> tuple[bool, str]:
    caps = adapter.capabilities()
    if caps.pointer == "absolute":
        res = adapter.point(0.25, 0.25)
        anchor = adapter.read().anchor
        ok = res.get("ok", False) and abs(anchor[0] - 0.25) < 0.05 and abs(anchor[1] - 0.25) < 0.05
        return ok, f"pointer landed at {anchor}"
    for axis, inputs in caps.movement_axes.items():
        for name in inputs:
            if name == "hold":
                continue
            before = adapter.read()
            adapter.move({axis: name})
            after = adapter.read()
            moved = before.anchor != after.anchor or float(np.mean(np.abs(
                before.image.astype(np.int16) - after.image.astype(np.int16)))) > 0.5
            if not moved:
                return False, f"movement input {axis}={name} produced no observable change"
    return True, "every movement input changed the anchor or view"


def check_faithful(adapter: Adapter, handle: Any, verb: str, expect: Callable[[Frame, dict], bool],
                   arg: Optional[str] = None) -> tuple[bool, str]:
    res = adapter.invoke(handle, verb, arg)
    ok = bool(res.get("ok")) and expect(adapter.read(), res)
    return ok, f"{verb} on {handle!r} -> {res.get('text')}"


def check_honest(adapter: Adapter, handle: Any, verb: str, truth: Callable[[], bool],
                 arg: Optional[str] = None) -> tuple[bool, str]:
    res = adapter.invoke(handle, verb, arg)
    reported, actual = bool(res.get("ok")), bool(truth())
    return reported == actual, f"reported ok={reported}, observed {actual}"


def run_conformance(adapter: Adapter, seed: int = 0) -> Report:
    report = Report(adapter.name)
    report.results["version"] = check_version(adapter)
    report.results["reset"] = check_reset(adapter, seed)
    report.results["schema"] = check_schema(adapter.read())
    report.results["stable_ids"] = check_stable_ids(adapter)
    report.results["movement"] = check_movement(adapter)
    return report
