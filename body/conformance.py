"""Conformance checks for v0.3 adapters (CONTRACT.md). Run by the protected core, never by the LLM.

`run_conformance` runs the public checks plus every module in the hidden split directory. A hidden
module defines `checks(adapter, integration) -> dict[str, tuple[bool, str]]` and lives outside any workspace.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .adapters.base import Adapter
from .core import changed
from .schema import CONTRACT_VERSION, Item
from .vocab import POINTER_MODES, ROLES, VERBS, meaningful

Check = tuple[bool, str]


@dataclass
class Report:
    adapter: str
    results: dict[str, Check] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.results.values())

    def __str__(self) -> str:
        lines = [f"Conformance for {self.adapter}: {'PASS' if self.passed else 'FAIL'}"]
        lines += [f"  [{'ok' if ok else 'XX'}] {name}: {detail}" for name, (ok, detail) in self.results.items()]
        return "\n".join(lines)


def check_manifest(adapter: Adapter) -> Check:
    m = adapter.manifest()
    if m.contract_version != CONTRACT_VERSION:
        return False, f"targets contract {m.contract_version}, expected {CONTRACT_VERSION}"
    if m.pointer_mode not in POINTER_MODES:
        return False, f"unknown pointer mode {m.pointer_mode!r}"
    bad = [v for v in m.verbs + m.self_verbs if v not in VERBS]
    return (not bad), (f"non-canonical verbs {bad}" if bad else f"{len(m.verbs)} verbs")


def check_items(items: list[Item]) -> Check:
    for item in items:
        x0, y0, x1, y1 = item.bbox
        if not (0.0 <= x0 <= x1 <= 1.0 and 0.0 <= y0 <= y1 <= 1.0):
            return False, f"{item.name!r} box outside the window: {item.bbox}"
        if item.role not in ROLES:
            return False, f"{item.name!r} has non-canonical role {item.role!r}"
        if any(v not in VERBS for v in item.verbs):
            return False, f"{item.name!r} has non-canonical verbs {item.verbs}"
        if item.kind not in ("element", "surface", "group"):
            return False, f"{item.name!r} has unknown kind {item.kind!r}"
        if item.kind == "surface" and item.dims not in (1, 2):
            return False, f"surface {item.name!r} needs dims 1 or 2"
        if not meaningful(item.name):
            return False, f"item name {item.name!r} is not meaningful"
        if not 0.0 <= item.confidence <= 1.0:
            return False, f"{item.name!r} confidence {item.confidence}"
    return True, f"{len(items)} items"


def check_stable(adapter: Adapter) -> Check:
    first = [(repr(i.handle), i.name) for i in adapter.find()]
    second = [(repr(i.handle), i.name) for i in adapter.find()]
    return first == second, "same handles for an unchanged window" if first == second else "handles changed with no action"


def check_scope(adapter: Adapter) -> Check:
    groups = [i for i in adapter.find() if i.kind == "group"]
    if not groups:
        return True, "no groups to expand"
    children = adapter.find(groups[0].handle)
    ok = bool(children) and len(children) <= max(groups[0].collapsed, len(children))
    return ok, f"expanding {groups[0].name!r} returned {len(children)} items"


def check_surface(adapter: Adapter, integration, tolerance: float = 0.05) -> Check:
    surfaces = [i for i in adapter.find() if i.kind == "surface" and "hover" in i.verbs]
    if not surfaces:
        return True, "no hoverable surfaces"
    s = surfaces[0]
    target = ((s.bbox[0] + s.bbox[2]) / 2, (s.bbox[1] + s.bbox[3]) / 2)
    adapter.act_at(s.handle, [target], "hover")
    pointer = integration.observe().pointer
    if pointer is None:
        return True, "integration has no pointer readback"
    ok = abs(pointer[0] - target[0]) <= tolerance and abs(pointer[1] - target[1]) <= tolerance
    return ok, f"pointer at {pointer}, requested {target}"


def check_faithful(adapter: Adapter, integration, handle, verb: str, arg: Optional[str] = None) -> Check:
    before = integration.observe().image
    res = adapter.invoke(handle, verb, arg)
    after = integration.observe().image
    ok = bool(res.get("ok")) and (changed(before, after) or bool(res.get("reward")) or bool(res.get("done")))
    return ok, f"{verb} -> {res.get('text')}"


def check_coverage(adapter: Adapter, integration, detector: Callable, minimum: float = 0.6) -> Check:
    detected = detector(integration.observe().image)
    if not detected:
        return True, "detector found nothing"
    items = adapter.find()

    def covered(box) -> bool:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return any(i.bbox[0] <= cx <= i.bbox[2] and i.bbox[1] <= cy <= i.bbox[3] for i in items)

    fraction = sum(covered(d["bbox"]) for d in detected) / len(detected)
    return fraction >= minimum, f"{fraction:.0%} of detected elements covered"


def check_reset(adapter: Adapter, integration, seed: int = 0) -> Check:
    integration.reset(seed)
    adapter.reset(seed)
    a = sorted(i.name for i in adapter.find())
    integration.reset(seed)
    adapter.reset(seed)
    b = sorted(i.name for i in adapter.find())
    return a == b, "same first state for the same seed" if a == b else "reset is not reproducible"


def _hidden(directory: Optional[str | Path], adapter: Adapter, integration) -> dict[str, Check]:
    results: dict[str, Check] = {}
    if not directory or not Path(directory).exists():
        return results
    for path in sorted(Path(directory).glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"hidden_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, outcome in module.checks(adapter, integration).items():
            results[f"hidden:{name}"] = outcome
    return results


def run_conformance(adapter: Adapter, integration, seed: int = 0, hidden_dir: Optional[str | Path] = None,
                    detector: Optional[Callable] = None) -> Report:
    report = Report(adapter.name)
    report.results["manifest"] = check_manifest(adapter)
    report.results["reset"] = check_reset(adapter, integration, seed)
    report.results["items"] = check_items(adapter.find())
    report.results["stable"] = check_stable(adapter)
    report.results["scope"] = check_scope(adapter)
    report.results["surface"] = check_surface(adapter, integration)
    if detector is not None:
        report.results["coverage"] = check_coverage(adapter, integration, detector)
    report.results.update(_hidden(hidden_dir, adapter, integration))
    return report
