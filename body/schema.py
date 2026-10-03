"""Contract schemas (CONTRACT.md v0.2). Plain data, no environment code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

CONTRACT_VERSION = "0.2"

BBox = tuple[float, float, float, float]  # normalized (x0, y0, x1, y1)

Pointer = Literal["absolute", "relative"]
NavStatus = Literal["reached", "moving", "blocked", "held"]
ActionStatus = Literal["done", "failed", "partial", "in_progress", "none"]

HOLD = "hold"
NONE = "none"


@dataclass
class NativeElement:
    """One element as an adapter reports it, before the contract turns it into a Target."""

    handle: Any
    bbox: BBox
    label: str
    role: str = ""
    native_actions: tuple[str, ...] = ()
    value: str = ""
    reversible: Optional[bool] = None
    confidence: float = 1.0


@dataclass
class Frame:
    """What `Adapter.read()` returns."""

    image: np.ndarray  # (H, W, 3) uint8
    anchor: tuple[float, float]
    elements: list[NativeElement]
    text: str = ""  # task text if the environment shows one
    info: dict = field(default_factory=dict)


@dataclass
class Target:
    """A thing the body can move to. `source`, `handle` and `kind` are internal, never shown to the brain."""

    id: str
    label: str
    verbs: tuple[str, ...]
    bbox: BBox
    reversible: Optional[bool] = None
    confidence: float = 1.0
    bearing: Optional[float] = None
    distance: Optional[str] = None
    value: str = ""
    source: str = ""
    handle: Any = None
    kind: str = "element"  # element | self | direction | positional | edge

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def describe(self) -> str:
        """The one-line text the decision models read."""
        text = self.label or "(unlabeled)"
        if self.value:
            text += f' [value "{self.value}"]'
        if self.bearing is not None:
            side = "ahead" if abs(self.bearing) < 5 else (f"{abs(self.bearing):.0f}° {'left' if self.bearing < 0 else 'right'}")
            text += f", {side}"
            if self.distance:
                text += f", {self.distance}"
        if self.confidence < 1.0:
            text += ", uncertain"
        return text

    def public(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "verbs": list(self.verbs),
            "bbox": [round(v, 4) for v in self.bbox],
            "reversible": self.reversible,
            "confidence": round(self.confidence, 3),
            "bearing": self.bearing,
            "distance": self.distance,
        }


@dataclass
class Option:
    """One choice offered to a decision model. `key` is what the body executes, `text` what the LLM reads."""

    key: str
    text: str


@dataclass
class Capabilities:
    pointer: Pointer
    verbs: tuple[str, ...]
    movement_axes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    stable_ids: bool = True
    realtime: bool = False
    has_source: bool = False
    fov_deg: float = 90.0  # relative mode: horizontal field of view used for bearings
    contract_version: str = CONTRACT_VERSION


@dataclass
class Observation:
    """What `Body.observe(level, cells)` returns."""

    image: np.ndarray
    level: int
    cells: tuple[int, ...]
    targets: list[Target]
    at: Optional[Target]
    navigation: dict[str, list[Option]]  # question name -> options
    action: list[Option]
    destination: Optional[Target] = None
    step: int = 0


@dataclass
class NavOutcome:
    status: NavStatus
    text: str


@dataclass
class ActionOutcome:
    status: ActionStatus
    changed: bool
    text: str


@dataclass
class Outcome:
    navigation: NavOutcome
    action: ActionOutcome
    step: int
    latency_ms: float
    env_reward: float = 0.0
    env_done: bool = False
    env_success: Optional[bool] = None

    def text(self) -> str:
        return f"Navigation {self.navigation.status}: {self.navigation.text}. Action {self.action.status}: {self.action.text}."
