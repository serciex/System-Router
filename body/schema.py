"""Contract schemas (CONTRACT.md v0.3). Plain data, no environment code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

CONTRACT_VERSION = "0.3"

BBox = tuple[float, float, float, float]  # window-normalized (x0, y0, x1, y1)
Point = tuple[float, float]

HOLD = "hold"
NONE = "none"
KEEP = "keep"

NavStatus = Literal["reached", "moving", "blocked", "held"]
ActionStatus = Literal["done", "failed", "partial", "in_progress", "unavailable", "none"]


# ---------------------------------------------------------------- sensory channel
@dataclass
class Screen:
    kind: str  # flat | stereo | camera | depth
    image: np.ndarray  # (H, W, 3) uint8


@dataclass
class SensorReading:
    name: str
    type: str
    value: Any
    units: str = ""


@dataclass
class Observation:
    """What the sensory channel delivers each step. Never produced by an adapter."""

    screens: list[Screen]
    sensors: list[SensorReading] = field(default_factory=list)
    pointer: Optional[Point] = None
    tick: int = 0

    @property
    def image(self) -> np.ndarray:
        return self.screens[0].image


# ---------------------------------------------------------------- manifest
@dataclass
class ScreenSpec:
    kind: str
    width: int
    height: int
    fov_deg: Optional[float] = None
    fps: Optional[float] = None


@dataclass
class SensorSpec:
    name: str
    type: str
    units: str = ""
    rate: Optional[float] = None


@dataclass
class SensoryManifest:
    """Declared by environment integration, read-only to the LLM."""

    screens: list[ScreenSpec]
    sensors: list[SensorSpec] = field(default_factory=list)
    tick_ms: Optional[float] = None
    realtime: bool = False


@dataclass
class InteractionManifest:
    """Declared by the adapter and checked by conformance."""

    pointer_mode: str  # absolute | relative | none
    verbs: tuple[str, ...]
    self_verbs: tuple[str, ...] = ()  # verbs not tied to an item, e.g. type into the focused field
    actuators: dict[str, dict] = field(default_factory=dict)
    movement_axes: dict[str, tuple[str, ...]] = field(default_factory=dict)  # relative mode
    fov_deg: float = 90.0  # relative mode, for bearings
    contract_version: str = CONTRACT_VERSION


@dataclass
class Manifest:
    body_id: str
    version: str
    sensory: SensoryManifest
    interaction: InteractionManifest


# ---------------------------------------------------------------- interaction channel
@dataclass
class Item:
    """One thing an adapter's find reports."""

    handle: Any
    kind: str  # element | surface | group
    role: str
    name: str
    verbs: tuple[str, ...]
    bbox: BBox
    container: Any = None
    collapsed: int = 0
    dims: int = 0  # surfaces: 1 or 2
    value: str = ""
    reversible: Optional[bool] = None
    confidence: float = 1.0


@dataclass
class Target:
    """An item as the brain sees it: stable id, canonical label. `handle` and `point` are internal."""

    id: str
    label: str
    kind: str
    verbs: tuple[str, ...]
    bbox: BBox
    handle: Any = None
    dims: int = 0
    collapsed: int = 0
    value: str = ""
    reversible: Optional[bool] = None
    confidence: float = 1.0
    bearing: Optional[float] = None
    distance: Optional[str] = None
    point: Optional[Point] = None  # set when a position on a surface was chosen

    @property
    def center(self) -> Point:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def describe(self) -> str:
        text = self.label
        if self.value:
            text += f' [value "{self.value}"]'
        if self.kind == "group" and self.collapsed:
            text += f" ({self.collapsed} more inside)"
        if self.bearing is not None:
            side = "ahead" if abs(self.bearing) < 5 else f"{abs(self.bearing):.0f} deg {'left' if self.bearing < 0 else 'right'}"
            text += f", {side}" + (f", {self.distance}" if self.distance else "")
        if self.confidence < 1.0:
            text += ", uncertain"
        return text


@dataclass
class Slot:
    """The body slot: the latest find result."""

    targets: list[Target]
    scope: Optional[str] = None
    tick: int = 0


@dataclass
class Option:
    """One choice offered to the model. `key` is executed, `text` is read."""

    key: str
    text: str


# ---------------------------------------------------------------- outcomes
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
