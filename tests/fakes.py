"""Fake integration and adapters for contract v0.3 tests (no browser, no model)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from body.adapters.base import Adapter, result
from body.schema import InteractionManifest, Item, Observation, Screen, ScreenSpec, SensoryManifest
from environments.base import Integration


class FakePage(Integration):
    """A page with a button, a text field, a link, a 15-item list and a canvas."""

    def __init__(self):
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        self.pointer = (0.5, 0.5)
        self.typed = ""
        self.done = False
        self.removed: set = set()
        self.points: list = []
        self.image = np.zeros((60, 80, 3), dtype=np.uint8)

    def touch(self, value: int) -> None:
        self.image[0, 0] = value
        self.image[10:20, 10:20] = value

    def observe(self) -> Observation:
        return Observation(screens=[Screen("flat", self.image.copy())], pointer=self.pointer)

    def sensory_manifest(self) -> SensoryManifest:
        return SensoryManifest(screens=[ScreenSpec("flat", 80, 60)])

    @property
    def goal(self) -> str:
        return "Type alice and press Submit"


class FakeWebAdapter(Adapter):
    name = "fake_web"

    def __init__(self, page: FakePage):
        self.page = page

    def manifest(self) -> InteractionManifest:
        return InteractionManifest(pointer_mode="absolute", verbs=("click", "type", "hover", "drag", "scroll_up"),
                                   self_verbs=("scroll_up",))

    def find(self, scope=None) -> list[Item]:
        if scope == 10:
            return [Item(100 + k, "element", "option", f"item {k}", ("click",), (0.6, 0.05 * k, 0.9, 0.05 * k + 0.04),
                         container=10) for k in range(15)]
        items = [
            Item(1, "element", "button", "Submit", ("click", "hover"), (0.1, 0.1, 0.3, 0.2)),
            Item(2, "element", "textbox", "Username", ("click", "type", "hover"), (0.1, 0.5, 0.5, 0.6), value=self.page.typed),
            Item(3, "element", "link", "Help", ("click",), (0.1, 0.8, 0.3, 0.9)),
            Item(10, "group", "group", "List", (), (0.6, 0.0, 0.9, 0.75), collapsed=15),
            Item(20, "surface", "surface", "canvas", ("click", "drag", "hover"), (0.35, 0.3, 0.55, 0.45), dims=2),
        ]
        return [i for i in items if i.handle not in self.page.removed]

    def invoke(self, handle, verb, arg=None) -> dict:
        if verb == "scroll_up":
            return result(True, "scrolled")
        if handle in self.page.removed:
            return result(False, "gone")
        if verb == "click" and handle == 1:
            self.page.done = True
            self.page.touch(255)
            return result(True, "clicked", reward=1.0, done=True, success=True)
        if verb == "type" and handle == 2:
            self.page.typed = arg or ""
            self.page.touch(128)
            return result(True, f"typed {arg}")
        if verb in ("click", "hover"):
            return result(True, verb)
        return result(False, "unsupported")

    def act_at(self, handle, points, verb) -> dict:
        self.page.points.append((verb, list(points)))
        if verb == "hover":
            self.page.pointer = points[0]
        return result(True, verb)


class FakeRelativeAdapter(Adapter):
    """First person: turning right brings the door to the centre, walking forward makes it bigger."""

    name = "fake_relative"

    def __init__(self):
        self.reset()

    def reset(self, seed=None) -> None:
        self.x, self.size = 0.8, 0.1

    def manifest(self) -> InteractionManifest:
        return InteractionManifest(pointer_mode="relative", verbs=("use",),
                                   movement_axes={"move": ("forward", "back"), "turn": ("left", "right")})

    def find(self, scope=None) -> list[Item]:
        half = self.size / 2
        box = (max(self.x - half, 0.0), max(0.5 - half, 0.0), min(self.x + half, 1.0), min(0.5 + half, 1.0))
        return [Item("door", "element", "entity", "door", ("use",), box)]

    def invoke(self, handle, verb, arg=None) -> dict:
        return result(verb == "use", "used door")

    def move(self, inputs) -> dict:
        if inputs.get("turn") == "right":
            self.x = max(self.x - 0.15, 0.5)
        if inputs.get("move") == "forward":
            self.size = min(self.size + 0.15, 0.9)
        return result(True, "moved")


class StillScreen(Integration):
    """A fixed screen for relative-mode tests."""

    def reset(self, seed=None) -> None:
        self.image = np.zeros((40, 40, 3), dtype=np.uint8)

    def observe(self) -> Observation:
        return Observation(screens=[Screen("flat", self.image)])

    def sensory_manifest(self) -> SensoryManifest:
        return SensoryManifest(screens=[ScreenSpec("flat", 40, 40)])
