"""A fake environment and adapters so the contract can be tested without a browser or a model."""

from __future__ import annotations

from typing import Optional

import numpy as np

from body.adapters.base import Adapter, result
from body.schema import Capabilities, Frame, NativeElement


class FakePage:
    """Three elements: a button, a text field and a link. Clicking the button finishes the task."""

    def __init__(self):
        self.reset()

    def reset(self, seed: Optional[int] = None):
        self.pointer = (0.5, 0.5)
        self.typed = ""
        self.clicked: list[int] = []
        self.done = False
        self.image = np.zeros((60, 80, 3), dtype=np.uint8)

    def elements(self) -> list[NativeElement]:
        return [
            NativeElement(1, (0.1, 0.1, 0.3, 0.2), "Submit (button)", "button", ("click",)),
            NativeElement(2, (0.1, 0.5, 0.6, 0.6), "Username (text)", "input_text", ("click", "type"), value=self.typed),
            NativeElement(3, (0.7, 0.8, 0.9, 0.9), "Help", "a", ("click",)),
        ]


class FakeAdapter(Adapter):
    name = "fake"

    def __init__(self, page: FakePage, stable: bool = True):
        self.page = page
        self.stable = stable

    def capabilities(self) -> Capabilities:
        return Capabilities(pointer="absolute", verbs=("click", "type", "scroll_up", "scroll_down"), stable_ids=self.stable,
                            has_source=True)

    def reset(self, seed=None) -> Frame:
        self.page.reset(seed)
        return self.read()

    def read(self) -> Frame:
        return Frame(image=self.page.image.copy(), anchor=self.page.pointer, elements=self.page.elements(),
                     text="Type alice and press Submit")

    def invoke(self, handle, native_action, arg=None) -> dict:
        if native_action in ("scroll_up", "scroll_down"):
            return result(True, native_action)
        if native_action == "click":
            self.page.clicked.append(handle)
            if handle == 1:
                self.page.done = True
                self.page.image[:] = 255
                return result(True, "clicked", reward=1.0, done=True, success=True)
            return result(True, "clicked")
        if native_action == "type" and handle == 2:
            self.page.typed = arg or ""
            self.page.image[0, 0] = 128
            return result(True, f"typed {arg}")
        return result(False, "unsupported")

    def point(self, x, y) -> dict:
        self.page.pointer = (x, y)
        return result(True, "moved")

    def source(self) -> Optional[str]:
        return "<button>Submit</button><input id=username><a>Help</a>"


class FakeRelativeAdapter(Adapter):
    """A first-person view: turning shifts the door's position; walking makes it bigger."""

    name = "fake_relative"

    def __init__(self):
        self.reset()

    def capabilities(self) -> Capabilities:
        return Capabilities(pointer="relative", verbs=("use",), movement_axes={"move": ("forward", "back"), "turn": ("left", "right")},
                            stable_ids=True, fov_deg=90.0)

    def reset(self, seed=None) -> Frame:
        self.x, self.size = 0.8, 0.1
        return self.read()

    def read(self) -> Frame:
        half = self.size / 2
        box = (max(self.x - half, 0.0), max(0.5 - half, 0.0), min(self.x + half, 1.0), min(0.5 + half, 1.0))
        image = np.full((40, 40, 3), int(self.x * 100), dtype=np.uint8)
        return Frame(image=image, anchor=(0.5, 0.5), elements=[NativeElement("door", box, "door", "door", ("use",))])

    def invoke(self, handle, native_action, arg=None) -> dict:
        return result(native_action == "use", "used door")

    def move(self, inputs) -> dict:
        if inputs.get("turn") == "right":
            self.x = max(self.x - 0.15, 0.5)
        if inputs.get("turn") == "left":
            self.x = min(self.x + 0.15, 1.0)
        if inputs.get("move") == "forward":
            self.size = min(self.size + 0.15, 0.9)
        return result(True, "moved")
