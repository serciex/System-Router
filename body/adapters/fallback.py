"""Universal fallback adapter: the whole window is one 2D surface, input through mouse and keyboard.

Works over any screen driver exposing `click_at`, `point`, and optionally `double_click_at`,
`right_click_at`, `drag`, `type_text`, `press_key`, `scroll`. Positions come from the narrowing library.
"""

from __future__ import annotations

from typing import Any, Optional

from ..schema import InteractionManifest, Item, Point
from .base import Adapter, result

WINDOW = "window"
_POINT_VERBS = {"click": "click_at", "double_click": "double_click_at", "right_click": "right_click_at",
                "hover": "point"}


class FallbackAdapter(Adapter):
    name = "fallback"

    def __init__(self, driver):
        self.driver = driver

    def _has(self, method: str) -> bool:
        return callable(getattr(self.driver, method, None))

    def manifest(self) -> InteractionManifest:
        surface = tuple(v for v, m in _POINT_VERBS.items() if self._has(m)) + (("drag",) if self._has("drag") else ())
        self_verbs = tuple(v for v, m in (("type", "type_text"), ("press_key", "press_key"),
                                          ("scroll_up", "scroll"), ("scroll_down", "scroll")) if self._has(m))
        return InteractionManifest(pointer_mode="absolute", verbs=surface + self_verbs, self_verbs=self_verbs,
                                   actuators={"pointer": {}, "keyboard": {}})

    def find(self, scope: Any = None) -> list[Item]:
        verbs = tuple(v for v in self.manifest().verbs if v not in self.manifest().self_verbs)
        return [Item(handle=WINDOW, kind="surface", role="surface", name="whole window", verbs=verbs,
                     bbox=(0.0, 0.0, 1.0, 1.0), dims=2, confidence=0.5)]

    def act_at(self, handle, points: list[Point], verb: str) -> dict:
        if verb == "drag" and len(points) >= 2 and self._has("drag"):
            return {**self.driver.drag(points[0], points[1]), "text": "dragged"}
        method = _POINT_VERBS.get(verb)
        if method and self._has(method):
            return {**getattr(self.driver, method)(*points[0]), "text": verb.replace("_", " ")}
        return result(False, f"unsupported verb {verb}")

    def invoke(self, handle, verb: str, arg: Optional[str] = None) -> dict:
        if verb == "type" and self._has("type_text"):
            return {**self.driver.type_text(arg or ""), "text": f'typed "{arg or ""}"'}
        if verb == "press_key" and self._has("press_key"):
            return {**self.driver.press_key(arg or ""), "text": f"pressed {arg}"}
        if verb in ("scroll_up", "scroll_down") and self._has("scroll"):
            return {**self.driver.scroll(verb.split("_")[1]), "text": verb.replace("_", " ")}
        return result(False, f"unsupported verb {verb}")
