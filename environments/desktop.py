"""Real desktop integration and input driver (screen via mss, input via pyautogui).

Needs a display; on Colab or a server run it inside a virtual display (Xvfb). Sandboxed machines only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from body.schema import Observation, Screen, ScreenSpec, SensoryManifest

from .base import Integration


def _ok(text: str = "") -> dict:
    return {"ok": True, "text": text, "reward": 0.0, "done": False, "success": None, "blocked": False}


class DesktopEnvironment(Integration):
    def __init__(self, monitor: int = 1):
        import mss
        import pyautogui

        pyautogui.FAILSAFE = True
        self._gui = pyautogui
        self._mss = mss.mss()
        self._monitor = self._mss.monitors[monitor]

    # ------------------------------------------------------------------ sensory channel
    def reset(self, seed: Optional[int] = None) -> None:
        pass

    @property
    def size(self) -> tuple[int, int]:
        return self._monitor["width"], self._monitor["height"]

    @property
    def pointer(self) -> tuple[float, float]:
        x, y = self._gui.position()
        w, h = self.size
        return (x - self._monitor["left"]) / w, (y - self._monitor["top"]) / h

    @property
    def image(self) -> np.ndarray:
        shot = np.asarray(self._mss.grab(self._monitor))
        return shot[:, :, 2::-1].copy()  # BGRA to RGB

    def observe(self) -> Observation:
        return Observation(screens=[Screen("flat", self.image)], pointer=self.pointer)

    def sensory_manifest(self) -> SensoryManifest:
        w, h = self.size
        return SensoryManifest(screens=[ScreenSpec("flat", w, h)], realtime=True)

    # ------------------------------------------------------------------ input driver
    def _px(self, x: float, y: float) -> tuple[int, int]:
        w, h = self.size
        return int(self._monitor["left"] + x * w), int(self._monitor["top"] + y * h)

    def point(self, x: float, y: float) -> dict:
        self._gui.moveTo(*self._px(x, y))
        return _ok()

    def click_at(self, x: float, y: float) -> dict:
        self._gui.click(*self._px(x, y))
        return _ok()

    def double_click_at(self, x: float, y: float) -> dict:
        self._gui.doubleClick(*self._px(x, y))
        return _ok()

    def right_click_at(self, x: float, y: float) -> dict:
        self._gui.rightClick(*self._px(x, y))
        return _ok()

    def drag(self, start: tuple[float, float], end: tuple[float, float]) -> dict:
        self._gui.moveTo(*self._px(*start))
        self._gui.dragTo(*self._px(*end), duration=0.2)
        return _ok()

    def type_text(self, text: str) -> dict:
        self._gui.write(text, interval=0.01)
        return _ok()

    def press_key(self, key: str) -> dict:
        keys = [k.strip() for k in key.split("+") if k.strip()]
        self._gui.hotkey(*keys) if len(keys) > 1 else self._gui.press(keys[0])
        return _ok()

    def scroll(self, direction: str) -> dict:
        self._gui.scroll(5 if direction == "up" else -5)
        return _ok()
