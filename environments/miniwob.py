"""MiniWoB++ (Farama `miniwob` package): small web tasks in headless Chrome via Selenium.

Integration for the sensory channel (screen, pointer, goal, episode lifecycle) plus the native
operations adapters use. Coordinates passed in are normalized and converted to pixels here.

Install: `pip install miniwob` plus Chrome/Chromium and a matching chromedriver.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np

from body.schema import Observation, Screen, ScreenSpec, SensoryManifest

from .base import Integration


class MiniWoBEnvironment(Integration):
    def __init__(self, tasks: list[str], render: bool = False, seed: int = 0):
        import gymnasium
        import miniwob

        gymnasium.register_envs(miniwob)
        self._gym = gymnasium
        self.tasks = list(tasks)
        self.render = render
        self._rng = random.Random(seed)
        self.env = None
        self.task: Optional[str] = None
        self.obs: dict[str, Any] = {}
        self.done = False
        self.success: Optional[bool] = None
        self.pointer = (0.5, 0.5)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: Optional[int] = None, task: Optional[str] = None) -> dict:
        if seed is not None:
            self._rng.seed(seed)
        task = task or self._rng.choice(self.tasks)
        if self.env is None or task != self.task:
            self.close()
            self.env = self._gym.make(
                f"miniwob/{task}-v1",
                render_mode="human" if self.render else None,
                action_space_config="all_supported",
            )
            self.task = task
        self.obs, _ = self.env.reset(seed=seed)
        self.done, self.success = False, None
        self.pointer = (0.5, 0.5)
        return self.obs

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    # ------------------------------------------------------------------ sensory channel
    def observe(self) -> Observation:
        return Observation(screens=[Screen("flat", self.image)], pointer=self.pointer)

    def sensory_manifest(self) -> SensoryManifest:
        width, height = self.size
        return SensoryManifest(screens=[ScreenSpec("flat", width, height)], realtime=False)

    @property
    def goal(self) -> str:
        return self.utterance

    # ------------------------------------------------------------------ properties
    @property
    def image(self) -> np.ndarray:
        return np.asarray(self.obs.get("screenshot"), dtype=np.uint8)

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels of the screenshot, which matches the DOM coordinates."""
        height, width = self.image.shape[:2]
        return width, height

    @property
    def utterance(self) -> str:
        return str(self.obs.get("utterance", ""))

    @property
    def dom_elements(self) -> tuple[dict, ...]:
        return tuple(self.obs.get("dom_elements", ()))

    @property
    def driver(self):
        """The Selenium driver if the installed miniwob version exposes it, else None."""
        instance = getattr(self.env.unwrapped, "instance", None) if self.env is not None else None
        return getattr(instance, "driver", None)

    # ------------------------------------------------------------------ native operations
    def step(self, action_type: str, **fields) -> dict:
        """Run one MiniWoB action. Returns the native result in the adapter result format."""
        from miniwob.action import ActionTypes

        if self.done:
            return {"ok": False, "text": "episode already finished", "reward": 0.0, "done": True,
                    "success": self.success, "blocked": True}
        action = self.env.unwrapped.create_action(getattr(ActionTypes, action_type), **fields)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.done = bool(terminated or truncated)
        if terminated:
            raw = info.get("raw_reward", reward) if isinstance(info, dict) else reward
            self.success = float(raw) > 0
        return {"ok": True, "text": "", "reward": float(reward), "done": self.done, "success": self.success,
                "blocked": False}

    def to_pixels(self, x: float, y: float) -> np.ndarray:
        width, height = self.size
        return np.array([x * width, y * height], dtype=np.float32)

    def point(self, x: float, y: float) -> dict:
        res = self.step("MOVE_COORDS", coords=self.to_pixels(x, y))
        if res["ok"]:
            self.pointer = (x, y)
        return res

    def click_at(self, x: float, y: float) -> dict:
        res = self.step("CLICK_COORDS", coords=self.to_pixels(x, y))
        if res["ok"]:
            self.pointer = (x, y)
        return res

    def double_click_at(self, x: float, y: float) -> dict:
        res = self.step("DBLCLICK_COORDS", coords=self.to_pixels(x, y))
        if res["ok"]:
            self.pointer = (x, y)
        return res

    def drag(self, start: tuple[float, float], end: tuple[float, float]) -> dict:
        for name, point in (("MOUSEDOWN_COORDS", start), ("MOVE_COORDS", end), ("MOUSEUP_COORDS", end)):
            res = self.step(name, coords=self.to_pixels(*point))
            if not res["ok"] or res["done"]:
                return res
        self.pointer = end
        return res

    def click_ref(self, ref: int) -> dict:
        return self.step("CLICK_ELEMENT", ref=int(ref))

    def type_into_ref(self, ref: int, text: str) -> dict:
        return self.step("FOCUS_ELEMENT_AND_TYPE_TEXT", ref=int(ref), text=str(text))

    def type_text(self, text: str) -> dict:
        return self.step("TYPE_TEXT", text=str(text))

    def scroll(self, direction: str) -> dict:
        name = "SCROLL_UP_COORDS" if direction == "up" else "SCROLL_DOWN_COORDS"
        return self.step(name, coords=self.to_pixels(*self.pointer))

    def page_source(self) -> Optional[str]:
        driver = self.driver
        if driver is not None:
            try:
                return driver.page_source
            except Exception:  # noqa: BLE001 - any Selenium failure falls back to the DOM list
                pass
        return None
