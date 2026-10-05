"""Adapter interface (CONTRACT.md v0.3). The interaction channel only.

Every native call returns a result dict:
    {"ok": bool, "text": str, "reward": float, "done": bool, "success": bool | None, "blocked": bool}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..schema import CONTRACT_VERSION, InteractionManifest, Item, Point


def result(ok: bool, text: str = "", reward: float = 0.0, done: bool = False, success: Optional[bool] = None,
           blocked: bool = False) -> dict:
    return {"ok": ok, "text": text, "reward": reward, "done": done, "success": success, "blocked": blocked}


class Adapter(ABC):
    name: str = "adapter"
    contract_version: str = CONTRACT_VERSION

    @abstractmethod
    def manifest(self) -> InteractionManifest:
        """Interaction half of the manifest."""

    @abstractmethod
    def find(self, scope: Any = None) -> list[Item]:
        """Visible items in the window, or inside the scope (a group handle or a region)."""

    @abstractmethod
    def invoke(self, handle: Any, verb: str, arg: Optional[str] = None) -> dict:
        """Run a verb on an item; handle None means the self target."""

    def act_at(self, handle: Any, points: list[Point], verb: str) -> dict:
        """Run a verb at window-normalized points on a surface."""
        return result(False, f"{self.name} has no surfaces")

    def move(self, inputs: dict[str, str]) -> dict:
        """Relative mode: one input per declared axis."""
        return result(False, f"{self.name} does not support relative movement")

    def reset(self, seed: Optional[int] = None) -> None:
        """Clear adapter state for a new episode (the environment is reset by integration)."""

    def close(self) -> None:
        pass
