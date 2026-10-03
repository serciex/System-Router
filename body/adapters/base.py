"""Adapter interface (CONTRACT.md, "What an adapter provides").

An adapter must translate honestly, keep handles stable, mark pixel-inferred elements with confidence < 1,
and pass interaction straight through. It must not rank by relevance, decide anything, time itself,
or judge completion.

Every native call returns a small result dict:
    {"ok": bool, "text": str, "reward": float, "done": bool, "success": bool | None, "blocked": bool}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..schema import CONTRACT_VERSION, Capabilities, Frame


def result(ok: bool, text: str = "", reward: float = 0.0, done: bool = False, success: Optional[bool] = None,
           blocked: bool = False) -> dict:
    return {"ok": ok, "text": text, "reward": reward, "done": done, "success": success, "blocked": blocked}


class Adapter(ABC):
    name: str = "adapter"
    contract_version: str = CONTRACT_VERSION

    @abstractmethod
    def read(self) -> Frame:
        """Current frame, anchor and native elements."""

    @abstractmethod
    def invoke(self, handle: Any, native_action: str, arg: Optional[str] = None) -> dict:
        """Run the environment's own function for a verb on an element. `handle=None` means the `self` target."""

    def move(self, inputs: dict[str, str]) -> dict:
        """Relative mode only: send movement inputs, one per axis."""
        raise NotImplementedError(f"{self.name} does not support relative movement")

    def point(self, x: float, y: float) -> dict:
        """Absolute mode only: move the pointer to a normalized position."""
        raise NotImplementedError(f"{self.name} does not support an absolute pointer")

    def source(self) -> Optional[str]:
        """Raw source for the labeler (DOM, accessibility tree, code), or None."""
        return None

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Declared capabilities."""

    def reset(self, seed: Optional[int] = None) -> Frame:
        """Restart the episode. Only the primary adapter of a body is reset; the others re-read."""
        raise NotImplementedError

    def close(self) -> None:
        pass
