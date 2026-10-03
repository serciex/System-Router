"""Degraded variant of any adapter: drops elements, shifts boxes and blanks labels.

Used in training so the world model has already seen imperfect adapters before the LLM writes its own.
Interaction still goes through the wrapped adapter unchanged.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Optional

from ..schema import Capabilities, Frame
from .base import Adapter


class DegradedAdapter(Adapter):
    def __init__(self, inner: Adapter, drop: float = 0.2, jitter: float = 0.02, blank_label: float = 0.2, seed: int = 0):
        self.inner = inner
        self.name = f"degraded:{inner.name}"
        self.drop, self.jitter, self.blank_label = float(drop), float(jitter), float(blank_label)
        self._rng = random.Random(seed)
        self._kept: dict = {}

    def capabilities(self) -> Capabilities:
        return self.inner.capabilities()

    def reset(self, seed: Optional[int] = None) -> Frame:
        self._kept = {}
        if seed is not None:
            self._rng.seed(seed)
        self.inner.reset(seed)
        return self.read()

    def read(self) -> Frame:
        frame = self.inner.read()
        elements = []
        for element in frame.elements:
            # The same element is consistently dropped or kept within an episode.
            keep = self._kept.setdefault(element.handle, self._rng.random() >= self.drop)
            if not keep:
                continue
            dx, dy = (self._rng.uniform(-self.jitter, self.jitter) for _ in range(2))
            x0, y0, x1, y1 = element.bbox
            box = (min(max(x0 + dx, 0.0), 1.0), min(max(y0 + dy, 0.0), 1.0), min(max(x1 + dx, 0.0), 1.0), min(max(y1 + dy, 0.0), 1.0))
            if box[0] >= box[2] or box[1] >= box[3]:
                box = element.bbox
            label = "" if self._rng.random() < self.blank_label else element.label
            elements.append(replace(element, bbox=box, label=label, confidence=min(element.confidence, 0.8)))
        return replace(frame, elements=elements)

    def invoke(self, handle, native_action: str, arg: Optional[str] = None) -> dict:
        return self.inner.invoke(handle, native_action, arg)

    def move(self, inputs: dict[str, str]) -> dict:
        return self.inner.move(inputs)

    def point(self, x: float, y: float) -> dict:
        return self.inner.point(x, y)

    def source(self) -> Optional[str]:
        return self.inner.source()

    def close(self) -> None:
        self.inner.close()
