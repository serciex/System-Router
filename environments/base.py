"""Environment integration: the sensory channel and the episode lifecycle. Read-only to the LLM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from body.schema import Observation, SensoryManifest


class Integration(ABC):
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Start a new episode."""

    @abstractmethod
    def observe(self) -> Observation:
        """Current screens and sensor readings."""

    @abstractmethod
    def sensory_manifest(self) -> SensoryManifest:
        """Sensory half of the manifest."""

    @property
    def goal(self) -> str:
        """Task text, if the environment provides one."""
        return ""

    def close(self) -> None:
        pass
