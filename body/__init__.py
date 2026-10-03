"""The body: one contract built once, shared by every environment (see CONTRACT.md)."""

from .schema import (
    CONTRACT_VERSION,
    ActionOutcome,
    Capabilities,
    Frame,
    NativeElement,
    NavOutcome,
    Observation,
    Option,
    Outcome,
    Target,
)

__all__ = [
    "CONTRACT_VERSION",
    "ActionOutcome",
    "Capabilities",
    "Frame",
    "NativeElement",
    "NavOutcome",
    "Observation",
    "Option",
    "Outcome",
    "Target",
]
