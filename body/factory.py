"""Build a body from config: one integration (sensory channel) plus one adapter (interaction channel)."""

from __future__ import annotations

from common.config import Config

from .adapters.base import Adapter
from .contract import Body
from .core import Core


def make_core(cfg: Config) -> Core:
    return Core(allow_irreversible=bool(cfg.limits.allow_irreversible), allowed_verbs=tuple(cfg.limits.allowed_verbs),
                change_threshold=float(cfg.loop.change_threshold), think_below=float(cfg.loop.think_below),
                goal_max_age=int(cfg.loop.goal_max_age))


def make_adapter(name: str, integration, cfg: Config) -> Adapter:
    if name == "web":
        from .adapters.miniwob_dom import MiniWoBDomAdapter

        return MiniWoBDomAdapter(integration, collapse_over=int(cfg.loop.collapse_over))
    if name == "fallback":
        from .adapters.fallback import FallbackAdapter

        return FallbackAdapter(integration)
    if name == "code":
        from .adapters.code_workspace import CodeWorkspaceAdapter

        return CodeWorkspaceAdapter(integration)
    raise ValueError(f"Unknown adapter {name!r}; v0.2 adapters (a11y, vision, degraded) are not ported yet")


def make_body(cfg: Config, integration, adapter_name: str) -> Body:
    loop = cfg.loop
    return Body(integration, make_adapter(adapter_name, integration, cfg), make_core(cfg),
                body_id=f"{cfg.env.suite}:{adapter_name}", wait_cap_s=float(loop.wait_cap_s),
                wait_poll_s=float(loop.wait_poll_s), narrow_min_px=int(loop.narrow_min_px),
                narrow_max_depth=int(loop.narrow_max_depth))
