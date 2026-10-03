"""Build a Body from config: adapter names like "dom", "a11y", "vision", "degraded:dom"."""

from __future__ import annotations

import random

from common.config import Config, resolve

from .adapters.base import Adapter
from .contract import Body
from .core import Core
from .grid import Grid


def make_grid(cfg: Config) -> Grid:
    return Grid(sizes=dict(cfg.grid.sizes), levels=list(cfg.grid.levels), level1_span=float(cfg.grid.level1_span))


def make_core(cfg: Config) -> Core:
    return Core(allow_irreversible=bool(cfg.limits.allow_irreversible), allowed_verbs=tuple(cfg.limits.allowed_verbs))


def make_adapter(name: str, environment, cfg: Config, seed: int = 0) -> Adapter:
    if name.startswith("degraded:"):
        from .adapters.degraded import DegradedAdapter

        degraded = cfg.env.degraded
        return DegradedAdapter(make_adapter(name.split(":", 1)[1], environment, cfg, seed),
                               drop=degraded.drop, jitter=degraded.jitter, blank_label=degraded.blank_label, seed=seed)
    if cfg.env.suite == "miniwob":
        if name == "dom":
            from .adapters.miniwob_dom import MiniWoBDomAdapter

            return MiniWoBDomAdapter(environment)
        if name == "a11y":
            from .adapters.miniwob_a11y import MiniWoBA11yAdapter

            return MiniWoBA11yAdapter(environment)
        if name == "vision":
            from .adapters.omniparser import load_omniparser
            from .adapters.vision import VisionAdapter

            return VisionAdapter(environment, load_omniparser(resolve(cfg.env.vision_weights)))
    raise ValueError(f"Unknown adapter {name!r} for suite {cfg.env.suite!r}")


def pick_adapters(cfg: Config, rng: random.Random, evaluate: bool = False) -> list[str]:
    """One adapter set per episode: from the pool in training, the held-out sets in evaluation."""
    if evaluate and cfg.env.heldout_adapters:
        return list(rng.choice(list(cfg.env.heldout_adapters)))
    pool = [list(s) for s in cfg.env.adapter_pool] if cfg.env.adapter_pool else [list(cfg.env.adapters)]
    heldout = [list(s) for s in cfg.env.heldout_adapters]
    pool = [s for s in pool if s not in heldout] or pool
    return rng.choice(pool)


def make_body(cfg: Config, environment, adapter_names: list[str], seed: int = 0) -> Body:
    adapters = [make_adapter(name, environment, cfg, seed) for name in adapter_names]
    return Body(adapters, make_grid(cfg), make_core(cfg))
