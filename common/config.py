"""Load the YAML config with dotted command-line overrides.

Example:
    cfg = load_config("configs/default.yaml", ["llm.quantization=4bit", "env.tasks=[click-test-2]"])
    cfg.llm.quantization  # "4bit"
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    """A dict with attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as error:
            raise AttributeError(name) from error
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def _to_config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({key: _to_config(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_config(val) for val in value]
    return value


def apply_override(cfg: dict, override: str) -> None:
    """Apply one `a.b.c=value` override; the value is parsed as YAML."""
    if "=" not in override:
        raise ValueError(f"Override must look like key=value, got {override!r}")
    key, raw = override.split("=", 1)
    node = cfg
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path = REPO_ROOT / "configs" / "default.yaml", overrides: Iterable[str] = ()) -> Config:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    for override in overrides:
        apply_override(raw, override)
    return _to_config(raw)


def resolve(path: str | Path) -> Path:
    """Resolve a config path against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def model_source(cfg: Config) -> str:
    """Local model directory if it holds weights, otherwise treat the value as a Hugging Face id."""
    local = resolve(cfg.paths.model)
    if local.exists():
        return str(local)
    return str(cfg.paths.model)
