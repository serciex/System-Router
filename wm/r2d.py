"""OBSOLETE under spec v3 (world model removed). Kept for reference; may not import. Ask the owner before deleting.

Put the r2dreamer submodule on the import path (it is a flat folder of modules, not a package).

Its modules (`dreamer`, `networks`, `tools`, `buffer`, `trainer`, `envs`, ...) are imported by name, which is
why this repository's own packages avoid those names.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from common.config import REPO_ROOT


def add_r2dreamer_to_path(path: str | Path | None = None) -> Path:
    root = Path(path or os.environ.get("R2DREAMER_PATH") or REPO_ROOT / "dreamer v3")
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()
    if not (root / "dreamer.py").exists():
        raise FileNotFoundError(f"r2dreamer not found at {root}; run `git submodule update --init`")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def compose_r2dreamer(cfg, logdir_name: str = "wm"):
    """r2dreamer's hydra config with this project's environment keys, sizes and log directory."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from common.config import resolve

    root = add_r2dreamer_to_path(cfg.paths.r2dreamer)
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        conf = compose(config_name="configs", overrides=["env=crafter", f"model={cfg.dreamer.model}"])
    OmegaConf.set_struct(conf, False)
    d = cfg.dreamer
    conf.device = d.device
    conf.batch_size, conf.batch_length = int(d.batch_size), int(d.batch_length)
    conf.logdir = str(resolve(cfg.paths.runs) / logdir_name)
    conf.env.task = "router"
    conf.env.steps = int(d.steps)
    conf.env.env_num = 1          # one worker holds the LLM; a second would load another copy
    conf.env.eval_episode_num = 0  # evaluation runs separately (held-out adapters)
    conf.env.train_ratio = int(d.train_ratio)
    conf.env.action_repeat = 1
    for part in ("encoder", "decoder"):
        conf.env[part].mlp_keys = "^(vis|txt|vec)$"
        conf.env[part].cnn_keys = "$^"
    conf.model.rep_loss = d.rep_loss
    conf.model.imag_horizon = int(d.imag_horizon)
    conf.model.compile = False
    OmegaConf.resolve(conf)
    return conf
