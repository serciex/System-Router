"""Put the r2dreamer submodule on the import path (it is a flat folder of modules, not a package).

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
