"""OBSOLETE under spec v3 (world model removed). Kept for reference; may not import. Ask the owner before deleting.

Collected episodes for offline training.

Each episode is two files: `episode_NNNNNN.npz` with per-step world model inputs (vis, txt, vec) and the
decision taken (action), and `episode_NNNNNN.json` with the step records (both systems' answers, target
positions, outcomes) used for hindsight labels.

The sampler concatenates episodes into one stream and cuts random windows, like r2dreamer's replay:
`is_first` resets the RSSM at episode boundaries, and `action` is shifted so step t holds the action that
led into it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

INPUT_KEYS = ("vis", "txt", "vec")


class EpisodeWriter:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.out_dir.glob("episode_*.npz"))
        self.index = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
        self._reset()

    def _reset(self) -> None:
        self.steps: dict[str, list] = {key: [] for key in (*INPUT_KEYS, "action")}
        self.records: list[dict] = []
        self.meta: dict = {}

    def start(self, **meta) -> None:
        self._reset()
        self.meta = meta

    def add(self, inputs: dict, action: np.ndarray, record: dict) -> None:
        for key in INPUT_KEYS:
            self.steps[key].append(np.asarray(inputs[key], dtype=np.float32))
        self.steps["action"].append(np.asarray(action, dtype=np.float32))
        self.records.append(record)

    def finish(self, success: Optional[bool]) -> Optional[Path]:
        if not self.records:
            return None
        stem = self.out_dir / f"episode_{self.index:06d}"
        np.savez_compressed(f"{stem}.npz", **{key: np.stack(values) for key, values in self.steps.items()})
        Path(f"{stem}.json").write_text(json.dumps({"meta": self.meta, "success": bool(success), "records": self.records}))
        self.index += 1
        self._reset()
        return Path(f"{stem}.npz")


def load_episodes(data_dir: str | Path) -> list[dict]:
    episodes = []
    for npz_path in sorted(Path(data_dir).glob("episode_*.npz")):
        info = json.loads(npz_path.with_suffix(".json").read_text())
        with np.load(npz_path) as arrays:
            episodes.append({"arrays": {k: arrays[k] for k in arrays.files}, **info})
    return episodes


class SequenceSampler:
    """Random windows of `length` steps over the concatenated episodes."""

    def __init__(self, episodes: list[dict], batch_size: int, length: int, seed: int = 0):
        if not episodes:
            raise ValueError("No episodes to sample from")
        self.batch_size, self.length = int(batch_size), int(length)
        self.rng = np.random.default_rng(seed)
        keys = episodes[0]["arrays"].keys()
        self.stream = {k: np.concatenate([e["arrays"][k] for e in episodes]) for k in keys}
        first = [np.zeros(len(e["records"]), dtype=bool) for e in episodes]
        for flags in first:
            flags[0] = True
        self.stream["is_first"] = np.concatenate(first)
        self.stream["is_terminal"] = np.zeros_like(self.stream["is_first"])
        self.size = len(self.stream["is_first"])
        if self.size < self.length + 1:
            raise ValueError(f"Only {self.size} steps collected, need at least {self.length + 1}")

    def sample(self) -> dict[str, np.ndarray]:
        starts = self.rng.integers(0, self.size - self.length, size=self.batch_size)
        index = starts[:, None] + np.arange(self.length + 1)[None]
        batch = {}
        for key, values in self.stream.items():
            window = values[index]
            if key == "action":
                batch[key] = window[:, :-1]  # the action that led into each step
            else:
                batch[key] = window[:, 1:]
            if batch[key].ndim == 2:
                batch[key] = batch[key][..., None]
        batch["is_first"][:, 0] = True  # every window starts a fresh latent
        return batch
