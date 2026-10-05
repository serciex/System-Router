"""OBSOLETE under spec v3 (world model removed). Kept for reference; may not import. Ask the owner before deleting.

Hindsight labels from collected episodes, checked against the environment's own success signal.

- Important targets: in episodes the environment marked successful, the targets the agent actually
  reached or acted on. Every step labels the cells (at every active level) containing those targets.
  Unsuccessful episodes give no segmentation labels (masked out).
- System 1 wrong: by default, System 1's answer differs from System 2's on the same step ("s2"). With
  "path", System 1 is wrong if its choice was not one of the important targets (successful episodes only).

No extra LLM judgment is involved, so the labels cannot reward System 1 for agreeing with itself.
"""

from __future__ import annotations

import numpy as np

from body.grid import Grid
from body.schema import HOLD, NONE

from .codec import ActionCodec


def important_targets(records: list[dict], success: bool) -> set[str]:
    if not success:
        return set()
    important = set()
    for record in records:
        if record.get("nav_target") and record.get("nav_status") == "reached":
            important.add(record["nav_target"])
        if record.get("acted_target") and record.get("action_status") == "done":
            important.add(record["acted_target"])
    return important


def _s1_correct_on_path(record: dict, important: set[str]) -> bool:
    nav = record["s1"]["navigation"]
    choice = nav.get("navigate", nav.get("destination"))
    action = record["s1"]["action"]
    acted = action.partition("@")[2] if action and action != NONE else None
    if choice not in (None, HOLD, "keep"):
        return choice in important
    return acted in important


def label_episode(episode: dict, grid: Grid, codec: ActionCodec, s1_reference: str = "s2") -> dict[str, np.ndarray]:
    records = episode["records"]
    success = bool(episode.get("success"))
    important = important_targets(records, success)
    steps = len(records)
    cell_target = np.zeros((steps, codec.n_cells), dtype=np.float32)
    cell_mask = np.full((steps, 1), 1.0 if success and important else 0.0, dtype=np.float32)
    s1_wrong = np.zeros((steps, 1), dtype=np.float32)
    s1_mask = np.zeros((steps, 1), dtype=np.float32)

    offsets = np.cumsum([0] + codec.cell_counts[:-1])
    for t, record in enumerate(records):
        anchor = tuple(record.get("anchor", (0.5, 0.5)))
        for target_id in important:
            center = record["positions"].get(target_id)
            if center is None:
                continue
            for level_index, level in enumerate(codec.levels):
                cell = grid.cell_of(level, center[0], center[1], anchor)
                if cell is not None:
                    cell_target[t, offsets[level_index] + cell] = 1.0
        if s1_reference == "s2":
            s1, s2 = record["s1"], record["s2"]
            s1_wrong[t] = float(s1["navigation"] != s2["navigation"] or s1["action"] != s2["action"])
            s1_mask[t] = 1.0
        elif s1_reference == "path" and success and important:
            s1_wrong[t] = float(not _s1_correct_on_path(record, important))
            s1_mask[t] = 1.0
    return {"cell_target": cell_target, "cell_mask": cell_mask, "s1_wrong": s1_wrong, "s1_mask": s1_mask}


def attach_labels(episodes: list[dict], grid: Grid, codec: ActionCodec, s1_reference: str = "s2") -> list[dict]:
    """Add the label arrays to each episode's arrays (in place) and return the episodes."""
    for episode in episodes:
        episode["arrays"].update(label_episode(episode, grid, codec, s1_reference))
    return episodes


def summary(episodes: list[dict]) -> dict:
    steps = sum(len(e["records"]) for e in episodes)
    labeled = sum(float(e["arrays"]["cell_mask"].sum()) for e in episodes if "cell_mask" in e["arrays"])
    wrong = sum(float(e["arrays"]["s1_wrong"].sum()) for e in episodes if "s1_wrong" in e["arrays"])
    checked = sum(float(e["arrays"]["s1_mask"].sum()) for e in episodes if "s1_mask" in e["arrays"])
    return {
        "episodes": len(episodes),
        "success_rate": float(np.mean([bool(e.get("success")) for e in episodes])) if episodes else 0.0,
        "steps": steps,
        "steps_with_cell_labels": int(labeled),
        "s1_wrong_rate": wrong / checked if checked else None,
    }
