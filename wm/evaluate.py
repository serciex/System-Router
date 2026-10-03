"""Held-out evaluation of the supervised heads, and the threshold/budget sweep used to tune HeadsPolicy."""

from __future__ import annotations

import numpy as np


def episode_batch(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One whole episode as a batch of one window (actions shifted, first step resets)."""
    steps = arrays["vis"].shape[0]
    batch = {k: v[None] for k, v in arrays.items() if k != "action"}
    action = np.zeros_like(arrays["action"])
    action[1:] = arrays["action"][:-1]
    batch["action"] = action[None]
    first = np.zeros((1, steps, 1), dtype=bool)
    first[0, 0, 0] = True
    batch["is_first"] = first
    batch["is_terminal"] = np.zeros((1, steps, 1), dtype=bool)
    return batch


def choose_cells(scores: np.ndarray, threshold: float, min_cells: int, max_cells: int) -> np.ndarray:
    order = np.argsort(-scores)
    chosen = [i for i in order if scores[i] >= threshold][:max_cells]
    if len(chosen) < min_cells:
        chosen = list(order[:min_cells])
    mask = np.zeros_like(scores, dtype=bool)
    mask[chosen] = True
    return mask


def sweep_cells(probs: np.ndarray, targets: np.ndarray, labeled: np.ndarray, thresholds, min_cells: int,
                max_cells: int) -> list[dict]:
    """probs, targets: (N, n_cells) for one level; labeled: (N,) bool. Recall of important cells and cells used."""
    rows = []
    probs, targets = probs[labeled], targets[labeled]
    for threshold in thresholds:
        hits = total = chosen_count = 0
        for p, y in zip(probs, targets):
            chosen = choose_cells(p, threshold, min_cells, max_cells)
            hits += int((chosen & (y > 0.5)).sum())
            total += int((y > 0.5).sum())
            chosen_count += int(chosen.sum())
        rows.append({"cell_threshold": float(threshold), "recall": hits / total if total else None,
                     "mean_cells": chosen_count / max(len(probs), 1)})
    return rows


def sweep_escalation(p_wrong: np.ndarray, wrong: np.ndarray, labeled: np.ndarray, thresholds) -> list[dict]:
    """How many System 1 errors are caught, and how often System 2 is called, for each threshold."""
    rows = []
    p, y = p_wrong[labeled], wrong[labeled] > 0.5
    for threshold in thresholds:
        escalate = p > threshold
        rows.append({
            "escalate_threshold": float(threshold),
            "caught": float((escalate & y).sum() / y.sum()) if y.sum() else None,
            "escalation_rate": float(escalate.mean()) if len(p) else None,
            "accuracy": float((escalate == y).mean()) if len(p) else None,
        })
    return rows


def suggest(cell_rows: list[dict], escalation_rows: list[dict], target_recall: float = 0.95,
            target_caught: float = 0.9) -> dict:
    """Fewest cells reaching the recall target; least escalation catching the target share of errors."""
    good_cells = [r for r in cell_rows if r["recall"] is not None and r["recall"] >= target_recall]
    cell = min(good_cells, key=lambda r: r["mean_cells"]) if good_cells else max(
        cell_rows, key=lambda r: r["recall"] or 0.0)
    good_esc = [r for r in escalation_rows if r["caught"] is not None and r["caught"] >= target_caught]
    esc = min(good_esc, key=lambda r: r["escalation_rate"]) if good_esc else max(
        escalation_rows, key=lambda r: r["caught"] or 0.0)
    return {"cell_threshold": cell["cell_threshold"], "expected_recall": cell["recall"],
            "expected_cells": cell["mean_cells"], "escalate_threshold": esc["escalate_threshold"],
            "expected_caught": esc["caught"], "expected_escalation_rate": esc["escalation_rate"]}
