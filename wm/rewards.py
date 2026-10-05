"""OBSOLETE under spec v3 (world model removed). Kept for reference; may not import. Ask the owner before deleting.

The two reward terms (spec v2, "Rewards").

Segmentation (trains level and cells):
    r_seg = beta * recall_important - c * n_unimportant - eps * n_options
An important target counts as covered only if selected at its needed level or finer. Counts are of
options the body returned after resolving cells.

Decision (trains route):
    s = min(1, L_expected / L)
    r_dec = a + b*s - kappa*[S2]                  if correct
    r_dec = -(p_irr + q*(1 - s)) - kappa*[S2]     if wrong
"""

from __future__ import annotations

from typing import Optional

from body.grid import DIRECTIONS, Grid
from body.schema import Observation, Target


def seg_reward(important: dict[str, dict], observation: Observation, all_targets: list[Target], grid: Grid,
               anchor: tuple[float, float], beta: float, c: float, eps: float) -> tuple[float, dict]:
    returned = observation.targets
    level = observation.level
    returned_elements = {t.id for t in returned if t.kind == "element"}
    directions = [t for t in returned if t.kind == "direction"]
    positions = {t.id: t for t in all_targets}

    def direction_covers(direction: Target, target_id: str) -> bool:
        target = positions.get(target_id)
        if target is None:
            return False
        index = DIRECTIONS.index(direction.id.split(":", 1)[1])
        return grid.cell_of(1, *target.center, anchor) == index

    covered = 0
    useful_directions: set[str] = set()
    for target_id, info in important.items():
        needed = int(info.get("level", 3))
        if target_id in returned_elements and level >= needed:
            covered += 1
            continue
        if needed == 1 and level == 1:
            hits = [d for d in directions if direction_covers(d, target_id)]
            if hits:
                covered += 1
                useful_directions.update(d.id for d in hits)

    recall = covered / len(important) if important else 1.0
    n_unimportant = (
        sum(1 for t in returned if t.kind == "element" and t.id not in important)
        + sum(1 for d in directions if d.id not in useful_directions)
        + sum(1 for t in returned if t.kind == "positional")
    )
    n_options = len(returned)
    reward = beta * recall - c * n_unimportant - eps * n_options
    return reward, {"recall": recall, "n_unimportant": n_unimportant, "n_options": n_options}


def dec_reward(correct: Optional[bool], latency_ms: float, used_s2: bool, irreversible: bool,
               expected_latency_ms: float, a: float, b: float, p: float, p_irr: float, q: float, kappa: float) -> float:
    if correct is None:
        return 0.0
    s = min(1.0, expected_latency_ms / max(latency_ms, 1e-6))
    cost = kappa if used_s2 else 0.0
    if correct:
        return a + b * s - cost
    return -((p_irr if irreversible else p) + q * (1.0 - s)) - cost
