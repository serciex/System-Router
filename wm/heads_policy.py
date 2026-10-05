"""OBSOLETE under spec v3 (world model removed). Kept for reference; may not import. Ask the owner before deleting.

Run-time policy from the supervised heads: a threshold for routing and a budget for cells.

    route: System 2 if P(System 1 wrong) > escalate_threshold, else System 1
    cells: every cell (of the chosen level) with probability >= cell_threshold, kept between min_cells
           and max_cells, highest first

The two thresholds and the budget are tuned on held-out episodes (scripts/train_offline.py prints a sweep).
"""

from __future__ import annotations

import numpy as np
import torch

from .codec import S1, S2, ActionCodec, Decision


class HeadsPolicy:
    def __init__(self, model, codec: ActionCodec, level: int = 3, escalate_threshold: float = 0.5,
                 cell_threshold: float = 0.3, min_cells: int = 1, max_cells: int = 12):
        if level not in codec.levels:
            raise ValueError(f"Level {level} is not active")
        self.model = model.eval()
        self.codec = codec
        self.level = level
        self.escalate_threshold = float(escalate_threshold)
        self.cell_threshold = float(cell_threshold)
        self.min_cells, self.max_cells = int(min_cells), int(max_cells)
        index = codec.levels.index(level)
        self.offset = int(sum(codec.cell_counts[:index]))
        self.count = int(codec.cell_counts[index])
        self.reset()

    def reset(self) -> None:
        self.state = self.model.initial_state()
        self.first = True
        self.last_p_wrong = 0.0

    def decide(self, inputs: dict) -> Decision:
        cells, p_wrong, self.state = self.model.predict_step(inputs, self.state, self.first)
        self.first = False
        self.last_p_wrong = p_wrong
        scores = cells[self.offset: self.offset + self.count]
        order = np.argsort(-scores)
        chosen = [int(i) for i in order if scores[i] >= self.cell_threshold][: self.max_cells]
        if len(chosen) < self.min_cells:
            chosen = [int(i) for i in order[: self.min_cells]]
        decision = Decision(route=S2 if p_wrong > self.escalate_threshold else S1, level=self.level,
                            cells=tuple(sorted(chosen)))
        action = torch.as_tensor(self.codec.encode(decision)[None], device=self.model.device)
        self.state["prev_action"] = action
        return decision
