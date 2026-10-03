"""The world model's flat multi-discrete action and its masks.

    [ route: S1 | S2 ]  [ level: one per active level ]  [ cells of every active level: 0/1 each ]

Every dimension is a categorical (r2dreamer's `multi_onehot` actor). Cells of levels other than the
chosen one are masked: ignored by the body and removed from the actor's log-probability and entropy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from body.grid import Grid

S1, S2 = 0, 1


@dataclass
class Decision:
    route: int  # S1 or S2
    level: int
    cells: tuple[int, ...]


class MultiOneHotSpace:
    """Minimal action space r2dreamer understands (`shape` per dimension, `multi_discrete` flag).

    A gym Box with this shape would allocate an array of size prod(shape), which is far too large.
    """

    multi_discrete = True
    dtype = np.float32

    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(int(s) for s in shape)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        parts = []
        for size in self.shape:
            onehot = np.zeros(size, dtype=np.float32)
            onehot[rng.integers(size)] = 1.0
            parts.append(onehot)
        return np.concatenate(parts)


class ActionCodec:
    def __init__(self, grid: Grid):
        self.levels = list(grid.levels)
        self.cell_counts = [grid.n_cells(level) for level in self.levels]
        self.shape = (2, len(self.levels)) + (2,) * sum(self.cell_counts)
        # Level index each cell dimension belongs to.
        self.cell_level = np.concatenate([np.full(n, i) for i, n in enumerate(self.cell_counts)]).astype(np.int64)
        self.space = MultiOneHotSpace(self.shape)

    @property
    def dim(self) -> int:
        return int(sum(self.shape))

    @property
    def n_cells(self) -> int:
        return int(sum(self.cell_counts))

    def decode(self, action: np.ndarray) -> Decision:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        route = int(np.argmax(action[0:2]))
        n_levels = len(self.levels)
        level_index = int(np.argmax(action[2:2 + n_levels]))
        bits = action[2 + n_levels:].reshape(-1, 2).argmax(-1)
        offset = sum(self.cell_counts[:level_index])
        chosen = bits[offset: offset + self.cell_counts[level_index]]
        return Decision(route=route, level=self.levels[level_index], cells=tuple(int(i) for i in np.flatnonzero(chosen)))

    def encode(self, decision: Decision) -> np.ndarray:
        parts = [np.eye(2, dtype=np.float32)[decision.route]]
        level_index = self.levels.index(decision.level)
        parts.append(np.eye(len(self.levels), dtype=np.float32)[level_index])
        bits = np.zeros(self.n_cells, dtype=np.int64)
        offset = sum(self.cell_counts[:level_index])
        for cell in decision.cells:
            bits[offset + cell] = 1
        parts.append(np.eye(2, dtype=np.float32)[bits].reshape(-1))
        return np.concatenate(parts)

    def cell_mask(self, level_onehot):
        """Torch mask (..., n_cells): 1 for cells of the chosen level, 0 otherwise."""
        import torch

        index = torch.as_tensor(self.cell_level, device=level_onehot.device)
        return level_onehot[..., index]
