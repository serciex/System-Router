"""Baseline for the stage 3 gate: the same two heads on the current step's inputs only (no world model).

If the world model's heads do not beat this, the dynamics are not adding anything on these tasks.
It exposes the same `supervised_update` and `predict_sequence` interface as SupervisedDreamer, so
scripts/train_offline.py can train and evaluate either one.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class StepClassifier(nn.Module):
    def __init__(self, input_dim: int, n_cells: int, hidden: int = 512, lr: float = 3e-4, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.body = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.cell_head = nn.Linear(hidden, n_cells)
        self.escalate_head = nn.Linear(hidden, 1)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _inputs(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        x = np.concatenate([batch["vis"], batch["txt"], batch["vec"]], axis=-1)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _forward(self, batch):
        h = self.body(self._inputs(batch))
        return self.cell_head(h), self.escalate_head(h)

    def supervised_update(self, batch: dict[str, np.ndarray]) -> dict:
        cell_logits, esc_logits = self._forward(batch)
        t = lambda key: torch.as_tensor(batch[key], dtype=torch.float32, device=self.device)  # noqa: E731
        cell_mask, esc_mask = t("cell_mask"), t("s1_mask")
        cells = (F.binary_cross_entropy_with_logits(cell_logits, t("cell_target"), reduction="none").mean(-1, keepdim=True)
                 * cell_mask).sum() / cell_mask.sum().clamp_min(1.0)
        escalate = (F.binary_cross_entropy_with_logits(esc_logits, t("s1_wrong"), reduction="none") * esc_mask).sum() \
            / esc_mask.sum().clamp_min(1.0)
        loss = cells + escalate
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss/cells": float(cells), "loss/escalate": float(escalate), "opt/loss": float(loss)}

    @torch.no_grad()
    def predict_sequence(self, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        cell_logits, esc_logits = self._forward(batch)
        return torch.sigmoid(cell_logits).cpu().numpy(), torch.sigmoid(esc_logits).cpu().numpy()
