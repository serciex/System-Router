"""World model with two supervised heads, trained offline on collected episodes (spec v2, stage 3).

The latent is r2dreamer's (same encoder, RSSM and representation losses), so it still learns dynamics.
Two heads read it:
    cells     one logit per grid cell: is an important target in this cell?   (hindsight labels)
    escalate  one logit: will System 1 be wrong on this step?                  (System 1 vs System 2)

Built on TwoHeadDreamer so the same weights carry over to the later RL fine-tuning stage.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import autocast

from .agent import TwoHeadDreamer, networks
from .codec import ActionCodec


def _mlp(config, inp_dim: int, out_dim: int, name: str) -> nn.Module:
    spec = SimpleNamespace(act=config.act, layers=2, units=int(config.units), symlog_inputs=False,
                           device=config.device, name=name)
    return nn.Sequential(networks.MLP(spec, inp_dim), nn.Linear(int(config.units), out_dim))


class SupervisedDreamer(TwoHeadDreamer):
    def __init__(self, config, obs_space, act_space, codec: ActionCodec, cells_scale: float = 1.0,
                 escalate_scale: float = 1.0):
        super().__init__(config, obs_space, act_space, codec)
        feat = self.rssm.feat_size
        self.cell_head = _mlp(config.reward, feat, codec.n_cells, "cells")
        self.escalate_head = _mlp(config.reward, feat, 1, "escalate")
        self._loss_scales.update({"cells": float(cells_scale), "escalate": float(escalate_scale)})
        for name, module in (("cell_head", self.cell_head), ("escalate_head", self.escalate_head)):
            for param_name, param in module.named_parameters():
                self._named_params[f"{name}.{param_name}"] = param
        self._build_optimizer(config)

    # ------------------------------------------------------------------ training
    def to_batch(self, batch: dict[str, np.ndarray]) -> TensorDict:
        B, T = batch["is_first"].shape[:2]
        tensors = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        return TensorDict(tensors, batch_size=(B, T))

    def supervised_update(self, batch: dict[str, np.ndarray]) -> dict:
        data = self.to_batch(batch)
        initial = self.rssm.initial(data.shape[0])
        with autocast(device_type=self.device.type, dtype=torch.float16):
            metrics = self._supervised_grad(data, initial)
        self._scaler.unscale_(self._optimizer)
        self._agc(self._named_params.values())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        return {k: float(v) for k, v in metrics.items()}

    def _world_model_losses(self, data, initial):
        B, T = data.shape
        losses = {}
        embed = self.encoder(data)
        post_stoch, post_deter, post_logit = self.rssm.observe(embed, data["action"], initial, data["is_first"])
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"], losses["rep"] = torch.mean(dyn_loss), torch.mean(rep_loss)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamer":
            losses.update({key: torch.mean(-dist.log_prob(data[key]))
                           for key, dist in self.decoder(post_stoch, post_deter).items()})
        else:
            x1 = self.prj(feat.reshape(B * T, -1))
            x2 = embed.reshape(B * T, -1).detach()
            x1n = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2n = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
            c = torch.mm(x1n.T, x2n) / (B * T)
            off_diag = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            losses["barlow"] = (torch.diagonal(c) - 1.0).pow(2).sum() + self.barlow_lambd * c[off_diag].pow(2).sum()
        cont = 1.0 - data["is_terminal"].float()
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
        return losses, feat

    def _supervised_grad(self, data, initial) -> dict:
        losses, feat = self._world_model_losses(data, initial)
        cell_logits = self.cell_head(feat).float()
        cell_bce = F.binary_cross_entropy_with_logits(cell_logits, data["cell_target"].float(), reduction="none")
        cell_mask = data["cell_mask"].float()
        losses["cells"] = (cell_bce.mean(-1, keepdim=True) * cell_mask).sum() / cell_mask.sum().clamp_min(1.0)
        esc_logits = self.escalate_head(feat).float()
        esc_bce = F.binary_cross_entropy_with_logits(esc_logits, data["s1_wrong"].float(), reduction="none")
        esc_mask = data["s1_mask"].float()
        losses["escalate"] = (esc_bce * esc_mask).sum() / esc_mask.sum().clamp_min(1.0)
        total = sum(v * self._loss_scales[k] for k, v in losses.items())
        self._scaler.scale(total).backward()
        metrics = {f"loss/{k}": v.detach() for k, v in losses.items()}
        metrics["opt/loss"] = total.detach()
        return metrics

    # ------------------------------------------------------------------ prediction
    @torch.no_grad()
    def predict_sequence(self, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Cell probabilities (B, T, n_cells) and P(System 1 wrong) (B, T, 1) for a batch of windows."""
        data = self.to_batch(batch)
        embed = self.encoder(data)
        stoch, deter, _ = self.rssm.observe(embed, data["action"], self.rssm.initial(data.shape[0]), data["is_first"])
        feat = self.rssm.get_feat(stoch, deter)
        return (torch.sigmoid(self.cell_head(feat)).float().cpu().numpy(),
                torch.sigmoid(self.escalate_head(feat)).float().cpu().numpy())

    @torch.no_grad()
    def predict_step(self, inputs: dict, state: dict, is_first: bool) -> tuple[np.ndarray, float, dict]:
        """One online step: update the latent with this observation and read both heads."""
        obs = {k: torch.as_tensor(np.asarray(inputs[k], dtype=np.float32)[None], device=self.device)
               for k in ("vis", "txt", "vec")}
        embed = self.encoder(obs)
        reset = torch.tensor([is_first], device=self.device)
        stoch, deter, _ = self.rssm.obs_step(state["stoch"], state["deter"], state["prev_action"], embed, reset)
        feat = self.rssm.get_feat(stoch, deter)
        cells = torch.sigmoid(self.cell_head(feat))[0].float().cpu().numpy()
        p_wrong = float(torch.sigmoid(self.escalate_head(feat))[0, 0])
        return cells, p_wrong, {"stoch": stoch, "deter": deter, "prev_action": state["prev_action"]}

    def initial_state(self) -> dict:
        stoch, deter = self.rssm.initial(1)
        return {"stoch": stoch, "deter": deter,
                "prev_action": torch.zeros(1, self.act_dim, dtype=torch.float32, device=self.device)}
