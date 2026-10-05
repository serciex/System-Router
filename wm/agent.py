"""OBSOLETE under spec v3 (world model removed). Kept for reference; may not import. Ask the owner before deleting.

r2dreamer's Dreamer with two reward heads and two critics (spec v2, "Training wiring").

- `reward` / `value`          learn the decision reward `rew_dec`; their advantage trains the route dimension.
- `reward_seg` / `value_seg`  learn the segmentation reward `rew_seg`; their advantage trains level and cells.
- Cells of levels other than the chosen one are masked out of the actor's log-probability and entropy.

Everything else (RSSM, encoder, representation loss, imagination, replay value learning) is r2dreamer's,
unchanged. The upstream submodule is not modified; `_cal_grad` is re-implemented here following its
structure. Supported representation losses: "r2dreamer" (default) and "dreamer".
"""

from __future__ import annotations

import copy

import torch
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR

from .r2d import add_r2dreamer_to_path

add_r2dreamer_to_path()

import networks  # noqa: E402  (r2dreamer)
import tools  # noqa: E402
from dreamer import Dreamer  # noqa: E402
from optim import LaProp, clip_grad_agc_  # noqa: E402
from tools import to_f32  # noqa: E402

from .codec import ActionCodec  # noqa: E402


def _frozen_view(module: torch.nn.Module) -> torch.nn.Module:
    """A copy whose parameters share storage with `module` but receive no gradient (as r2dreamer does)."""
    frozen = copy.deepcopy(module)
    for (_, original), (_, shadow) in zip(module.named_parameters(), frozen.named_parameters()):
        shadow.data = original.data
        shadow.requires_grad_(False)
    return frozen


class TwoHeadDreamer(Dreamer):
    def __init__(self, config, obs_space, act_space, codec: ActionCodec):
        if config.rep_loss not in ("r2dreamer", "dreamer"):
            raise ValueError("TwoHeadDreamer supports rep_loss 'r2dreamer' or 'dreamer'")
        compile_requested = bool(config.compile)
        config.compile = False  # the optimizer is rebuilt below, compile afterwards
        super().__init__(config, obs_space, act_space)
        self.codec = codec

        feat = self.rssm.feat_size
        self.reward_seg = networks.MLPHead(config.reward, feat)
        self.value_seg = networks.MLPHead(config.critic, feat)
        self._slow_value_seg = copy.deepcopy(self.value_seg)
        for param in self._slow_value_seg.parameters():
            param.requires_grad = False
        self.return_ema_seg = networks.ReturnEMA(device=self.device)

        self._loss_scales.update({
            "rew_seg": self._loss_scales["rew"],
            "value_seg": self._loss_scales["value"],
            "repval_seg": self._loss_scales["repval"],
        })
        for name, module in (("reward_seg", self.reward_seg), ("value_seg", self.value_seg)):
            for param_name, param in module.named_parameters():
                self._named_params[f"{name}.{param_name}"] = param
        self._build_optimizer(config)
        self.clone_and_freeze()
        if compile_requested:
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    def _build_optimizer(self, config) -> None:
        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(self._named_params.values(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps)
        self._scaler = GradScaler()
        warmup = config.warmup
        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0)

    # ------------------------------------------------------------------ target networks
    def clone_and_freeze(self):
        super().clone_and_freeze()
        if "reward_seg" in self._modules:  # absent while the base constructor runs
            self._frozen_reward_seg = _frozen_view(self.reward_seg)
            self._frozen_value_seg = _frozen_view(self.value_seg)
            self._frozen_slow_value_seg = _frozen_view(self._slow_value_seg)

    def train(self, mode: bool = True):
        super().train(mode)
        if "_slow_value_seg" in self._modules:
            self._slow_value_seg.train(False)
        return self

    def _update_slow_target(self):
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value_seg.parameters(), self._slow_value_seg.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        super()._update_slow_target()

    # ------------------------------------------------------------------ policy terms
    def _policy_terms(self, policy, action):
        """Per-dimension log-probabilities and entropy, with cells of other levels masked out."""
        splits = torch.split(action, list(self.codec.shape), dim=-1)
        log_probs = [dist.log_prob(part) for dist, part in zip(policy.onehots, splits)]
        entropies = [dist.entropy() for dist in policy.onehots]
        mask = self.codec.cell_mask(splits[1]).to(log_probs[0].dtype)  # (N, T, n_cells)
        cell_lp = torch.stack(log_probs[2:], dim=-1)
        cell_ent = torch.stack(entropies[2:], dim=-1)
        lp_route = log_probs[0]
        lp_seg = log_probs[1] + (mask * cell_lp).sum(-1)
        entropy = entropies[0] + entropies[1] + (mask * cell_ent).sum(-1)
        return lp_route, lp_seg, entropy

    # ------------------------------------------------------------------ gradients
    def _cal_grad(self, data, initial):
        losses, metrics = {}, {}
        B, T = data.shape

        # World model (unchanged from r2dreamer).
        embed = self.encoder(data)
        post_stoch, post_deter, post_logit = self.rssm.observe(embed, data["action"], initial, data["is_first"])
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)
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

        # Two reward heads and the shared continue head.
        losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["rew_dec"])))
        losses["rew_seg"] = torch.mean(-self.reward_seg(feat).log_prob(to_f32(data["rew_seg"])))
        cont = 1.0 - to_f32(data["is_terminal"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))
        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        # Imagination.
        start = (post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
                 post_deter.reshape(-1, *post_deter.shape[2:]).detach())
        imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()
        imag_cont = self._frozen_cont(imag_feat).mean
        disc = 1 - 1 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont

        heads = {
            "dec": (self._frozen_reward, self._frozen_value, self._frozen_slow_value, self.value, self.return_ema),
            "seg": (self._frozen_reward_seg, self._frozen_value_seg, self._frozen_slow_value_seg, self.value_seg,
                    self.return_ema_seg),
        }
        advantages, returns = {}, {}
        for name, (reward_head, value_head, slow_head, live_value, ema) in heads.items():
            imag_reward = reward_head(imag_feat).mode()
            imag_value = value_head(imag_feat).mode()
            imag_slow = slow_head(imag_feat).mode()
            ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
            _, scale = ema(ret)
            advantages[name] = (ret - imag_value[:, :-1]) / scale
            returns[name] = ret
            value_dist = live_value(imag_feat)
            padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
            key = "value" if name == "dec" else "value_seg"
            losses[key] = torch.mean(weight[:, :-1].detach() * (
                -value_dist.log_prob(padded.detach()) - value_dist.log_prob(imag_slow.detach()))[:, :-1].unsqueeze(-1))
            metrics[f"rew_{name}"] = torch.mean(imag_reward)
            metrics[f"adv_{name}"] = torch.mean(advantages[name])

        policy = self.actor(imag_feat)
        lp_route, lp_seg, entropy = self._policy_terms(policy, imag_action)
        lp_route, lp_seg = lp_route[:, :-1].unsqueeze(-1), lp_seg[:, :-1].unsqueeze(-1)
        entropy = entropy[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(
            lp_route * advantages["dec"].detach() + lp_seg * advantages["seg"].detach() + self.act_entropy * entropy))
        metrics["action_entropy"] = torch.mean(entropy)
        metrics["con"] = torch.mean(imag_cont)
        metrics.update(tools.tensorstats(imag_action[..., :2], "route"))

        # Replay-based value learning for both critics (gradients flow into the world model).
        is_last, is_terminal = to_f32(data["is_last"]), to_f32(data["is_terminal"])
        feat = self.rssm.get_feat(post_stoch, post_deter)
        replay = {
            "dec": (to_f32(data["rew_dec"]), self._frozen_value, self._frozen_slow_value, self.value, "repval"),
            "seg": (to_f32(data["rew_seg"]), self._frozen_value_seg, self._frozen_slow_value_seg, self.value_seg, "repval_seg"),
        }
        for name, (reward, value_head, slow_head, live_value, key) in replay.items():
            boot = returns[name][:, 0].reshape(B, T, 1)
            value = value_head(feat).mode()
            slow = slow_head(feat).mode()
            ret = self._lambda_return(is_last, is_terminal, reward, value, boot, disc, self.lamb)
            padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
            value_dist = live_value(feat)
            losses[key] = torch.mean((1.0 - is_last)[:, :-1] * (
                -value_dist.log_prob(padded.detach()) - value_dist.log_prob(slow.detach()))[:, :-1].unsqueeze(-1))

        total = sum(v * self._loss_scales[k] for k, v in losses.items())
        self._scaler.scale(total).backward()
        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics["opt/loss"] = total
        return (post_stoch, post_deter), metrics
