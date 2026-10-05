"""OBSOLETE under spec v3 (world model removed). Kept for reference; may not import. Ask the owner before deleting.

The whole system as an environment for r2dreamer (its old-gym API: reset() -> obs, step() -> obs, r, done, info).

The world model only sees and chooses what the spec says it does:
    observation: vis (VLM patch cells), txt (VLM hidden state), vec (cheap numbers)
    action:      route, level, cells (flat multi-discrete, see codec.py)
Everything else (body, System 1/2, labeler, rewards) runs inside this environment.

The two reward terms travel as observation keys `rew_seg` and `rew_dec` so the two-head agent can learn
them separately; `reward` (their weighted sum) is for logging only. The encoder only reads keys matching
'^(vis|txt|vec)$'. The LLM is loaded lazily on the first reset, inside r2dreamer's worker process.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from body.factory import make_grid
from common.config import Config, _to_config, resolve

from .codec import ActionCodec

VEC_SIZE = 10  # must match brain.step.VEC_SIZE


class RouterEnv:
    metadata: dict = {}

    def __init__(self, cfg: dict | Config, seed: int = 0, evaluate: bool = False, log_name: str = "steps.jsonl"):
        self.cfg = cfg if isinstance(cfg, Config) else _to_config(cfg)
        self.seed = int(seed)
        self.evaluate = evaluate
        self.log_name = log_name
        self.codec = ActionCodec(make_grid(self.cfg))
        self.brain = None
        self._episode = 0
        self._last: Optional[dict] = None

    # ------------------------------------------------------------------ spaces
    @property
    def observation_space(self) -> gym.spaces.Dict:
        f = self.cfg.features
        vis = int(f.vision_grid) ** 2 * int(f.cell_dim)
        box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), dtype=np.float32)  # noqa: E731
        return gym.spaces.Dict({
            "vis": box(vis), "txt": box(int(f.text_dim)), "vec": box(VEC_SIZE),
            "rew_seg": box(1), "rew_dec": box(1), "log_success": box(1),
        })

    @property
    def action_space(self):
        return self.codec.space

    # ------------------------------------------------------------------ episodes
    def _build(self) -> None:
        from brain.step import Brain
        from environments import make_environment

        environment = make_environment(self.cfg)
        log_path = resolve(self.cfg.paths.runs) / self.log_name
        self.brain = Brain(self.cfg, environment, log_path=log_path, seed=self.seed)

    def reset(self) -> dict:
        if self.brain is None:
            self._build()
        self._episode += 1
        inputs = self.brain.reset(seed=self.seed * 100000 + self._episode, evaluate=self.evaluate)
        self._last = inputs
        return self._obs(inputs, first=True, last=False, terminal=False, r_seg=0.0, r_dec=0.0, success=False)

    def step(self, action: np.ndarray):
        decision = self.codec.decode(action)
        report = self.brain.step(decision)
        inputs = self._last if report.done else self.brain.wm_inputs()
        self._last = inputs
        terminal = bool(report.outcome.env_done or self.brain.goal.done)
        weights = self.cfg.rewards.log_weights
        reward = float(weights.seg * report.r_seg + weights.dec * report.r_dec)
        obs = self._obs(inputs, first=False, last=report.done, terminal=terminal,
                        r_seg=report.r_seg, r_dec=report.r_dec, success=bool(report.success))
        info = {"route": report.route_used, "escalated": report.escalated, "correct": report.correct}
        return obs, np.float32(reward), bool(report.done), info

    def close(self) -> None:
        if self.brain is not None and self.brain.body is not None:
            self.brain.body.close()
            self.brain.environment.close()

    @staticmethod
    def _obs(inputs: dict, first: bool, last: bool, terminal: bool, r_seg: float, r_dec: float, success: bool) -> dict:
        return {
            "vis": np.asarray(inputs["vis"], dtype=np.float32),
            "txt": np.asarray(inputs["txt"], dtype=np.float32),
            "vec": np.asarray(inputs["vec"], dtype=np.float32),
            "rew_seg": np.array([r_seg], dtype=np.float32),
            "rew_dec": np.array([r_dec], dtype=np.float32),
            "log_success": np.array([1.0 if success else 0.0], dtype=np.float32),
            "is_first": np.bool_(first),
            "is_last": np.bool_(last),
            "is_terminal": np.bool_(terminal),
        }
