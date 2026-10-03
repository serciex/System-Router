"""Stage 3: train the world model (route, level, cells) with r2dreamer inside the whole system.

The LLM, body and rewards run inside the environment worker; r2dreamer trains the two-head agent in the
main process. Checkpoints are written every --save-every updates and on exit, and training resumes from
`latest.pt` in the log directory (the replay buffer is not saved, so it refills after a resume).

    python scripts/train_wm.py
    python scripts/train_wm.py --set paths.runs=/content/drive/MyDrive/system-router/runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from body.factory import make_grid  # noqa: E402
from common.config import load_config, resolve  # noqa: E402
from wm.codec import ActionCodec  # noqa: E402
from wm.r2d import add_r2dreamer_to_path  # noqa: E402


def compose_r2dreamer(cfg):
    """r2dreamer's hydra config with this project's environment, sizes and log directory."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    root = add_r2dreamer_to_path(cfg.paths.r2dreamer)
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        conf = compose(config_name="configs", overrides=["env=crafter", f"model={cfg.dreamer.model}"])
    OmegaConf.set_struct(conf, False)
    d = cfg.dreamer
    conf.device = d.device
    conf.batch_size, conf.batch_length = int(d.batch_size), int(d.batch_length)
    conf.logdir = str(resolve(cfg.paths.runs) / "wm")
    conf.env.task = "router"
    conf.env.steps = int(d.steps)
    conf.env.env_num = 1          # one worker holds the LLM; a second would load another copy
    conf.env.eval_episode_num = 0  # evaluation runs separately (held-out adapters)
    conf.env.train_ratio = int(d.train_ratio)
    conf.env.action_repeat = 1
    for part in ("encoder", "decoder"):
        conf.env[part].mlp_keys = "^(vis|txt|vec)$"
        conf.env[part].cnn_keys = "$^"
    conf.model.rep_loss = d.rep_loss
    conf.model.imag_horizon = int(d.imag_horizon)
    conf.model.compile = False
    OmegaConf.resolve(conf)
    return conf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--save-every", type=int, default=1000, help="updates between checkpoints")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing latest.pt")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    conf = compose_r2dreamer(cfg)

    import torch
    import tools  # r2dreamer
    from buffer import Buffer
    from envs.parallel import ParallelEnv
    from trainer import OnlineTrainer

    from wm.agent import TwoHeadDreamer
    from wm.gym_env import RouterEnv

    tools.set_seed_everywhere(conf.seed)
    logdir = Path(conf.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir)
    logger.log_hydra_config(conf)

    cfg_dict = cfg.to_dict()
    train_envs = ParallelEnv(lambda index: (lambda: RouterEnv(cfg_dict, seed=index)), 1, conf.device)
    codec = ActionCodec(make_grid(cfg))
    agent = TwoHeadDreamer(conf.model, train_envs.observation_space, train_envs.action_space, codec).to(conf.device)

    checkpoint = logdir / "latest.pt"
    if checkpoint.exists() and not args.fresh:
        state = torch.load(checkpoint, map_location=conf.device)
        agent.load_state_dict(state["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, state["optims_state_dict"])
        print(f"Resumed from {checkpoint}")

    def save() -> None:
        torch.save({"agent_state_dict": agent.state_dict(),
                    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent)}, checkpoint)

    update = agent.update
    counter = {"n": 0}

    def update_and_save(replay_buffer):
        metrics = update(replay_buffer)
        counter["n"] += 1
        if counter["n"] % args.save_every == 0:
            save()
        return metrics

    agent.update = update_and_save
    trainer = OnlineTrainer(conf.trainer, Buffer(conf.buffer), logger, logdir, train_envs, None)
    try:
        trainer.begin(agent)
    finally:
        save()
        print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
