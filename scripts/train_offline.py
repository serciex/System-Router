"""Stage 3: train the world model and its two supervised heads offline, then tune the run-time thresholds.

No LLM is loaded here: training only reads the collected episodes. Episodes collected with --heldout
(held-out adapters) are used for evaluation, so the numbers measure transfer to an unseen adapter.
Writes latest.pt and tuning.json (suggested thresholds) into runs/offline.

    python scripts/train_offline.py
    python scripts/train_offline.py --steps 50000 --set offline.batch_size=32
    python scripts/train_offline.py --baseline     # same heads without the world model (runs/offline_baseline)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from body.factory import make_grid  # noqa: E402
from common.config import load_config, resolve  # noqa: E402
from wm.codec import ActionCodec  # noqa: E402
from wm.dataset import SequenceSampler, load_episodes  # noqa: E402
from wm.evaluate import episode_batch, suggest, sweep_cells, sweep_escalation  # noqa: E402
from wm.hindsight import attach_labels, summary  # noqa: E402
from wm.r2d import compose_r2dreamer  # noqa: E402

THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]


def evaluate(model, episodes: list[dict], codec: ActionCodec, cfg) -> dict:
    level_index = codec.levels.index(int(cfg.heads.level))
    start = sum(codec.cell_counts[:level_index])
    stop = start + codec.cell_counts[level_index]
    probs, targets, labeled, p_wrong, wrong, wrong_labeled = [], [], [], [], [], []
    for episode in episodes:
        cells, escalate = model.predict_sequence(episode_batch(episode["arrays"]))
        arrays = episode["arrays"]
        probs.append(cells[0, :, start:stop])
        targets.append(arrays["cell_target"][:, start:stop])
        labeled.append(arrays["cell_mask"][:, 0] > 0)
        p_wrong.append(escalate[0, :, 0])
        wrong.append(arrays["s1_wrong"][:, 0])
        wrong_labeled.append(arrays["s1_mask"][:, 0] > 0)
    cat = np.concatenate
    cell_rows = sweep_cells(cat(probs), cat(targets), cat(labeled), THRESHOLDS,
                            int(cfg.heads.min_cells), int(cfg.heads.max_cells))
    escalation_rows = sweep_escalation(cat(p_wrong), cat(wrong), cat(wrong_labeled), THRESHOLDS)
    return {"suggested": suggest(cell_rows, escalation_rows, float(cfg.offline.target_recall),
                                 float(cfg.offline.target_caught)),
            "cells": cell_rows, "escalation": escalation_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore an existing latest.pt")
    parser.add_argument("--baseline", action="store_true", help="train the plain per-step classifier instead")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    conf = compose_r2dreamer(cfg, logdir_name="offline_baseline" if args.baseline else "offline")
    import torch

    from wm.gym_env import RouterEnv
    from wm.supervised import SupervisedDreamer

    grid = make_grid(cfg)
    codec = ActionCodec(grid)
    data_root = resolve(cfg.collect.out)
    reference = cfg.hindsight.s1_reference
    train = attach_labels(load_episodes(data_root / "train"), grid, codec, reference)
    heldout = attach_labels(load_episodes(data_root / "heldout"), grid, codec, reference) \
        if (data_root / "heldout").exists() else []
    print("train data:", json.dumps(summary(train)))
    print("held-out data:", json.dumps(summary(heldout)) if heldout else "none (evaluating on training episodes)")

    logdir = Path(conf.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    obs_space = RouterEnv(cfg).observation_space  # spaces only; no LLM is loaded
    if args.baseline:
        from wm.baseline import StepClassifier

        input_dim = sum(int(obs_space[k].shape[0]) for k in ("vis", "txt", "vec"))
        model = StepClassifier(input_dim, codec.n_cells, device=conf.device)
    else:
        model = SupervisedDreamer(conf.model, obs_space, codec.space, codec,
                                  cells_scale=cfg.offline.loss_scales.cells,
                                  escalate_scale=cfg.offline.loss_scales.escalate).to(conf.device)
    checkpoint = logdir / "latest.pt"
    if checkpoint.exists() and not args.fresh:
        model.load_state_dict(torch.load(checkpoint, map_location=conf.device)["agent_state_dict"])
        if not args.baseline:
            model.clone_and_freeze()
        print(f"Resumed from {checkpoint}")

    def save() -> None:
        torch.save({"agent_state_dict": model.state_dict()}, checkpoint)

    sampler = SequenceSampler(train, int(cfg.offline.batch_size), int(cfg.offline.batch_length), seed=int(cfg.features.seed))
    steps = args.steps or int(cfg.offline.steps)
    eval_episodes = heldout or train
    with (logdir / "metrics.jsonl").open("a") as log:
        for step in range(1, steps + 1):
            metrics = model.supervised_update(sampler.sample())
            if step % 100 == 0:
                log.write(json.dumps({"step": step, **metrics}) + "\n")
                log.flush()
            if step % int(cfg.offline.eval_every) == 0 or step == steps:
                model.eval()
                report = evaluate(model, eval_episodes, codec, cfg)
                model.train()
                print(f"[{step}] loss {metrics['opt/loss']:.3f} suggested {json.dumps(report['suggested'])}")
                (logdir / "tuning.json").write_text(json.dumps(report, indent=2))
                save()
    save()
    print(f"Saved {checkpoint} and {logdir / 'tuning.json'}")


if __name__ == "__main__":
    main()
