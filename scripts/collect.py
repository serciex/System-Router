"""OBSOLETE under spec v3 (world model removed). Kept for reference; may not import. Ask the owner before deleting.

Stage 2: collect episodes once, with System 1 and System 2 both answering every step.

The expensive LLM calls happen here, once. The world model then trains offline on the saved data
(scripts/train_offline.py) as many times as needed. Segmentation uses the stage-1 rule (every cell that
holds a target, finest level) so every target is offered and hindsight labels are complete.

    python scripts/collect.py --episodes 500 --execute s2
    python scripts/collect.py --episodes 100 --heldout          # episodes with held-out adapters, for evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from body.factory import make_grid  # noqa: E402
from brain.step import Brain  # noqa: E402
from common.config import load_config, resolve  # noqa: E402
from environments import make_environment  # noqa: E402
from wm.codec import S1, S2, ActionCodec, Decision  # noqa: E402
from wm.dataset import EpisodeWriter  # noqa: E402
from wm.rules import RulePolicy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--execute", choices=["s2", "s1", "mix"], default=None)
    parser.add_argument("--heldout", action="store_true", help="use the held-out adapter sets")
    parser.add_argument("--seed-offset", type=int, default=0, help="first episode seed (to extend a dataset)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    episodes = args.episodes or int(cfg.collect.episodes)
    execute = args.execute or cfg.collect.execute
    out_dir = resolve(cfg.collect.out) / ("heldout" if args.heldout else "train")
    grid = make_grid(cfg)
    codec = ActionCodec(grid)
    rules = RulePolicy(grid, route="s1")

    environment = make_environment(cfg)
    brain = Brain(cfg, environment, use_labeler=False, log_path=out_dir / "steps.jsonl")
    writer = EpisodeWriter(out_dir)
    successes = 0
    try:
        for episode in range(args.seed_offset, args.seed_offset + episodes):
            inputs = brain.reset(seed=episode, evaluate=args.heldout)
            writer.start(task=environment.task, adapters=brain.adapter_names, heldout=args.heldout, seed=episode)
            while True:
                decision = rules.decide(brain.body)
                report, record = brain.collect_step(decision, execute=execute)
                taken = Decision(route=S2 if report.route_used == "s2" else S1, level=decision.level, cells=decision.cells)
                writer.add(inputs, codec.encode(taken), record)
                if report.done:
                    break
                inputs = brain.wm_inputs()
            successes += int(bool(report.success))
            writer.finish(report.success)
            print(json.dumps({"episode": episode, "task": environment.task, "adapters": brain.adapter_names,
                              "steps": brain.steps, "success": bool(report.success)}))
    finally:
        if brain.body is not None:
            brain.body.close()
        environment.close()
    print(f"Collected {episodes} episodes into {out_dir}, success rate {successes / max(episodes, 1):.2f}")


if __name__ == "__main__":
    main()
