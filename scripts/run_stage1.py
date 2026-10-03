"""Run the whole system online and measure it.

Stage 1 uses hand-written rules instead of the world model (no training); stage 4 uses the trained
supervised heads with the thresholds from tuning.json. Gate for both: task success, System 1 share and
seconds per step, against System 2 alone on the same tasks.

    python scripts/run_stage1.py --episodes 20 --route s1
    python scripts/run_stage1.py --episodes 20 --route s2      # reasoning-only baseline
    python scripts/run_stage1.py --episodes 20 --policy heads  # stage 4: trained world model heads
    python scripts/run_stage1.py --set llm.quantization=4bit --set paths.model=Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from body.factory import make_grid  # noqa: E402
from brain.step import Brain  # noqa: E402
from common.config import load_config, resolve  # noqa: E402
from environments import make_environment  # noqa: E402
from wm.rules import RulePolicy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[], help="override, e.g. llm.quantization=4bit")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--route", choices=["s1", "s2"], default="s1")
    parser.add_argument("--policy", choices=["rules", "heads"], default="rules")
    parser.add_argument("--adapters", nargs="*", default=None, help="adapter set for every episode, e.g. dom")
    parser.add_argument("--heldout", action="store_true", help="use the held-out adapter sets")
    parser.add_argument("--labeler", action="store_true", help="also score steps with the LLM labeler")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    run_name = f"{args.policy}_{args.route}" + ("_heldout" if args.heldout else "")
    out_dir = resolve(cfg.paths.runs) / "online"
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = make_environment(cfg)
    brain = Brain(cfg, environment, use_labeler=args.labeler, log_path=out_dir / f"steps_{run_name}.jsonl")
    heads = load_heads(cfg) if args.policy == "heads" else None
    rules = RulePolicy(make_grid(cfg), route=args.route)

    episodes = []
    try:
        for episode in range(args.episodes):
            inputs = brain.reset(seed=episode, adapters=args.adapters, evaluate=args.heldout,
                                 features=heads is not None)
            if heads is not None:
                heads.reset()
            reports = []
            while True:
                decision = heads.decide(inputs) if heads is not None else rules.decide(brain.body)
                report = brain.step(decision)
                reports.append(report)
                if report.done:
                    break
                if heads is not None:
                    inputs = brain.wm_inputs()
                else:
                    brain.advance_context()
            decided = [r for r in reports if not r.forced_plan]
            episodes.append({
                "task": environment.task,
                "adapters": brain.adapter_names,
                "success": bool(reports[-1].success),
                "steps": len(reports),
                "s1_share": sum(r.route_used == "s1" for r in decided) / max(len(decided), 1),
                "escalations": sum(r.escalated for r in reports),
                "latency_ms": statistics.mean(r.outcome.latency_ms for r in reports),
                "recall": statistics.mean(r.seg_info.get("recall", 0.0) for r in reports) if args.labeler else None,
                "correct": statistics.mean(1.0 if r.correct else 0.0 for r in reports if r.correct is not None)
                if any(r.correct is not None for r in reports) else None,
            })
            print(json.dumps(episodes[-1]))
    finally:
        if brain.body is not None:
            brain.body.close()
        environment.close()

    summary = {
        "policy": args.policy,
        "route": args.route if args.policy == "rules" else "learned",
        "heldout": args.heldout,
        "episodes": len(episodes),
        "success_rate": statistics.mean(e["success"] for e in episodes) if episodes else 0.0,
        "s1_share": statistics.mean(e["s1_share"] for e in episodes) if episodes else 0.0,
        "mean_steps": statistics.mean(e["steps"] for e in episodes) if episodes else 0.0,
        "mean_latency_ms": statistics.mean(e["latency_ms"] for e in episodes) if episodes else 0.0,
        "per_episode": episodes,
    }
    (out_dir / f"summary_{run_name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))


def load_heads(cfg):
    """The supervised world model and its run-time thresholds (config values, or tuning.json if present)."""
    import torch

    from wm.codec import ActionCodec
    from wm.gym_env import RouterEnv
    from wm.heads_policy import HeadsPolicy
    from wm.r2d import compose_r2dreamer
    from wm.supervised import SupervisedDreamer

    conf = compose_r2dreamer(cfg, logdir_name="offline")
    codec = ActionCodec(make_grid(cfg))
    model = SupervisedDreamer(conf.model, RouterEnv(cfg).observation_space, codec.space, codec).to(conf.device)
    checkpoint = resolve(cfg.heads.checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=conf.device)["agent_state_dict"])
    model.clone_and_freeze()
    settings = {"escalate_threshold": cfg.heads.escalate_threshold, "cell_threshold": cfg.heads.cell_threshold}
    tuning = checkpoint.parent / "tuning.json"
    if cfg.heads.use_tuning and tuning.exists():
        suggested = json.loads(tuning.read_text())["suggested"]
        settings = {k: suggested[k] for k in settings}
        print(f"Using tuned thresholds {settings}")
    return HeadsPolicy(model, codec, level=int(cfg.heads.level), min_cells=int(cfg.heads.min_cells),
                       max_cells=int(cfg.heads.max_cells), **settings)


if __name__ == "__main__":
    main()
