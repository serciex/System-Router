"""Stage 1: the whole system with hand-written rules instead of the world model (no training).

Gate: task success and System 1 share, measured against System 2 alone on the same tasks.

    python scripts/run_stage1.py --episodes 20 --route s1
    python scripts/run_stage1.py --episodes 20 --route s2      # reasoning-only baseline
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
    parser.add_argument("--adapters", nargs="*", default=None, help="adapter set for every episode, e.g. dom")
    parser.add_argument("--no-labeler", action="store_true", help="skip the labeler (no segmentation reward)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_dir = resolve(cfg.paths.runs) / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = make_environment(cfg)
    brain = Brain(cfg, environment, use_labeler=not args.no_labeler, log_path=out_dir / f"steps_{args.route}.jsonl")
    policy = RulePolicy(make_grid(cfg), route=args.route)

    episodes = []
    try:
        for episode in range(args.episodes):
            brain.reset(seed=episode, adapters=args.adapters, features=False)
            reports = []
            while True:
                report = brain.step(policy.decide(brain.body))
                reports.append(report)
                if report.done:
                    break
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
                "recall": statistics.mean(r.seg_info.get("recall", 0.0) for r in reports) if not args.no_labeler else None,
                "correct": statistics.mean(1.0 if r.correct else 0.0 for r in reports if r.correct is not None)
                if any(r.correct is not None for r in reports) else None,
            })
            print(json.dumps(episodes[-1]))
    finally:
        if brain.body is not None:
            brain.body.close()
        environment.close()

    summary = {
        "route": args.route,
        "episodes": len(episodes),
        "success_rate": statistics.mean(e["success"] for e in episodes) if episodes else 0.0,
        "s1_share": statistics.mean(e["s1_share"] for e in episodes) if episodes else 0.0,
        "mean_steps": statistics.mean(e["steps"] for e in episodes) if episodes else 0.0,
        "mean_latency_ms": statistics.mean(e["latency_ms"] for e in episodes) if episodes else 0.0,
        "per_episode": episodes,
    }
    (out_dir / f"summary_{args.route}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))


if __name__ == "__main__":
    main()
