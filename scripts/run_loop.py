"""Stage 2: the untrained v3 loop on base Qwen. Measures success, think share, latency and narrowing.

    python scripts/run_loop.py --episodes 20                       # act, think when needed
    python scripts/run_loop.py --episodes 20 --think-always        # baseline: think every step
    python scripts/run_loop.py --episodes 20 --adapter fallback    # universal fallback on the same tasks
    python scripts/run_loop.py --episodes 20 --adapter fallback --grounding native
    python scripts/run_loop.py --episodes 50 --shadow-think 0.3    # also record think answers for training data
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain.step import Brain  # noqa: E402
from common.config import load_config, resolve  # noqa: E402
from environments import make_environment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--adapter", default=None, help="web | fallback | code")
    parser.add_argument("--think-always", action="store_true")
    parser.add_argument("--grounding", choices=["narrowing", "native"], default="narrowing")
    parser.add_argument("--shadow-think", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    adapter = args.adapter or cfg.body.adapter
    name = f"{adapter}_{'think' if args.think_always else 'loop'}_{args.grounding}"
    out_dir = resolve(cfg.paths.runs) / "loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    integration = make_environment(cfg)
    brain = Brain(cfg, integration, log_path=out_dir / f"steps_{name}.jsonl", think_always=args.think_always,
                  grounding=args.grounding, shadow_think=args.shadow_think)

    episodes = []
    try:
        for episode in range(args.episodes):
            brain.reset(seed=episode, adapter=adapter)
            reports = []
            while True:
                report = brain.step()
                reports.append(report)
                if report.done:
                    break
            modes = [r.mode for r in reports]
            episodes.append({
                "task": getattr(integration, "task", None),
                "success": bool(reports[-1].success),
                "steps": len(reports),
                "think_share": modes.count("think") / len(modes),
                "forced_think": sum(r.forced_think for r in reports),
                "find_calls": modes.count("find"),
                "waits": modes.count("wait"),
                "latency_ms": statistics.mean(r.outcome.latency_ms for r in reports),
                "narrowing_depth": statistics.mean(r.narrowing_depth for r in reports),
            })
            print(json.dumps(episodes[-1]))
    finally:
        if brain.body is not None:
            brain.body.close()
        integration.close()

    def mean(key):
        return statistics.mean(e[key] for e in episodes) if episodes else 0.0

    summary = {"run": name, "episodes": len(episodes), "success_rate": mean("success"), "mean_steps": mean("steps"),
               "think_share": mean("think_share"), "mean_latency_ms": mean("latency_ms"),
               "mean_narrowing_depth": mean("narrowing_depth"), "per_episode": episodes}
    (out_dir / f"summary_{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))


if __name__ == "__main__":
    main()
