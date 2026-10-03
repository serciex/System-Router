"""Run the contract's conformance checks on each adapter (stage 0 gate). No LLM needed.

    python scripts/check_adapters.py --adapters dom a11y "degraded:dom" --task click-test-2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from body.conformance import check_faithful, run_conformance  # noqa: E402
from body.factory import make_adapter  # noqa: E402
from common.config import load_config  # noqa: E402
from environments import make_environment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--adapters", nargs="+", default=["dom"])
    parser.add_argument("--task", default="click-test-2")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set + [f"env.tasks=[{args.task}]"])
    environment = make_environment(cfg)
    failed = False
    try:
        for name in args.adapters:
            adapter = make_adapter(name, environment, cfg)
            report = run_conformance(adapter, seed=0)
            # Faithful execution: clicking the first clickable element must change the episode state.
            adapter.reset(0)
            clickable = [e for e in adapter.read().elements if "click" in e.native_actions]
            if clickable:
                report.results["faithful"] = check_faithful(
                    adapter, clickable[0].handle, "click", lambda frame, res: bool(res.get("ok")))
            print(report)
            failed = failed or not report.passed
    finally:
        environment.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
