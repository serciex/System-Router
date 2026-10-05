"""Stage 1 gate: run contract v0.3 conformance (including the hidden split) on adapters. No LLM needed.

    python scripts/check_adapters.py --adapters web fallback --task click-test-2
    python scripts/check_adapters.py --adapters code --set env.suite=workspace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from body.conformance import check_faithful, run_conformance  # noqa: E402
from body.factory import make_adapter  # noqa: E402
from common.config import load_config, resolve  # noqa: E402
from environments import make_environment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--adapters", nargs="+", default=["web"])
    parser.add_argument("--task", default="click-test-2")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set + [f"env.tasks=[{args.task}]"])
    integration = make_environment(cfg)
    failed = False
    try:
        for name in args.adapters:
            adapter = make_adapter(name, integration, cfg)
            report = run_conformance(adapter, integration, seed=0, hidden_dir=resolve(cfg.conformance.hidden_dir))
            integration.reset(0)
            adapter.reset(0)
            clickable = [i for i in adapter.find() if i.kind == "element" and "click" in i.verbs]
            if clickable:
                report.results["faithful"] = check_faithful(adapter, integration, clickable[0].handle, "click")
            print(report)
            failed = failed or not report.passed
    finally:
        integration.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
