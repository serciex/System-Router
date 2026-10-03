# System-Router

System Router for System 1 and System 2 thinking. It uses a crawler with predefined contracts for interacting with environments.

One universal brain operates any environment through interchangeable bodies that all follow the same contract:

- **A frozen VLM** (Qwen3.5) used two ways. **System 1** picks from verified options in a single forward pass (the [system-one](https://github.com/sgoedecke/system-one) method). **System 2** reasons with the screenshot, plans the task list and takes over when System 1 is unsure.
- **A trained world model** ([R2-Dreamer](https://github.com/NM512/r2dreamer)) that learns the dynamics and decides only three things per step: **which route** (System 1 or 2), **what level of detail**, and **which grid cells** to look at.
- **A body contract built once.** Each environment only needs a thin adapter. Navigation and action are two decision models that run in parallel, each with its own sub-contract.

Design: [docs/Universal Brain, Interchangeable Bodies Technical Design Spec v2.md](docs/Universal%20Brain,%20Interchangeable%20Bodies%20Technical%20Design%20Spec%20v2.md). Contract: [body/CONTRACT.md](body/CONTRACT.md).

## Layout

```
body/           the contract, built once
  schema.py       Target, Option, Observation, Outcome, Capabilities
  grid.py         levels and cells (level 1 = 3x3 directions around the anchor)
  contract.py     Body: sense, observe(level, cells), act(navigation, action, arg)
  core.py         protected core: completion checks, latency, executor limits
  log.py          JSON-lines step log (System 2 replans from it)
  conformance.py  checks every adapter must pass
  adapters/       MiniWoB DOM, accessibility tree, vision (OmniParser), degraded variants
brain/          the frozen-LLM roles
  llm.py          load Qwen3.5 (processor + image-text model), generation, JSON parsing
  system1.py      constrained one-pass choices with the goal context cached across steps
  system2.py      planning and fallback decisions with the image
  goal_context.py goal, task list, current subtask
  labeler.py      privileged labels: important targets and the level they need (training only)
  features.py     VLM vision-patch cells and text hidden state for the world model
  step.py         Brain: one step of the whole system
wm/             the trained part
  codec.py        flat action [route | level | cells] and masks
  dataset.py      collected episodes and the sequence sampler
  hindsight.py    labels from verified success: important cells, System 1 wrong
  supervised.py   world model + cells head + escalation head (offline)
  baseline.py     the same heads without the world model
  evaluate.py     held-out sweep that suggests thresholds and cell budget
  heads_policy.py run-time routing and segmentation from the heads
  rewards.py      r_seg and r_dec (RL fine-tuning)
  rules.py        stage-1 rules in place of the world model
  gym_env.py      the whole system as an r2dreamer environment
  agent.py        r2dreamer with two reward heads and two critics (RL fine-tuning)
environments/   MiniWoB++ (WebArena and OSWorld come in stages 4-5)
scripts/        check_adapters, run_stage1, collect, train_offline, train_wm
configs/        default.yaml (paths, model, grid, rewards, environments)
notebooks/      colab.ipynb
system one/     submodule: sgoedecke/system-one (reference for the System 1 method)
dreamer v3/     submodule: NM512/r2dreamer (world model)
```

## Setup

```bash
git clone --recurse-submodules https://github.com/serciex/System-Router.git
cd System-Router
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match your GPU
pip install -r requirements.txt
```

MiniWoB++ also needs Chrome or Chromium. The model is not in the repository. Either point `paths.model` at a Hugging Face id (`--set paths.model=Qwen/Qwen3.5-9B`) or download it into `models/Qwen3.5-9B`.

On an 8 GB GPU use `--set paths.model=Qwen/Qwen3.5-4B --set llm.quantization=4bit`. For the full model, use Colab with an A100 ([notebooks/colab.ipynb](notebooks/colab.ipynb)).

## Running the stages

The world model is trained **supervised first, on data collected once**, then fine-tuned with RL:

```bash
python -m pytest -q tests                                            # no model, no browser
python scripts/check_adapters.py --adapters dom a11y "degraded:dom"  # 0: contract conformance
python scripts/run_stage1.py --episodes 20 --route s1                # 1: rules, System 1 + escalation
python scripts/run_stage1.py --episodes 20 --route s2                #    baseline: System 2 alone
python scripts/collect.py --episodes 500                             # 2: both systems answer every step
python scripts/collect.py --episodes 100 --heldout                   #    held-out adapters, for evaluation
python scripts/train_offline.py                                      # 3: world model + supervised heads (no LLM)
python scripts/train_offline.py --baseline                           #    plain classifier to compare against
python scripts/run_stage1.py --episodes 20 --policy heads            # 4: online with the trained heads
python scripts/train_wm.py --init-from runs/offline/latest.pt        # 5: RL fine-tuning (later)
```

Labels for stage 3 come from hindsight. Important cells are where the agent actually acted in episodes MiniWoB marked successful. "System 1 wrong" means System 1 disagreed with System 2 on the same step. `train_offline.py` writes `tuning.json` with suggested thresholds, which stage 4 reads.

Any config value can be overridden with `--set key=value`, for example `--set env.adapter_pool='[[dom],[a11y],["degraded:dom"]]'` to train across adapters.

## Status

All code is written but has **not been run yet**. The model weights are added once the full system is built. Things to verify first:

1. Qwen3.5 with System 1's cache reuse (its hybrid linear attention must deep-copy and extend correctly; otherwise `System1(reuse_cache=False)`).
2. The vision-feature call (`get_image_features`) against the installed `transformers` version.
3. The accessibility and OmniParser adapters against the installed MiniWoB, Selenium and OmniParser versions.
4. System 1 latency on the target GPU, before setting `rewards.expected_latency_ms`.
