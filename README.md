# System-Router

System Router for System 1 and System 2 thinking. It uses a crawler with predefined contracts for interacting with environments.

One brain operates any environment through interchangeable bodies that all follow the same contract:

- **One LLM** (Qwen3.5, native vision) with three modes: **act** (two parallel constrained picks in one pass each, the [system-one](https://github.com/sgoedecke/system-one) method), **silent steps** (latent self-direction, stage 6), and **think** (token reasoning for plans, word goals and code).
- **Core actions in every body:** **think**, **find** (refresh the options; run automatically when the window changes) and **wait** (no inference until something changes).
- **A body contract built once.** A protected sensory channel (what the model sees) and an adapter-owned interaction channel (what can be done). Positions on surfaces come from a shared narrowing library. The model later writes its own adapters.

Design: [spec v3](docs/Universal%20Brain,%20Interchangeable%20Bodies%20Technical%20Design%20Spec%20v3.md). Contract: [body/CONTRACT.md](body/CONTRACT.md) (v0.3). Handoff: [docs/HANDOFF.md](docs/HANDOFF.md).

## Layout

```
body/             the contract, built once
  vocab.py          canonical roles, verbs, core actions, sensor types
  schema.py         Observation, manifests, Item, Target, Slot, Option, Outcome
  narrowing.py      regions inside surfaces (9 for 2D, 3 for 1D, plus here)
  prompt.py         fixed prompt layout built by the core
  contract.py       Body: observe, find and the body slot, questions, act, wait
  core.py           protected core: change check, think rule, goal age, limits, completion checks
  conformance.py    v0.3 checks plus the hidden split
  factory.py        integration + adapter -> Body
  adapters/         web (MiniWoB DOM), fallback (whole window), code workspace;
                    a11y, vision, degraded are pending port to v0.3
brain/            the LLM side
  llm.py            load Qwen3.5, generation, JSON parsing
  system1.py        act: one prefill with the screen, one constrained pass per question
  think.py          think: plans, word goals, picks, points, code
  grounding.py      native coordinate pointing (compared against narrowing)
  goal_context.py   goal, task list, current subtask, word goal
  step.py           the v3 step loop
environments/     integrations: MiniWoB++, real desktop (mss + pyautogui), code workspace
scripts/          check_adapters (stage 1 gate), run_loop (stage 2)
configs/          default.yaml
tests/            contract, narrowing, conformance, prompt, rewards (fakes, no model)
wm/, some brain/ and scripts/ files   v2 world-model design, marked OBSOLETE, kept for reference
system one/       submodule: sgoedecke/system-one
dreamer v3/       submodule: NM512/r2dreamer (unused under v3)
```

## Setup

```bash
git clone --recurse-submodules https://github.com/serciex/System-Router.git
cd System-Router
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match your GPU
pip install -r requirements.txt
```

MiniWoB++ needs Chrome or Chromium. The desktop body needs a display (Xvfb on Colab or servers). The model is not in the repository: point `paths.model` at a Hugging Face id or a local folder. On an 8 GB GPU use `--set paths.model=Qwen/Qwen3.5-4B --set llm.quantization=4bit`.

## Running

```bash
python -m pytest -q tests                                                   # no model, no browser
python scripts/check_adapters.py --adapters web fallback                    # stage 1 gate
python scripts/run_loop.py --episodes 20                                    # stage 2: untrained loop
python scripts/run_loop.py --episodes 20 --think-always                     # baseline: think every step
python scripts/run_loop.py --episodes 20 --adapter fallback --grounding native
```

## Status

Stages 0 to 2 are written but **nothing has been run**. Stages 3 to 8 (data with counterfactual branches, LoRA training, adapter writing, silent mode, more bodies) are not implemented. To verify first: Qwen3.5 cache copy and extension after an image prefill (else `System1(reuse_cache=False)`), the multimodal chat template with the sentinel split, MiniWoB action names, and latency per step.
