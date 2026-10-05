# System Router: handoff summary

State as of 2026-10-04. Read this before touching anything.

- Owner: Mawunge Teye (GitHub `serciex`)
- Repository: https://github.com/serciex/System-Router (local: `C:\Users\serci\OneDrive\Desktop\Projects\System Router`)
- **Current design: [spec v3](Universal%20Brain,%20Interchangeable%20Bodies%20Technical%20Design%20Spec%20v3.md) and [contract v0.3](../body/CONTRACT.md).** Spec v2 and contract v0.2 are kept as the record of the earlier world-model design.
- Status: **stages 0 to 2 of v3 are coded** (contract v0.3, starter adapters, untrained loop); stages 3 to 8 are not; nothing has been run; no model weights downloaded.

## 1. Goal

A persistent "soul" (one LLM plus memory) that can inhabit any body: website, desktop or VM, codebase, game, simulated robot. For a new body it identifies the system, queries how it works, writes its own adapter, and operates it well. **The project is not presented until adapter writing is a trained capability**: a model-written adapter for a held-out application passes conformance and performs close to a hand-built one.

## 2. Design in one page (spec v3)

- **One LLM** (Qwen3.5-9B, native vision), three modes in rising cost: **act** (two parallel constrained picks, the sgoedecke/system-one method), **silent steps** (latent steps fed back as soft tokens, own LoRA), **think** (token reasoning for plans, word goals, code).
- **Two questions** every step: "Where next?" (body slot targets, surfaces, `hold`) and "What now?" (verbs, `none`, and the core actions). Two-letter labels.
- **Core actions** (only in the action question, identical in every body): **think** (replaces "none fit"; forced by a hard rule while being learned), **find** (refreshes the body slot; run automatically by the core on window change), **wait** (no inference; core change check).
- **Body**: protected **sensory channel** (screens plus typed sensors, read-only to the LLM) and adapter-owned **interaction channel** (find, execute, outcome reports into history only). **Capability manifest** with a sensory half and an interaction half.
- **Positions**: adapters declare **surfaces**; a shared **narrowing library** (9 regions for 2D, 3 for 1D, plus `here`) picks positions within one step. To be compared against native coordinate grounding.
- **Adapters**: hand-built starter set (universal fallback on screen plus OS input, web/DOM, code workspace, then desktop via AT-SPI, VR, physics); the model then writes app-specific ones, sandboxed, with a hidden conformance split.
- **Training**: frozen base, working LoRA, EMA LoRA (acting copy, source of all training targets), KL anchor to base, separate silent LoRA. Signal for picks: verified-success imitation (set targets) → think distillation → counterfactual branches (deterministic replay) → RL (e.g. GRPO). Think is rewarded only for turning a wrong pick into a right one. Silent mode is a parallel research track with two swap tests before injection (closest prior: Coconut).
- **Stages**: 0 contract v0.3 → 1 starter adapters → 2 untrained loop (+ narrowing vs grounding) → 3 data (≈5,000 verified think calls) → 4 LLM training of picks and think timing → **5 adapter writing (headline)** → 6 silent mode → 7 more bodies → 8 held-out evaluation. Held-out application: a Doom setup like the system-one demo.

## 3. Repository and code status

**Pushed** (`6756155`): v2 implementation. **Local, not pushed** (`17b2d42`): v2 supervised-first pipeline. **Uncommitted**: spec v3, contract v0.3, `body/CONTRACT_v0.2.md`, spec v2 edits, this file, and the v3 code below.

### v3 code (stages 0 to 2, written, not run)

| Path | What it does |
| --- | --- |
| `body/vocab.py`, `body/schema.py` | Canonical roles, verbs, core actions, sensor types; Observation, manifests, Item, Target, Slot, Option, Outcome (`unavailable` added) |
| `body/narrowing.py` | 9 regions (2D), 3 (1D), `here`; option text with coordinates; stops when small |
| `body/prompt.py` | Fixed prompt layout (instructions, history, screen, sensors, goals, status, options, questions) |
| `body/core.py` | Protected core: change check, hard think rule, goal age, limits, completion checks |
| `body/contract.py` | `Body`: observe (integration), find and body slot with stable ids and canonical labels, auto-find, questions (core actions only in "What now?"), act (action then navigation; groups expand; surfaces take a narrowing point; drag), revalidation to `unavailable`, wait, relative mode |
| `body/adapters/base.py` | v0.3 adapter interface: `manifest`, `find(scope)`, `invoke`, `act_at`, `move`, `reset`, `close` |
| `body/adapters/miniwob_dom.py` | Web starter adapter: canonical roles, grouping and collapse, canvases as surfaces |
| `body/adapters/fallback.py` | Universal fallback: whole window as one surface over any screen driver |
| `body/adapters/code_workspace.py` | Code workspace adapter: files, folders, search hits, terminal; open, read, search, write, run |
| `body/conformance.py` | Manifest, items, stable handles, scope, surface, faithful, coverage, reset, hidden split |
| `body/factory.py` | Integration plus adapter (web, fallback, code) into a Body |
| `environments/base.py`, `miniwob.py`, `desktop.py`, `workspace.py` | Integrations (sensory channel): MiniWoB, real desktop (mss, pyautogui), sandboxed code workspace rendered as a screen |
| `brain/system1.py` | Act: one multimodal prefill per step, one constrained pass per question, labels unique across questions, `ask_extra` for narrowing |
| `brain/think.py` | Think: plan, decide (labels, point, word goal, replan, missing) |
| `brain/grounding.py` | Native coordinate pointing for the narrowing comparison |
| `brain/step.py` | v3 loop: auto-find, plan, picks, hard think rule, think, find, wait, narrowing within the step, shadow think for data, step records with full prompts |
| `scripts/check_adapters.py`, `scripts/run_loop.py` | Stage 1 gate; stage 2 loop with think-always baseline and grounding comparison |
| `tests/` | v3 tests on fakes (contract, narrowing, conformance, prompt, rewards); v2 tests skipped |

### Kept but obsolete or pending (nothing deleted; ask the owner before removing anything)

| Path | Status |
| --- | --- |
| `wm/*`, `brain/features.py`, `brain/labeler.py`, `brain/system2.py`, `body/grid.py`, `scripts/collect.py`, `train_offline.py`, `train_wm.py`, `run_stage1.py` | Marked OBSOLETE (v2 world model); some no longer import. `dec_reward` in `wm/rewards.py` still matches v3 |
| `body/adapters/miniwob_a11y.py`, `vision.py`, `omniparser.py`, `degraded.py` | Marked PENDING PORT to v0.3 |
| `dreamer v3/` | Unused under v3 |

### Not implemented yet (stages 3 to 8)

Counterfactual branches (deterministic replay), think-call dataset builder, working/EMA LoRA training with KL, adapter-writing stage (repair, port, new) with sandboxed conformance runs, silent LoRA and swap tests, desktop AT-SPI, VR and physics adapters, Colab notebook update.

### To verify once weights exist

Qwen3.5 hybrid attention (Gated DeltaNet) cache snapshot and extension for parallel picks and silent steps; image-token prefill cost; native coordinate grounding format; MiniWoB, Selenium, AT-SPI APIs; latency per step.

## 4. Environment and hardware

- `models/Qwen3.5-9B/`: Hugging Face repo **without weights** (git-ignored). Weights are added only when the owner says so.
- Owner's laptop: RTX 5050 Laptop 8 GB, 16 GB RAM, Windows 11 (9B only heavily quantized). Heavy work on **Google Colab** (A100; no Docker, VMs or display; use Xvfb), possibly HPC.
- Environments: sandboxed only (MiniWoB++, later WebArena, OSWorld, a simulator), never the live web. Adapter writing only for authorized systems.

## 5. Open decisions

Gate numbers per stage; reward constants and think, find, wait costs; hard think threshold schedule; stereo VR handling (one eye provisionally); per-body LoRA (deferred).

## 6. How the owner works (follow these)

- **Explain understanding and plan, then wait for a go-ahead** before multi-part code changes.
- **Do not run** tests, trainings or end-to-end checks unless asked. (The v3 draft also lists "smoke-test everything connected to a change"; confirm with the owner which applies.)
- **Never download model weights**; clone repositories only. Everything goes inside the project folder.
- **Ask before pushing** and list what will be pushed. Commits here keep the `Co-Authored-By: Claude` line.
- Code style: short minimalist docstrings, minimal inline comments per functional block, no changelog comments, no em dashes; undo diagnostic changes that were not accepted as the fix.
- Prefers concise answers, figures for structure, and method-level explanations when discussing design.
- Never use colons in emails or paragraphs drafted for the owner to send.
