# Universal Brain, Interchangeable Bodies: Technical Design Spec (v3)

Oct 4, 2026 · @Mawunge Teye

Supersedes [v2](Universal%20Brain,%20Interchangeable%20Bodies%20Technical%20Design%20Spec%20v2.md). Contract: [body/CONTRACT.md](../body/CONTRACT.md) v0.3. Status: design agreed, nothing implemented, nothing run.

## What changed from v2

| Area | v2 | v3 |
| --- | --- | --- |
| World model | Separate trained component (R2-Dreamer) choosing route and cells | **Removed.** Its role moves into the same LLM through silent steps and find. `dreamer v3/` is unused |
| Escalation | World model routes to System 2 | The LLM picks the core action **think** (with a hard safety rule while think is being learned) |
| Positions | Occupancy grid, levels 1/2/3 | **Surfaces** declared by adapters, narrowed by a shared **narrowing library** |
| What the model sees as options | Targets resolved for the cells the world model picked | The **body slot**, filled by **find** (run automatically on window change) |
| Core actions | "None fit" | **think, find, wait**, identical in every body |
| Self-direction | None | **Silent steps**: latent steps of the same LLM, fed back as soft tokens |
| LLM training | Staggered LoRA rounds | Continuous: working LoRA, EMA LoRA (acting copy), KL anchor to base |
| Body | Adapter resolves and executes | Adds a protected **sensory channel** and a **capability manifest**; adapters handle interaction only |
| Adapters | Hand-written, then workshop | Hand-built starter set with a **universal fallback**, then the model writes app-specific ones |
| Labels for training | Hindsight cell labels | Verified-success imitation, think distillation, **counterfactual branches**, then RL |

## 1. Goal and core principle

One brain operates any environment through interchangeable bodies. All variation between environments lives in adapters. The brain's interface (prompt layout, questions, core actions, decoding) never changes.

Any software that accepts input already has an input interface, and that interface is the foundation of an adapter. Every program with a screen and an input can therefore be operated, at minimum through the universal fallback adapter.

For a new body the system should identify the system, query how it works, write its own adapter, and operate it well. **The project is not presented until adapter writing is a trained capability**: an adapter the model wrote for a held-out application passes conformance and performs close to a hand-built one.

## 2. The brain: one LLM, three modes

The brain is one LLM (Qwen3.5-9B, native vision) with three modes in rising cost:

1. **Act:** answer the two questions directly, one constrained forward pass each (the sgoedecke/system-one method), sharing the cached prompt.
2. **Silent steps:** a few latent steps that steer the model's own next choices, with no tokens generated.
3. **Think in words:** token reasoning, used only when needed. Produces plans, word goals and code.

The model learns when acting directly is enough, when silent direction is enough, and when words are worth their cost.

## 3. Operation

### 3.1 One step

All modes run **inline** within the step (asynchronous thinking is deferred to real-time bodies, after stage 6).

1. The sensory channel delivers the current screen(s) and sensor readings.
2. The protected core builds the prompt (3.2). If its change check reports a window change, it first runs **find** automatically.
3. One LLM pass on the cached prompt. From it the model either:
   - **acts**: answers the navigation and action questions in parallel;
   - **takes a silent step**: a latent is fed back into the prompt as soft tokens, then it passes again (at most 3 per step);
   - **thinks**: generates tokens; the resulting word goal or plan is added to the prompt.
4. Picks go to the core, which has the adapter execute them, runs a scoped find, or starts a wait.
5. The environment changes; the next step begins with a new observation.

Only acting touches the environment. Silent steps and thinking only change the next pass. Execution order: the action applies to the current target, then the body moves to the navigation target. Clicking a new target therefore takes two steps; this is accepted.

### 3.2 Prompt layout

Built by generic code in the protected core. Only the options depend on the application.

1. Fixed instructions (identical everywhere)
2. History (past steps and adapter outcome reports, as text)
3. Current screen image(s) and sensor readings
4. Goals: word goal from the last think, plus silent-step soft tokens
5. Status lines from the core, e.g. `window changed since last find: no`
6. Options: the body slot contents
7. The two questions

Rules:

- Stable parts first, changing parts late, so the cached prefix (instructions and older history) is reused. History is append-only between summaries.
- History is summarized only at fixed points (subtask change), since summarizing invalidates the cache.
- Only the **current** screen is an image. Past screens appear in history as text.
- Sensors are encoded compactly: scalars and flags as short text lines, joint arrays as compact text, depth maps as an extra image. The manifest caps what is included.
- Options use **labels** (two-letter), not numeric indexes (`system1.labels: letters`).

### 3.3 Questions and core actions

| Question | Options |
| --- | --- |
| **Where next?** (navigation) | Targets in the body slot, surfaces, `hold` |
| **What now?** (action) | Verbs of the current target, verbs of `self`, `none`, and the core actions **think, find, wait** |

Core actions appear **only in the action question**, so the two answers can never conflict. A core action overrides the navigation answer for that step. Core actions are provided by the contract and the core, never by adapters.

### 3.4 think

- Switches to token reasoning. Outputs plans, word goals, and adapter code (through the code adapter).
- Replaces "none fit". Why it was used is recovered afterwards: if the reasoning ends in an existing option, the cause was uncertainty; if it invents an action or replans, the options did not fit.
- **Hard safety rule while think is being learned:** if the top pick's probability is below a threshold, the core forces think. The threshold is lowered as think timing is learned.

### 3.5 find and the body slot

- **find** asks the current application's adapter what is available in the current window. Results replace the **body slot** contents: navigation targets (elements, surfaces) and action verbs, mapped to the contract's canonical verbs and labels.
- The **core runs find automatically** whenever its change check reports a window change (no LLM call). The model calls find itself only for scoped or expanded lookups.
- Optional scope argument (default: whole window). Adapters return visible items only, grouped by container, with large groups collapsed; picking a collapsed group lets the next find open just that group.
- Executing a stale option returns an honest `unavailable` outcome.

### 3.6 wait

- No inference while waiting: no LLM, no silent steps.
- The core runs a cheap change check: pixel difference on watched regions, sensor thresholds, or OS or engine events. Watched regions come from the model's last state before the wait.
- Ends on a meaningful change or at a time cap (about 2 s for web; matched to the tick for real-time bodies), then the loop reprompts with the fresh state.

### 3.7 Silent steps

- Produced by the same LLM with a dedicated **silent LoRA**, active only during latent steps. With it off, the model is base Qwen.
- The latent (last hidden state through a small learned projection into input-embedding space) is placed in the prompt as **4 to 8 soft tokens** at position 4 of 3.2.
- Every latent or word goal carries the tick of the observation it was based on. The core drops goals on a window change or after 5 steps.
- **Injection gate:** silent latents are not injected into the acting prompt until they pass the swap tests (5.5). Until then only word goals steer.
- Closest prior work: Coconut (Hao et al., 2024), which feeds the last hidden state back as the next input embedding and needed a staged curriculum. Expect the same here.

### 3.8 Positions: surfaces and the narrowing library

- Adapters declare **surfaces** (canvas, slider track, map, or the whole window under the fallback) by box and dimensionality (1D or 2D).
- When a surface is picked, the shared narrowing library offers regions inside it: 9 for 2D (top-left to bottom-right), 3 for 1D, plus `here`. Option text includes coordinates ("top-left, x 0 to 640, y 0 to 360") to help grounding.
- **All narrowing picks happen within one step.** Nothing executes until `here`, which resolves to the region centre where the adapter acts.
- Drags are two points (from, to); drawing is a sequence of points.
- The library lives in the contract, written once. Adapters never implement narrowing.
- In relative-pointer bodies (games, VR), movement options replace positions.
- **To measure in stage 2:** narrowing against the model's native coordinate grounding (Qwen vision-language models are trained to point), by hit rate per target size and by latency.

## 4. Body

### 4.1 Two channels

| Channel | Direction | Produced by | Written by the LLM? |
| --- | --- | --- | --- |
| **Sensory** | Input only | Environment integration, inside the protected core | No |
| **Interaction** | Two-way (find, execute, outcome reports) | Adapter | Yes, through adapter writing |

- The sensory channel always contains at least one screen, of a declared kind (flat, stereo, camera, depth). Typed sensors accompany it where the body has them.
- Adapter outcome reports go into history only, never into the sensory channel. A buggy adapter cannot alter what the model sees.
- Sensor types: `joint_position`, `joint_velocity`, `contact`, `pose` (6DoF), `depth_map`, `force_torque`, `imu`, `scalar` (named, with units), `flag`. New types are added by a person, never by the LLM.

### 4.2 Capability manifest

| Field | Half | Contents |
| --- | --- | --- |
| `body_id`, `version` | Both | Identity and versioning |
| `screens` | Sensory | Kind, resolution, field of view, frame rate |
| `sensors` | Sensory | Name, type, units, rate |
| `pointer_mode` | Interaction | absolute, relative or none |
| `actuators` | Interaction | keyboard, pointer, gamepad, head_pose, controller, hand, joint; limits |
| `verbs` | Interaction | Canonical actions the adapter executes |
| `timing` | Both | Tick length, real-time or turn-based |

The sensory half is declared by environment integration (read-only to the LLM). The interaction half is declared by the adapter and checked by conformance. With several screens, choosing a screen is the coarsest narrowing level. **Privileged engine state lives in the training harness, never in the manifest**, so it cannot leak into deployment.

### 4.3 Adapters

- Every adapter maps its application onto the same fixed interface. Body-specific complexity (two VR hands, 3D reaching from 2D picks plus depth, smooth robot motion) is solved in adapter code, never by adding questions to the model.
- Labels must be meaningful ("Submit button", not internal IDs). Verbs use the canonical vocabulary. Conformance checks both.
- Adapters that ship with software are used when available.

**Starter set (hand-built), in build order:**

| Adapter | Built on | Covers | Notes |
| --- | --- | --- | --- |
| Universal fallback | Screen plus OS mouse and keyboard; whole window as one surface | Anything with a screen and input; games at first | Needs a display; on Colab a virtual display (Xvfb) |
| Web | DOM | MiniWoB, most websites | Existing MiniWoB DOM adapter is the starting point |
| Code workspace | Editor plus terminal | Required for adapter writing | Sandboxed |
| Desktop | OS accessibility tree via AT-SPI (xa11y) | Standard desktop apps | Linux and OSWorld compatible; Windows UI Automation later |
| VR | OpenXR input | VR bodies | Stage 7 |
| Physics | Simulator API | Simulated embodiment | Stage 7 |

## 5. Training

### 5.1 Model setup

- Base: Qwen3.5-9B, frozen.
- **Working LoRA:** receives gradients.
- **EMA LoRA:** the acting copy; slowly tracks the working LoRA.
- **KL anchor:** the base with adapters switched off (no extra copy).
- **Silent LoRA:** separate, for silent steps only.
- Training targets (think outputs, think labels, adapter-writing samples) come from the **EMA copy**, never the working copy, to avoid a self-distillation loop.
- The LLM learning rate can be set to zero at any time; the EMA adapter is checkpointed.
- **Update trigger:** LLM updates start once the untrained loop's baseline is measured **and** about 5,000 verified think-call examples exist.

### 5.2 Data collection

- Every step stores the full prompt, the pick distributions, any think output, and the outcome, so steps can be re-scored offline after LLM updates.
- At sampled steps both the direct pick and think answer (the existing `collect_step` structure).
- **Counterfactual branches:** deterministic environments (MiniWoB per seed) are replayed to the same step and the alternative pick is executed, to verify whether think changed a wrong pick into a right one.

### 5.3 Training signal for picks

In order of introduction:

1. **Verified-success imitation.** On steps from episodes the environment marked successful, the executed option is a positive target. When several options succeeded from equivalent states, use a **set target** that maximizes their total probability.
2. **Think distillation.** When think was called and the episode succeeded, think's final choice is the target for the pre-think pick distribution.
3. **Counterfactual labels.** Branches (5.2) label whether the direct pick was wrong and think right.
4. **RL last.** Episode-level policy gradient (e.g. GRPO) over the picks with the decision reward (5.6), once 1 to 3 work.

### 5.4 Teaching think

A base model rarely picks an abstain-style option. Early training relies on the hard safety rule (3.4) and on examples where think is the target: steps where the best non-think option was verified wrong (counterfactual branches) and token reasoning got it right. Early training is think-heavy by design; silent mode takes over as it learns.

### 5.5 Training silent mode (parallel research track)

Data: every think call yields (prompt state, full thinking text, decision tokens: chosen action plus word goal).

Losses:

1. **Decision reconstruction (main):** the frozen LLM, reading the silent latents as soft tokens, must reproduce the decision tokens.
2. **Thinking reconstruction (light weight):** the same decoder reproduces the full thinking text.
3. **Behaviour matching:** KL between the choice distribution given the word goal and given the silent latents at the same moment. Only the silent side trains; the choosing side is frozen or the EMA copy, so it cannot match by ignoring goals.

Curriculum: as in Coconut, replace text thoughts with latents gradually.

Injection is enabled only after both tests pass:

- **Swap test:** latents from a different situation must change the choices.
- **Same-state counterfactual test:** latents from a think with goal A versus goal B, in the same state, must lead to different choices.

Silent mode is **off the critical path**: adapter writing (the headline) does not depend on it.

### 5.6 Rewards (training only, in the protected core)

Decision reward per step:

```
s = min(1, expected_latency / latency)
correct: r = a + b * s
wrong:   r = -(p + q * (1 - s))
```

Ordering: fast-correct > slow-correct > fast-wrong > slow-wrong. `p` and `q` are chosen so that thinking under real uncertainty has better expected reward than guessing.

| Core action | Rule |
| --- | --- |
| think | Rewarded only when it **turns a wrong pick into a right one** (verified by counterfactual branch); otherwise pays the think cost. Never rewarded merely for disagreeing |
| find | Mostly automatic. A model-called find is useful if a following action uses something new from it; a small cost if the result matched the current slot; executing a stale option is penalized |
| wait | Waiting time is not decision latency but counts against an episode time budget; hitting the cap with no change carries a small cost |

## 6. Adapter writing (headline)

- The model explores the target application by operating software (file browser, editor, terminal, settings, the application's own UI) through existing adapters, mostly by acting.
- When enough is gathered, it uses think to plan and to write the adapter through the code adapter. The original task is **paused** while it does.
- Written adapters run sandboxed and must pass conformance, including a **hidden conformance split** kept outside the workspace and never shown to the model.
- Written adapters are scored only by behaviour (conformance plus downstream success), never by text similarity to a reference.
- Tasks rise in difficulty: **repair** a deliberately broken adapter, **port** an application that already has one, **new** application with none. One held-out application is sealed before training.
- Written adapters persist per body.
- Training data: successful adapter-writing episodes (expert iteration), with targets from the EMA copy.

## 7. Latency model (estimates, to be measured)

| Cost | Estimate (9B, A100) | Notes |
| --- | --- | --- |
| Prefill of a new screenshot | a few hundred ms | Dominant per-step floor |
| One constrained pick or silent step | about 10 to 30 ms | On the cached prompt |
| Think, a few hundred tokens | several seconds | Why words should be rare |

Routine steps should take about 0.4 to 0.5 s against about 6 s for a think-every-step agent. Levers: fewer image tokens (resolution, cropping), automatic find (no extra pass), reusing the cached image when the change check reports no change (which also makes wait nearly free), and the prompt ordering of 3.2. On the RTX 5050 (8 GB) the model only fits heavily quantized and will be several times slower.

## 8. Evaluation

- **Headline:** in the held-out application, for model-written versus hand-built adapters: conformance pass, zero-shot success, success after N practice episodes.
- Success, steps, **think share**, silent steps per step, and latency per step, against a think-every-step baseline.
- Narrowing versus native coordinate grounding (hit rate by target size, latency).
- **Same task across bodies** (e.g. completing one goal through the web and through the fallback adapter): tests that goal-level knowledge transfers between bodies.

## 9. Build stages

| # | Stage | Gate (numbers OPEN) |
| --- | --- | --- |
| 0 | Contract v0.3: channels, manifest, core actions, body slot, surfaces, narrowing library, canonical verbs and labels | Spec and conformance suite agreed |
| 1 | Starter adapters: fallback, web (MiniWoB), code workspace; conformance suite with hidden split | All three pass conformance |
| 2 | Untrained loop: base Qwen, picks plus think plus find plus wait, hard think rule; narrowing versus native grounding | Latency, success and think share measured against think-every-step |
| 3 | Data: stored prompts, think calls, counterfactual branches | About 5,000 verified think-call examples |
| 4 | LLM training of picks and think timing (working/EMA LoRA, KL) | Success up, think share down, latency down, no held-out regression |
| 5 | **Adapter writing** (headline): repair, port, new; held-out application | Model-written adapter passes conformance and performs close to hand-built |
| 6 | Silent mode (parallel track from stage 3): silent LoRA, swap tests, then injection | Both swap tests pass; silent steps replace some thinks with no success loss |
| 7 | More bodies: desktop (AT-SPI), games on the fallback, VR, physics | Each passes conformance; success measured |
| 8 | Held-out evaluation, including same task across bodies | Close to training performance |

Held-out application for stage 5: a Doom setup like the system-one demo (relative mode), chosen and sealed before training.

## 10. Safety

- The protected core (prompt building, change check, conformance including the hidden split, sandbox, rewards, executor limits) is read-only to the LLM.
- The sensory channel is produced by environment integration only; adapters cannot alter what the model sees.
- Written adapters run sandboxed with no network and time limits; only authorized systems may be targets of adapter writing.
- All training runs in sandboxed environments.

## 11. Risks

| Risk | Mitigation |
| --- | --- |
| No per-step signal for picks | Verified-success imitation, think distillation, counterfactual branches, RL last (5.3) |
| Think reward gamed by disagreement | Rewarded only for turning wrong into right, verified by branch |
| Base model never picks think | Hard safety rule while think is learned |
| Narrowing slower or less accurate than native grounding | Measured in stage 2; adopt the better one |
| Silent latents ignored or carrying nothing | Behaviour matching with a frozen choosing side; two swap tests before injection |
| Silent mode delays the headline | Parallel track, off the critical path |
| Self-distillation loop | Targets from the EMA copy only |
| Forgetting general skills | KL to base; LR can be zeroed; held-out checks |
| Fallback weak on small targets | Measured by target size; app adapters written for heavy-use applications |
| Practical: Colab has no display and no Windows APIs | Xvfb for the fallback; AT-SPI for desktop |

## 12. Remaining open decisions

- Gate numbers for every stage.
- Values of `a, b, p, q`, think cost, find and wait costs, the hard think threshold and its schedule.
- Stereo VR: one eye at first, both eyes when depth matters (provisional).
- Per-body LoRA adapters (deferred).

## 13. Obsolete from v2 (kept in the repository, not deleted)

The separate world model and everything built for it: R2-Dreamer agent and RL in imagination, the occupancy grid levels, cell and escalation heads, hindsight cell labels, routing by the world model, "none fit", confidence-scaled narrowing, recall monitor, staggered LoRA rounds, and the two-reward-head agent. File-level status is in [HANDOFF.md](HANDOFF.md).

## 14. Related work

- [Coconut, Training LLMs to Reason in a Continuous Latent Space (Hao et al., 2024)](https://arxiv.org/abs/2412.06769): latent reasoning by feeding hidden states back as input; closest to silent steps.
- [iSHIFT](https://arxiv.org/pdf/2512.22009) and [FaST](https://arxiv.org/abs/2408.08862): learned focus and fast/slow switching inside one model.
- [SkillWeaver](https://arxiv.org/pdf/2504.07079), [ALIGN](https://arxiv.org/abs/2505.21055), [EvoHarness-RL](https://arxiv.org/html/2608.05446v1): agents writing skills or interfaces, or using supplied harnesses; none writes perception and action adapters against a fixed contract with a protected conformance suite.
- [sgoedecke/system-one](https://github.com/sgoedecke/system-one) and [vLLM single-pass structured mode](https://github.com/vllm-project/vllm/pull/57250): the constrained single-pass choice mechanism.
- Distilling System 2 into System 1 (Yu et al.), Voyager (skill libraries), SayCan (choosing among verified skills).

The literature search was not exhaustive and should be repeated before publication.
