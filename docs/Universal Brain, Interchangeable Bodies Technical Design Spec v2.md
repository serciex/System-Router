# Universal Brain, Interchangeable Bodies: Technical Design Spec (v2)

Oct 2, 2026 · @Mawunge Teye

Supersedes [v1](Universal%20Brain,%20Interchangeable%20Bodies%20Technical%20Design%20Spec%20v1.md). The body contract is specified separately in [body/CONTRACT.md](../body/CONTRACT.md).

### What changed from v1

- The world model no longer drives decisions. It learns dynamics and decides **what to look at, how coarsely, and who acts**. All decisions stay with the LLM.
- The world model sees through the **frozen VLM's own features** (vision patches and a text hidden state). The separate BERT encoder and Dreamer CNN are removed.
- The body is a **target contract built once**, with one sub-contract per decision model (navigation and action). Environments are added with a thin adapter only.
- The world model is trained **supervised first**: episodes are collected once with both System 1 and System 2 answering every step, labeled in hindsight from the environment's verified success, and used offline to train two heads (which cells matter, will System 1 be wrong). **RL fine-tuning** with the two reward terms comes after, starting from that trained model.
- Rewards are **two separate terms**, segmentation and decision, used in the later RL stage.
- Scope is narrowed: **the LLM stays frozen, only the world model is trained**, across many adapters. LLM training comes later, to teach it to create adapters.
- Training uses **sandboxed environments only** (MiniWoB++, WebArena, OSWorld VM), never the live web.

## Overview and core principle

One universal brain operates any environment through interchangeable bodies that all follow the same contract. New environments are absorbed by writing a thin adapter, never by retraining the LLM.

The brain has two parts with a strict division of labor:

- **A frozen VLM**, used in two sampling modes. System 1 picks from verified options in a single forward pass. System 2 reasons, plans, picks or invents. The same model also serves as the labeler and teacher.
- **A trained world model**, which learns the dynamics of what it observes, including the frozen LLM's behavior, and uses them to decide which parts of the view to segment, at what level of detail, and whether System 1 or System 2 acts.

Design goals:

- **Speed:** most steps run as one forward pass.
- **Reliability:** System 1 only sees targets the body verified exist.
- **Generality:** one brain across websites, desktop apps, games and simulated embodiment.
- **Adapter robustness:** the world model is trained across many adapters, so new ones (eventually written by the LLM) are just another case.
- **No forgetting:** adding a body never changes brain weights.

## Scope of this phase

| In scope now | Deferred to the next phase |
| --- | --- |
| Train the world model only | Train the LLM, specifically on tasks that teach it to create adapters |
| Frozen Qwen3.5-9B for all LLM roles | Teacher curriculum, mastery lists, step-down rules |
| Hand-written adapters (the LLM may draft them) | Skill promotion, body writing and repair tasks |
| Sandboxed web and VM environments | Games, VR and simulated robotics bodies |

The phase succeeds if the world model, trained across many adapters, performs well through an adapter it has never seen.

## Components

| Component | Role | Trained now? |
| --- | --- | --- |
| Body contract | Built once. Targets, `observe` and `act`, outcomes, levels and directions, ID tracking, plus one sub-contract per decision model (navigation, action) | No |
| Adapters | Thin translators between an environment's native interface and the contract (DOM, accessibility tree, vision). Also list the environment's own movement inputs | No (hand-written, LLM-writable later) |
| Protected core | Completion checks, latency measurement, executor limits, conformance suite. Read-only to the LLM | No |
| World model | R2-Dreamer latent dynamics on VLM features. Policy outputs route, level and cells | **Yes** |
| VLM (Qwen3.5-9B) | System 1, System 2, labeler, teacher, and the world model's feature source | No (frozen) |
| Goal context | Per-goal LLM context holding the goal, task list and current subtask | No (data) |
| Log | Structured history of every observation, decision and outcome | No (data) |

Key terms:

- **Target:** anything the body can move to, with a stable ID and the verbs that can be applied to it. See the contract.
- **Navigation:** getting to a target. **Action:** applying a verb to the target you are at. Each is its own decision model with its own sub-contract, and they run in parallel.
- **Pointer mode:** *absolute* environments (web, desktop, VM) move the pointer straight to a chosen target; *relative* environments (games, VR, robots) move through their own movement inputs, which the navigation model picks directly.
- **Anchor:** where "self" is in the view (screen center in first person, cursor on desktop and web, character in top-down games).
- **Level:** how coarse the world model's selection is. Level 1 is directions around the anchor, level 3 is specific targets.
- **Important target:** a target on any valid path from the current screen to the goal, as judged by the labeler.

## One step

1. The **adapter** reads the environment and the **contract** builds the frame, targets and direction targets.
2. The **VLM vision tower** produces patch features, pooled per grid cell. The **goal context** is extended with the last outcome, and the VLM's hidden state at that point becomes the text feature.
3. The **world model** updates its latent. Its policy outputs **route**, **level** and **cells**.
4. The **contract** returns only the targets inside the selected cells, at the chosen level.
5. **If routed to System 1:** the navigation and action questions run in one batched pass from the cached goal context. The *navigation* question(s) pick the next target or movement input, the *action* question picks a verb for the current target. A hard rule escalates to System 2 if System 1's probabilities are spread out. **If routed to System 2:** it sees the image and the log, and picks, invents, or writes a new task list.
6. The **contract** acts. The action verb runs through the environment's own function on the current target, and navigation runs in the same step: the pointer moves to the chosen target (absolute mode) or the chosen movement inputs are sent (relative mode).
7. The **core** checks the subtask condition and measures latency, the **rewards** are computed, and the **log** records the step.

Hard rule: with no task list, the step always goes to System 2 to plan first.

## The body

The body is the only environment-specific part. It is split into a contract built once and a thin adapter per environment. Full definitions are in [body/CONTRACT.md](../body/CONTRACT.md).

### Built once, shared by every environment

- `Target`, `Outcome` and `Capabilities` schemas.
- One sub-contract per decision model: the **navigation contract** (its options and how a choice is executed) and the **action contract** (its options and how a verb is executed).
- `observe(level, cells)` and `act(navigation, action, arg)`.
- Window-normalized coordinates, so cells, boxes and lines mean the same thing everywhere.
- Generated targets: directions around the anchor, positional targets for empty space, and edge targets for off-screen exploration.
- Target ID tracking across frames and across sources, revalidation before acting, and merging of several adapters.

### Per environment, kept small

- **Adapter (the only per-environment piece):** translates native elements to targets, lists the environment's own movement inputs (relative mode only), and passes interaction straight through to the environment's own functions (a DOM click, an accessibility invoke, a game input). It never reimplements interaction and never filters by relevance.

There is no navigation translator. Navigation is a decision, not code: in absolute mode the navigation model picks a target and the contract moves the pointer there; in relative mode it picks among the environment's movement inputs directly, as the Doom demo's `move`, `turn` and `strafe` heads do.

### Two decision models in parallel

Navigation and action are separate decision models, each restricted to its own sub-contract's options, answered in parallel in one System 1 pass.

- The action applies to the target you are at; navigation chooses where to go next (or how to move). Both execute in the same step, so moving and acting can happen at once (strafe while firing, walk while turning).
- Navigation can be several parallel questions when the environment has several movement axes (move and turn). In relative mode it also keeps a destination target, and every target carries its bearing and distance from the anchor, so the movement questions know which way to go.
- Each decision model gets its own outcome (navigation: reached, moving, blocked, held; action: done, failed, partial, in progress, none), so failures are attributed to the right one.
- They are aware of each other through the shared goal context and the "currently at" field, without an extra forward pass.
- Global actions (fire, jump, keyboard shortcuts) are verbs on a built-in `self` target.
- Each increment of movement is one System 1 decision (one pass), which matches the Doom demo's decision rate.

### Adapter set for this phase

| Adapter | Source | Used on |
| --- | --- | --- |
| DOM | Playwright | MiniWoB++, WebArena |
| Accessibility tree | Browser or OS accessibility API | MiniWoB++, WebArena, OSWorld VM |
| Vision only | OmniParser (marked low confidence) | All, **held out on OSWorld** |
| Degraded variants | Any of the above with dropped targets, shifted boxes or missing labels | All |

## World model

### Role

The world model learns dynamics, predicting how the view and the available targets change after actions, including how the frozen LLM will behave. It makes three choices per step and nothing else: **route, level, cells**.

It is trained in two phases:

1. **Supervised, offline (this phase).** Two heads read the latent. The **cells head** predicts, per grid cell, whether an important target is there. The **escalation head** predicts whether System 1 will be wrong on this step. At run time, route is a threshold on the escalation head and cells are the highest-scoring cells above a threshold, within a budget. The thresholds and budget are tuned on held-out episodes.
2. **RL fine-tuning, online (after the supervised heads work).** The policy trains on imagined rollouts with the two reward terms, starting from the supervised world model rather than from scratch.

### Inputs

| Input | Source |
| --- | --- |
| Vision | Frozen VLM vision tower patch features, pooled per grid cell. Replaces the Dreamer CNN and removes the resolution problem |
| Text | VLM hidden state (middle-to-late layer) after the last outcome in the goal context. Replaces BERT |
| Vectors | Last step's System 1 probability spread, target summary (count, mean confidence), last action |
| Not given | Adapter identity, so the model cannot become a per-adapter specialist |

### Model

R2-Dreamer (decoder-free, PyTorch) from `dreamer v3/`, with the image encoder replaced by an MLP over the VLM features. Decoder-free suits this, since it never reconstructs pixels.

### Action space

One flat multi-discrete action per step:

```
[ route: S1 | S2 ]  [ level: 1 | 2 | 3 ]  [ grid cells: 0/1 each ]
```

| Level | Grid | Body returns | Use |
| --- | --- | --- | --- |
| 1 | 3×3 around the anchor | Directions (in relative mode, the environment's movement inputs) | "Go left", side-scrollers, exploring |
| 2 | 6×6 over the window | Regions and their targets | Moving toward an area |
| 3 | 12×12 over the window | Specific targets | Clicking a specific element |

Masking rules:

- Cells of levels other than the chosen one are masked.
- A coarse cell set to 0 masks its finer children.
- Routing never masks segmentation, since System 1 and System 2 both receive the same targets.
- Masked dimensions are removed from the actor's log-probability and entropy terms.

Start with levels 1 and 3 only. The route is chosen at the same moment as segmentation, so the router predicts System 1's confidence rather than seeing it. The hard escalation rule covers its mistakes.

## LLM: System 1 and System 2

### Model

**Qwen3.5-9B** (native vision, Apache 2.0). The repository is cloned in `models/Qwen3.5-9B/` without weights; weights are added once the full system is built. One frozen copy serves System 1, System 2, the labeler and the teacher.

### System 1

- The `sgoedecke/system-one` method: chat template, prefilled answer prefix, logits constrained to option labels, one forward pass.
- Two decision models per step, batched: **navigation** (which target or movement input next, possibly several questions for several movement axes) and **action** (which verb on the current target).
- Option limits: about 10 with numeric indexes, 100 with two-letter labels. The world model's segmentation keeps option sets inside this range.
- Output: the choice plus a full probability distribution, used for the escalation rule and as a world model input.

### System 2

- Inputs: the image, the targets, and the log.
- Outputs: a task list (each item with a checkable completion condition and any free-form arguments, such as text to type), a pick from the targets, or an invented action.
- Replans from the log, not from the goal context.

### Goal context

One LLM context per goal, ordered so the stable parts can be cached across steps:

```
goal → task list → current subtask → last outcome │ state + targets → System 1 questions
└──────── cached across steps ───────┘ └─ added ─┘ │
                                       world model reads the text feature here
```

- The current subtask is swapped when its condition passes. Only the cache from the subtask onward is recomputed.
- The whole context is wiped when the goal completes.
- System 2 replans from the log, so wiping loses nothing.

### Required changes to System One (verify once weights are added)

1. Keep the prefix cache across steps, not just across questions in one call.
2. Load a natively multimodal model through its processor and image-text model class.
3. Check that Qwen3.5's hybrid attention (Gated DeltaNet recurrent state) works with `cache_prefix`, left padding, and cross-step caching.
4. Match the chat template's thinking toggle to the flag System One passes.

## Collected data and hindsight labels

The expensive LLM calls happen once. Episodes are collected with the stage-1 segmentation rule (every cell that holds a target, so every target is offered) and with **both System 1 and System 2 answering every step**. One of them is executed (System 2 by default, for the most successes). Each step stores the world model's inputs, the decision taken, every target's position, and both systems' answers.

Labels come from the environment's verified success, not from an extra LLM judgment, so they cannot reward System 1 for agreeing with itself:

| Label | Source |
| --- | --- |
| Important cells | In episodes the environment marked successful, the cells (at every active level) holding targets the agent actually reached or acted on |
| System 1 wrong | System 1's answer differs from System 2's on the same step (or, optionally, is not on the successful path) |

Unsuccessful episodes give no cell labels. Tasks no episode solved give no segmentation signal until System 2 solves them. Episodes collected with held-out adapters are kept for evaluation only.

## Labeler (optional)

Hindsight labels are the default. The LLM labeler remains available for scoring and for the RL stage, but its verdicts must first be checked against the environment's real success on a few hundred episodes, since it is the same model as System 1.

The labeler is the frozen VLM with privileged access to the environment's source through the contract's `describe_source()` (DOM, accessibility tree, or code). It is used only during training. For each screen and subtask it produces:

- the list of targets that really exist;
- the **important** targets, meaning those on any valid path from the current screen to the goal;
- the **level each important target needs**, either "a direction is enough" or "needs the specific element".

Labels are cached per screen hash and subtask, so each screen is labeled once. The body cross-checks labels against what is actually visible and reachable before they count.

## Rewards (RL fine-tuning stage)

Two separate terms, each training its own part of the action. They are used after the supervised heads work.

### Segmentation reward (trains level and cells)

$$r^{\text{seg}} = \beta \cdot \text{recall}_{\text{important}} - c \cdot n_{\text{unimportant}} - \epsilon \cdot n_{\text{options}}, \quad \beta \gg c > \epsilon$$

- An important target counts as covered only if selected at its needed level or finer.
- Counts are of options the body returns after resolving cells, not raw cells.
- Coarse selections are cheap (a direction is one option), so the model learns to be as coarse as possible and as fine as necessary.

### Decision reward (trains route)

$$s = \min\left(1, \frac{L_{\text{expected}}}{L}\right)$$

$$r^{\text{dec}} = \begin{cases} a + b\,s - \kappa\,\mathbb{1}[\text{S2}] & \text{if correct} \\ -\left(p_{\text{irr}} + q\,(1 - s)\right) - \kappa\,\mathbb{1}[\text{S2}] & \text{if wrong} \end{cases}$$

| Outcome | Reward |
| --- | --- |
| Fast, correct | a + b (b much larger than a) |
| Slow, correct | about a |
| Fast, wrong | −p |
| Slow, wrong | −(p + q), worst |

- $\kappa$ is a small cost for using System 2. $p$ and $q$ are set so escalating under real uncertainty beats guessing.
- $p_{\text{irr}}$ is raised for steps tagged irreversible.
- Latency is measured by the core and normalized per environment type.
- The initial planning call is excluded from $r^{\text{dec}}$.

### Where correctness comes from

| Check | Judges | Feeds |
| --- | --- | --- |
| Labeler, every step | Was the chosen target important at the needed level, with a fitting verb? | $r^{\text{dec}}$ |
| Subtask condition, written by System 2, checked by the core | Is the subtask done? | Advancing the goal context |
| Environment checker (WebArena, OSWorld) | Is the whole task done? | Final success and stage gates |

The same cached label serves both $r^{\text{seg}}$ and the per-step verdict, so no extra LLM call is needed.

### Training wiring

- Two reward heads and two critics, each with its own return normalization.
- $r^{\text{seg}}$ advantages train only the level and cell dimensions. $r^{\text{dec}}$ advantages train only the route dimension. This stops segmentation from colluding with routing, for example by dropping options to make System 1 look confident.
- For logging only, $r = w_s r^{\text{seg}} + w_d r^{\text{dec}}$.

## Environments and adapter training

Sandboxed only, so every environment can be reset, acted in safely, and checked.

| Environment | What it is | Adapters |
| --- | --- | --- |
| MiniWoB++ | Hundreds of small web tasks | DOM, accessibility, vision, degraded |
| WebArena / VisualWebArena | Self-hosted realistic sites with programmatic task checks | DOM, accessibility, vision, degraded |
| OSWorld | Ubuntu or Windows VMs with evaluation scripts and snapshot reset | Accessibility, vision (**held out**) |

Training rules:

- The same task is served by different adapters, so the world model depends on the contract, not on any adapter's quirks.
- Degraded adapters prepare it for imperfect LLM-written adapters later.
- Adapter identity is never an input.
- One adapter is held out entirely for evaluation.

## Evaluation

- **Held-out adapter:** performance through an unseen adapter compared with training adapters. A large gap means per-adapter specialists.
- **Task success** from environment checkers.
- **System 1 share** of steps and **step latency**, the main speed metrics.
- **Segmentation recall** of important targets and **level usage**.
- **Routing calibration:** whether escalations happen where System 1 would have been wrong.

## Safety

- The protected core (completion checks, latency measurement, executor limits, conformance suite) is read-only to the LLM.
- All training runs in sandboxed sites or VMs.
- Irreversible actions are flagged before execution and carry a higher wrong-answer penalty.
- Adapters translate only. They never measure their own latency, judge completion, or filter targets by relevance.

## Risks and open questions

| Risk | Mitigation |
| --- | --- |
| World model does not generalize across adapters | Many adapters per environment, degraded variants, no adapter ID, held-out evaluation |
| Labels reward System 1 for agreeing with itself | Hindsight labels from verified environment success by default; the LLM labeler is optional and checked against success first |
| Credit assignment over many cells | One supervised label per cell instead of one scalar reward for all cells; RL only fine-tunes afterwards |
| Cost of LLM calls during training | Collect once, train offline as often as needed |
| Unsolved tasks give no cell labels | Collect with System 2 executing; extend the dataset as more tasks are solved |
| Dynamics may not matter on short MiniWoB tasks | Compare the world model heads with a plain classifier on the same data |
| Qwen3.5 incompatible with System One caching or padding | Verify first once weights are added; fall back to no `cache_prefix` or Qwen3-VL-8B |
| Latency stacking (vision pass, world model, System 1, body) | Measure in stage 1 before setting expected latencies |
| Segmentation and routing collude | Separate reward heads, critics and action slices |
| Compute fit (9B VLM, world model, browser or VM on one GPU) | Measure at stage 1; quantize if needed |

Open values to set:

- Reward constants $a, b, p, q, \kappa, \beta, c, \epsilon$ and the expected latencies.
- Grid sizes per level, and the hidden-state layer used as the text feature.
- The System 1 spread threshold for the escalation rule.
- The run-time escalation threshold, cell threshold and cell budget (suggested by the held-out sweep).
- How many degraded adapter variants to use.

## Staged build plan (this phase)

Each stage is usable on its own and gated on a measurable result.

0. **Contract, protected core, log**, with the DOM adapter on MiniWoB++. *Gate:* the adapter passes the conformance suite. (`scripts/check_adapters.py`)
1. **System 1, goal context, hard escalation rule**, no training. *Gate:* task success, System 1 share and seconds per step, measured against System 2 alone. (`scripts/run_stage1.py`)
2. **Collect** episodes on MiniWoB++ across adapters (DOM, accessibility, degraded), with both systems answering every step, plus a held-out-adapter set. *Gate:* enough successful episodes for cell labels, and a measured System 1 error rate. (`scripts/collect.py`)
3. **Offline world model with supervised heads.** *Gate:* on held-out adapters, the cells head covers ≥95% of important targets with few cells, and the escalation head catches ≥90% of System 1 errors with little escalation; it also beats a plain classifier on the same data. (`scripts/train_offline.py`)
4. **Online with the trained heads.** *Gate:* matches the stage 1 rules on success with fewer options shown and less System 2 use, including on the held-out adapter. (`scripts/run_stage1.py --policy heads`)
5. **RL fine-tuning** in imagination with the two rewards, starting from the stage 3 model. *Gate:* improves on stage 4 without losing held-out performance. (`scripts/train_wm.py --init-from`)
6. **WebArena and the OSWorld VM** (needs a cloud VM next to Colab). *Gate:* held-out performance close to training adapters.

Next phase: train the LLM to create adapters, then teacher curriculum, mastery lists, skill promotion and new body types (games, VR, simulated robotics), as described in v1.

## Project layout

```
System Router/
├── docs/              this spec and v1
├── body/              the contract (CONTRACT.md, schema, grid, core, log, conformance, adapters)
├── brain/             frozen-LLM roles (System 1, System 2, goal context, labeler, features, one step)
├── wm/                the trained part (action codec, rewards, stage-1 rules, gym wrapper, two-head agent)
├── environments/      MiniWoB++ (WebArena and OSWorld later)
├── scripts/           check_adapters, run_stage1, train_wm
├── configs/           default.yaml
├── notebooks/         colab.ipynb
├── system one/        submodule: sgoedecke/system-one, the System 1 mechanism
├── dreamer v3/        submodule: NM512/r2dreamer, the world model (plain DreamerV3 via model.rep_loss=dreamer)
└── models/            not versioned; weights added later
```

## Prior work this builds on

In addition to v1's sources:

- [NM512/r2dreamer](https://github.com/NM512/r2dreamer): decoder-free world model with a fast PyTorch DreamerV3 reproduction.
- [Microsoft OmniParser](https://learnopencv.com/omniparser-vision-based-gui-agent/): vision-only detection of interactable screen elements.
- [xa11y](https://pypi.org/project/xa11y/): one accessibility-tree API across Windows, macOS and Linux.
- MiniWoB++, WebArena and OSWorld: sandboxed web and VM environments with programmatic task checks (from memory, not re-checked).
- [Qwen3.5](https://qudata.com/en/news/exploring-qwen35-family/): natively multimodal open-weight models.
- [Dreamer 4 (Hafner, Yan, Lillicrap)](https://arxiv.org/abs/2509.24527): world models on tokenized observations and agents trained in imagination.
