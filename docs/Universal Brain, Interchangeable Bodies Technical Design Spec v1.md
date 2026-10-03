# Universal Brain, Interchangeable Bodies: Technical Design Spec

Oct 2, 2026 · @Mawunge Teye

## Overview and core principle

One universal brain operates any environment through interchangeable bodies that all follow the same contract. New environments are absorbed by writing a new body, never by retraining the brain.

The brain is a trained world model (WM RL) plus one LLM used in two sampling modes: System 1, a single forward pass that picks from a verified action set, and System 2, full reasoning that plans, picks or invents actions. The world model decides where to look, whether a step is complete, and which mode acts next.

The body (crawler) is code. It senses the environment, resolves what is in the areas the world model points at, executes semantic actions with its own controller, and reports outcomes. Because every body speaks the same contract, the action space and decision process are identical in every environment.

Design goals:

- **Speed:** most steps run as one forward pass, so actions take a fraction of a second instead of seconds.
- **Reliability:** System 1 only ever sees options the body verified exist, so it cannot act on hallucinated elements.
- **Generality:** websites, desktop apps, games, VR and simulated robotics share one brain.
- **Self-improvement:** a frozen teacher LLM generates an unlimited curriculum, and the LLM can write and repair its own bodies under a protected core.
- **No forgetting:** adding a body never changes brain weights, so existing skills are untouched.

## Component summary and glossary

Seven components make up the system; only the world model and, slowly, the student LLM are trained.

| Component | Role | Trained? |
| --- | --- | --- |
| Body (crawler) | Senses, resolves cells, builds action sets, executes, reports outcomes | No (code, LLM can extend its periphery) |
| Protected core | Completion verification, reward and latency measurement, safety limits, executor permissions, conformance suite | No (read-only to the LLM) |
| World model RL | Encodes text + image into a latent, outputs the occupancy grid, checks completion, routes | Yes, fast |
| Text encoder | Pretrained BERT-like model reading the subtask and outcome report | Yes, fine-tuned slowly |
| Student LLM | Runs System 1 and System 2; writes bodies and skills | Yes, very slowly, anchored to the teacher |
| Teacher LLM | Generates tasks, subgoals, success conditions; decomposes failures; reviews body changes | No (frozen, unrewarded) |
| Memory | Two-way lookup table, history cache, skill library, mastery lists | No (data) |

Key terms:

- **Occupancy grid:** a multi-scale binary grid over the screen or view. A 1 marks a cell the world model wants resolved; choosing coarseness is itself an action.
- **Action set:** the clean, deduplicated, described list of options the body builds from the nonzero cells.
- **Subtask:** one item of the task list System 2 writes; System 1 always acts against the current subtask.
- **Outcome report:** the body's description of what an action did, in the shared vocabulary.
- **Body contract:** the four obligations every body meets (observe, act, report, timing), defined in the next sections.
- **Mastery list:** grouped one-sentence summaries of skills the student consistently succeeds at.

## Architecture overview

The system is five layers: the environment, the world model, the sensing half of the body, the LLM, and the acting half of the body. Putting the brain between the two halves of the body makes the split visible: the body handles everything environment-specific, the brain handles everything else.

&#91;embedded content: system architecture · 5 layers, router decides who acts\]

The router (highlighted) is the only decision point about who acts; System 2 hands its task list to System 1, and both outputs pass through the lookup table before the executor acts, with the protected core verifying the result.

## The body (crawler) and its contract

The body is the only environment-specific part of the system: if a body honors the contract, the brain works in that environment unchanged. Think of it as a device driver: the brain is the operating system, each body a driver for one kind of world.

### The contract

| Obligation | What the body must do |
| --- | --- |
| Observe | For every nonzero grid cell, report typed entities in the shared vocabulary (pressable, movable, graspable, surface, text, agent), plus positional options for empty or unidentified space |
| Act | Accept semantic actions (press, grab, move to, type, drop) and carry them out with its own controller, interpolating continuous motion between chosen cells |
| Report | Describe each outcome in shared terms (pressed and state changed, grab failed, moved partway, still in progress) |
| Timing | Present time uniformly as steps or ticks, so real-time and turn-based environments look alike |

### Sense organs per environment

- **Websites:** the DOM.
- **Desktop apps:** OS accessibility APIs (UI Automation, AX, AT-SPI).
- **Phones:** Android accessibility layer (iOS is restricted).
- **Games and VR:** engine state where available, removed over training; vision and OCR otherwise.
- **Robotics (simulated):** physics engine state early, then camera and depth perception.
- **Fallback everywhere:** a vision element detector plus OCR, marked low confidence in the lookup table.

### Structure

The body is split into a protected core and an extensible periphery. The core holds verification, reward and latency measurement, safety limits, executor permissions and the conformance suite, and is read-only to the LLM. The periphery holds adapters, skills and action descriptions, and the LLM may modify it (see Safety).

### Conformance

Every new or modified body must pass the protected conformance suite before the world model uses it. The suite checks entity typing, faithful execution of semantic actions, honest outcome reporting, and uniform timing.

## World model RL

The world model is the only fast-trained component and the main determinant of generality: it compresses every environment into one latent state, and an RL policy on that latent decides where to look and who acts.

### Inputs

- **Text stream:** the current subtask and the latest outcome report, fed as a sentence pair (`[CLS] subtask [SEP] outcome [SEP]`) into a pretrained BERT-like encoder. The pair format matches BERT's pretraining on judging how two sentences relate, which is close to the completion question.
- **Image stream:** the screen or view through a vision encoder.
- **Fusion:** the text conditions the image features (cross-attention or FiLM), not just concatenation, so the goal can raise or lower individual grid cells.
- **Text-only mode:** when there is no screen (for example during planning), the world model runs on text alone and can roll out in latent space.

The text encoder is fine-tuned jointly with the world model at a low learning rate, or with most layers frozen, so noisy RL gradients do not erase its language knowledge.

### Latent state

The latent must capture four things, which generalize differently:

| Content | Generalizes across environments |
| --- | --- |
| Where to look, given the goal | Well |
| Task progress and completion | Well (compared in language) |
| Uncertainty | Well |
| Dynamics: what an action will do | Least; most environment-specific |

Auxiliary objectives push the latent toward what matters: predict completion, and predict the body's structured state from pixels (using engine-state labels in games and VR).

### Action space: the occupancy grid

- A multi-scale binary grid over the view. Each cell at each level is 0 or 1.
- Levels are hierarchical: finer cells exist only under coarse cells set to 1, so the policy never faces the full 2^N space. Choosing a coarseness level is itself a discrete action.
- Cells set to 0 are ignored; every cell set to 1 is queried in the body.
- Continuous control (drags, sliders, drawing) is a sequence of fine-cell choices; the body interpolates smooth motion between them.
- The policy otherwise learns when to move forward and when to hold at a place until a subtask completes.
- Grid coordinates are relative to the window or view, not raw pixels, so they transfer across resolutions.
- The mask is trained to err toward keeping cells, since a wrongly zeroed cell is never offered to System 1.

### Routing

- **No active task list:** always System 2 (hard rule; see Training aids).
- **Broad prompt:** learned to route to System 2.
- **Decomposed, specific subtask:** learned to route to System 1.
- **Escalation signals:** spread-out System 1 probabilities, lookup table misses (novel elements or screens), low-confidence fallback entries, and outcomes that did not match expectations.
- **Completion:** judged from the subtask and outcome pair plus the image; on completion the router advances to the next subtask, on stall or drift it hands control to System 2 to replan.

## LLM sampling modes: System 1 and System 2

There is one LLM, used two ways; the difference is the sampling method, not the weights.

### System 1: fast choice

- **Input:** the current subtask and the action set from the body, as a compact prompt with relational descriptions ("on the slider track, about 30% from the left").
- **Method:** the chat template is prefilled with an answer prefix, and logits are constrained to the option labels, so the most likely label is chosen in a single forward pass (the technique in sgoedecke/system-one).
- **Output:** one label plus the full probability distribution over options, which the world model uses as a routing signal.
- **Batching:** the shared prefix is cached and several questions are evaluated in one pass, allowing simultaneous inputs (for example strafe, turn and fire).
- **Large option sets:** handled by the occupancy grid narrowing the set, and by labels rather than numeric indexes.
- **Constraint:** System 1 never runs without a current subtask.

### System 2: deliberate reasoning

- **Input:** the same action set as System 1, plus the raw image and the history cache.
- **Outputs, any of:**
  1. A new or revised task list, each item specific enough for System 1, with free-form arguments (text to type, values) written into the item.
  2. A pick from the existing action set.
  3. An invented action not in the set, which the lookup table registers before the executor runs it.
- **Skill promotion:** an invented action that succeeds consistently is written into the body's periphery as a reusable skill, so it appears in future action sets for System 1.

### Why System 1 transfers across bodies

System 1's job is matching a subtask to an option label, which is the same skill in every body. Because the same LLM writes the subtasks, writes the body's labels and reads both, the match is in its own terms. Accuracy depends on the body being correct and the subtask being clear, not on the environment.

### Model requirements

- System 1 needs full next-token logits (open-weight models; closed APIs often expose only partial log-probabilities).
- System 2 needs vision.
- Training the student requires open weights. Swapping the LLM later needs light router retraining, since the router learns the model's habits.

## Runtime loop

Every step runs the same loop: the world model points, the body resolves, the world model routes, the LLM chooses, the body acts and reports, and the world model checks completion.

&#91;embedded content: runtime loop · one step, two decisions\]

With no active task list the step always goes to System 2 to plan; once a list exists, System 1 handles the step unless its probabilities are spread out or the world model detects novelty, stall or drift, in which case System 2 picks, invents or replans.

## Memory: lookup table, history cache, skill library

Because the LLM is frozen or slow-moving, memory is how the system gets faster on familiar ground: seen screens, elements and skills become cheap.

### Two-way lookup table

- **Outbound:** grid cell, then the body resolves it, then an option label in the action set, with an entry stored.
- **Inbound:** chosen label, then the concrete element and action, then the executor.
- **Keys:** element identity from the body (ID, or role, name and position in the element tree), with screen position as a secondary field, so entries survive scrolling, resizing and layout changes.
- **Revalidation:** before acting on a label, the body confirms the element still exists and means the same thing.
- **Registration:** invented actions are added before execution; low-confidence vision entries are flagged.

### History cache

- Stores screens, resolved entities, decisions and outcome reports as a structured log, readable by any LLM.
- The task description and task list are pinned and never summarized away; the current subtask is marked.
- Older entries are summarized periodically so the cache stays bounded.
- Fed to System 2 on every call; outcome reports also go to the world model through the text encoder.

### Skill library

- Invented actions promoted after consistent success, stored as code in the body's periphery.
- Pruned periodically to remove near-duplicates that would clutter action sets.

### Mastery lists

- One per environment type; each entry is a grouped one-sentence summary of a skill the student consistently succeeds at, including within the expected time.
- Used by the teacher to set harder tasks and as a replay bank to prevent forgetting.

## Training: teacher and student

A frozen teacher LLM generates the curriculum; the student system (world model, text encoder, student LLM and its body) learns from it, with the parts moving at very different speeds.

### Teacher

- A frozen copy of the base LLM. It receives no reward, so it has no incentive to generate easy tasks.
- Given an environment, it generates a task, intermediate goals, and a success condition for each, written in terms the body can verify (a field holds a value, a file exists, a score changed).
- It sees the student's mastery list and recent failures, and decomposes tasks the student keeps failing into lighter sub-tasks.
- It reviews proposed body changes and student-proposed conformance tests.

### Students

| Component | Learning rate | Constraint |
| --- | --- | --- |
| World model RL | Fast | None beyond reward design |
| Text encoder | Low | Most layers may stay frozen |
| Student LLM | Very slow | Kept close to the teacher's output distribution (KL penalty) |

The two-timescale setup is deliberate: the fast world model always sees a nearly fixed LLM partner, which keeps training stable and lets the router track the LLM's slow drift.

### Outer loop (future work)

The student LLM's slow updates are rewarded by improvement in the world model's reward, measured over a sliding window. This is what can teach behaviors the inner loop never discovers, such as inventing actions or asking how to do something. Any "ask" action must target sources available at deployment (documentation, the web, the user), never the teacher.

### Self-generated data

Training data is limited only by available environments: the teacher proposes, the student attempts, the body verifies. No human labeling is required.

## Environment types and curriculum

Training spans five environment types, sampled equally by training updates, with rewards normalized within each type so no type dominates.

| Type | What it trains | Body source |
| --- | --- | --- |
| Website tasks | Structured interfaces, forms, navigation | DOM |
| Desktop application tasks | Native apps, mixed accessibility quality | OS accessibility APIs, vision fallback |
| Gaming tasks | Real-time control, fast reactions | Engine state, then vision |
| VR and simulated embodiment | 3D navigation and manipulation (grasp, push, place), physics simulators for robotics | Engine state removed over training, then vision and depth |
| Body-building tasks | Writing, repairing and testing bodies | Code sandbox, ends with a short evaluation in the target environment |

### Curriculum rules

1. **Start from primitives.** Each body begins with its basic actions: move the cursor or press any button; look around or reach; move one joint ("wave your hand").
2. **Retry with variations until mastered.** Variations change substance (site, layout, values, starting state), not just wording.
3. **Mastery includes speed.** A task is mastered when the student succeeds on most recent variations within the expected time, which in practice means it runs mostly in System 1.
4. **Store only mastery.** Mastered tasks are grouped into one-sentence summaries in a per-type mastery list.
5. **Raise difficulty relative to mastery.** The teacher generates harder tasks from the mastery list.
6. **Step down on high failure.** When failure rate passes a threshold, the teacher decomposes the task into lighter sub-tasks. Stepping back up requires clearly better performance than the step-down threshold, to avoid oscillation.
7. **Use failure type.** Decision failures get simpler sub-tasks; perception failures (the body misreported) spawn body-repair tasks.
8. **Replay.** A sample of mastered groups is re-tested periodically; failing groups return to retry.

### Required goal categories

Alongside ordinary tasks, training goals always include: writing and repairing bodies (including deliberately broken ones), tasks in environments new to the training pool, and conformance-testing tasks.

Equal sampling is the starting point; once a type is largely mastered, sampling can shift toward types where the student is still improving.

## Reward design

Each stage is rewarded on correctness and speed, ordered: fast and correct, then slow and correct, then fast and wrong, with slow and wrong worst.

A plain product of 1/latency and a plus or minus 1 correctness sign gets the wrong-answer ordering backwards (fast and wrong would be penalized most), so the speed term is applied differently for right and wrong outcomes:

```latex
s = \min\left(1, \frac{L_{\text{expected}}}{L}\right)
```

```latex
r = \begin{cases} a + b\,s & \text{if correct} \\ -\left(p + q\,(1 - s)\right) & \text{if wrong} \end{cases}
```

| Outcome | Reward |
| --- | --- |
| Fast, correct | a + b (heavy, with b much larger than a) |
| Slow, correct | about a (slight) |
| Fast, wrong | -p |
| Slow, wrong | -(p + q) (worst) |

### Rules that keep the reward honest

- **Latency is normalized** against an expected latency per environment type and stage, so naturally slow environments are not penalized for being slow.
- **Escalation must pay.** Because slow and wrong is worse than fast and wrong, p and q are set so that routing to System 2 under real uncertainty still has the better expected reward than guessing.
- **Irreversible steps cost more.** The teacher tags steps as reversible or not; for irreversible ones (deleting, purchasing) the wrong-answer penalty is raised so a careful slow action beats a fast wrong one.
- **Planning is setup time.** The initial System 2 planning call is excluded from the per-stage latency term and gets its own reward, scored by how well the plan's later steps go.
- **Completion is verified,** never self-declared: correctness comes from the protected core's checks against body-observable success conditions.
- **System 2 cost is a secondary penalty** on top of verified task success, so the router learns to use System 2 only when it pays.

## Training aids

Three aids shortcut slow RL discovery: randomly enforced planning, privileged engine state, and body tasks.

### Randomly enforced plan-first and replanning

- In a share of training episodes, the rule "no active task list, route to System 2 first" is enforced; in the rest, the router decides.
- Enforced episodes still teach the world model how plan-first trajectories unfold, and the router compares planning versus acting blind in imagined rollouts, where it makes its own choices.
- Enforcement starts high and is reduced toward zero over training. If the router skips planning once unforced, the reward still favors fast guessing and needs adjusting.
- Whether a rule was enforced is recorded in training data but never shown to the router as an input.
- The same trick is applied mid-task: replanning is randomly forced at different points, teaching when replanning helps.
- At deployment the plan-first rule stays as a hard rule regardless.

### Privileged engine state (games, VR, simulation)

- Early in training, the body reads object positions, types and physics directly from the engine, acting as a perfect harness.
- Engine state is removed gradually, so the body and world model learn to rely on vision and depth, the way a robot must.
- After removal, engine state still serves as free labels for training perception, and as ground truth for completion checks inside the protected core.
- Visual variation (lighting, textures, noise) is randomized so perception does not depend on game-specific looks.

### Body tasks

- **Writing bodies:** the student writes adapters for environments the body cannot read well; System 2 cross-checks the adapter against the screenshot, and only agreeing adapters are kept.
- **Repairing bodies:** the teacher deliberately breaks working bodies (removes an entity type, corrupts an outcome report, distorts timing) and the student must find and fix the fault, giving unlimited repair practice with known answers.
- **Failure-driven repair:** when a task fails because the body misperceived or misexecuted, that failure spawns a body-repair task.
- **Conformance testing:** the student may propose new test cases; accepted ones go into an extension of the suite after teacher review, never into the core.
- Reward for body tasks comes from passing the protected conformance suite and from downstream improvement in task performance.

## Generalization and evaluation

With correct bodies and honest rewards, generalization comes down to the world model; held-out environments are the only honest measure of it.

### The generalization claim

Because the action space and decision process are fixed, and each body is written by the same model in its own terms, a new body requires no change to the brain. Decision quality carries over to the extent the body passes conformance and its situations resemble those seen in training.

### What is invariant across bodies

- The occupancy grid action space and its hierarchy.
- The decision procedure: the same weights map "these options plus this context" to a choice.
- The language space for goals, progress and completion (pretrained text encoder and LLM).
- The body contract: observation types, semantic actions, outcome reports, timing.

### What is not invariant

- The distribution of situations: new entities, slower or less certain outcomes, different rhythms.
- Dynamics: what an action does differs most between, for example, a click and a physical grasp.
- The actions taken: these should differ per environment; the target is invariant decision quality, not invariant actions.

### Degradation

Rich-structure environments run mostly in System 1. Poor-structure ones shift more steps to System 2, which sees the raw image, so the system gets slower rather than failing. Expected order from fastest to slowest: websites, accessible native apps, web-based desktop apps, poorly labeled apps, pixel-only environments.

### Evaluation plan

- **Held-out pool per environment type:** environments never used in training, evaluated with no training and with a few adaptation episodes.
- **Hidden-specialist check:** compare held-out performance to training performance; a large gap means the world model learned per-environment specialists rather than a general latent.
- **System 1 transfer test:** write a new body, run System 1 on its tasks, compare accuracy with trained bodies.
- **Routing share:** fraction of steps handled by System 1, per environment type, as the main speed metric.
- **Calibration:** whether System 1 confidence and world model uncertainty predict actual errors.
- **Forgetting:** replay of mastered groups across environment types after new bodies are added.

## Safety: protected core and self-modification

The body both acts and keeps score, so a system that can edit its body can learn to edit its scorekeeping; the protected core exists to make that impossible.

### Reward tampering

If the LLM could modify completion checks or latency measurement, the cheapest way to raise reward would be to make the body report success or shorter times. An optimizing system will find this if the door is open.

### Core versus periphery

| Protected core (read-only to the LLM) | Extensible periphery (LLM may modify) |
| --- | --- |
| Completion verification | Adapters (sense organs) |
| Reward and latency measurement | Skills (promoted invented actions) |
| Safety limits | Action descriptions and labels |
| Executor permissions | Proposed extra conformance tests (after review) |
| Core conformance suite |  |

### Change control for the periphery

1. Every change runs first in a sandbox.
2. It must pass the core conformance suite.
3. The frozen teacher reviews it; the teacher has no reward and no stake in the outcome.
4. Changes are versioned and can be rolled back.
5. Changes are introduced gradually, so the world model can adapt to what it now observes.
6. All changes stay inspectable by a human.

### Environment safety

- All training runs inside virtual machines or simulators, so a bad task or action cannot do real damage.
- Teacher-generated tasks are filtered before execution; tasks the body cannot verify are rejected.
- Executor permissions bound what any action can do on the machine, regardless of what the LLM proposes.

## Risks and open questions

The largest unknown is whether one world model's latent dynamics generalize across very different worlds; most other risks are about keeping training honest and debuggable.

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| World model generalization | It is the only fast-trained part and carries everything | Structured inputs from the body, language goals, diverse environments, held-out evaluation |
| Hidden specialists | The world model may recognize each environment and act as a per-environment specialist | Held-out pool; compare held-out with training performance |
| Reward exploits | Self-improving systems find reward loopholes (fast guessing already found) | Verified completion, escalation must pay, irreversibility weighting, monitoring router behavior |
| Premature completion | Cheapest path is declaring subtasks done early | Completion only from protected-core checks |
| Shared misunderstanding | The same LLM writes and reads labels, so a mislabel is acted on confidently | Conformance suite, System 2 cross-checks against the image |
| Debuggability | Many parts learn at once; failures are hard to attribute | Staged build plan; freeze all but one learner when diagnosing |
| Body engineering cost | Cross-platform adapters are large engineering work | Contract first; LLM-written adapters with verification |
| Compute cost | VR, physics and many bodies are expensive | Cheap environments carry early training; equal sampling by updates |
| Interference between types | Learning one type can hurt another in a shared model | Shared core with small type-specific parts if measured |
| Sim-to-real gap | Game physics and visuals differ from reality | Physics simulators, visual randomization, real-robot fine-tuning for the executor |
| Self-modification | A system editing its own body can drift or tamper | Protected core, sandbox, review, versioning, human inspection |

### Open questions

- One world model for all types, or a shared core with type-specific parts?
- Exact fusion method for text and image (cross-attention or FiLM)?
- Grid resolution levels and the coarse-to-fine schedule?
- Thresholds for mastery, step-down and step-up?
- Values of a, b, p and q, and the expected-latency baselines per type?
- How fast to phase out enforced planning and engine state?
- What the student's "ask" action targets at deployment?

## Staged build plan

Build in eight stages (0 to 7), each useful on its own and each gated on a measurable result, so progress is real even if a later stage stalls.

0. **Body contract.** Specify observation types, semantic actions, outcome reports, timing, and the core conformance suite. *Gate:* a website body and a desktop body both pass the suite.
1. **Web only, no training.** DOM body, System 1 via constrained choice, hard plan-first rule, escalation when System 1 probabilities are spread out. *Gate:* task success and System 1 share measured against a reasoning-only agent on the same tasks.
2. **Trained world model router.** Replace hand-written routing with the world model and its text and image inputs. *Gate:* beats the stage 1 rules on success and latency.
3. **Occupancy grid and continuous actions.** Multi-scale binary grid, body-side interpolation, positional options. *Gate:* drags, sliders and drawing tasks succeed at usable speed.
4. **Teacher curriculum.** Task generation, mastery lists, step-down decomposition, randomly enforced planning. *Gate:* mastery list grows steadily without oscillation; held-out website tasks improve.
5. **New body types.** Desktop apps, games, VR and simulated embodiment, with engine state phased out. *Gate:* held-out performance per type; hidden-specialist check passes.
6. **Body writing and repair.** LLM-written adapters, fault-injected repair, skill promotion, all under the protected core. *Gate:* new adapters pass conformance and match System 2 cross-checks.
7. **Slow student fine-tuning and outer loop.** Unfreeze the student LLM with a KL anchor; reward by world model reward improvement. *Gate:* no regression on held-out pools; new behaviors such as asking or inventing emerge.

The ordering keeps one new learner or one new difficulty per stage, which is what makes failures attributable.

## Prior work this builds on

Most components exist individually; the original parts are the occupancy grid as a universal action space that also drives what the body is queried for, and an LLM maintaining its own bodies under a fixed contract.

Sources opened while developing this design:

- [sgoedecke/system-one](https://github.com/sgoedecke/system-one): single-forward-pass constrained choice for open LLMs; the System 1 mechanism. Its Doom demo was 3.5x faster between actions than tool calling (172 ms versus 600 ms median, Qwen3-8B).
- [Introducing System One Models & Jev (TypeSafe AI)](https://typesafe.ai/blog/introducing-system-one-models-and-jev): the System One framing, typed probabilistic decisions, and calibration as a training target.
- [Distilling System 2 into System 1 (Yu et al., Meta FAIR)](https://arxiv.org/html/2407.06023v3): reasoning distills well for judgment-style tasks, poorly for multi-step math.
- [From Explicit CoT to Implicit CoT (Deng, Choi, Shieber)](https://arxiv.org/html/2405.14838v1): gradually removing reasoning steps; a speed-accuracy dial.

Referenced from memory, not re-checked for this document (approximate):

- Voyager (2023): an LLM agent in Minecraft that writes successful behaviors into a reusable code skill library; precedent for skill promotion.
- SayCan (Google, 2022): an LLM choosing among a robot's available skills weighted by feasibility; precedent for choosing from a verified action set.
- Dreamer-style world-model RL: learning a latent dynamics model and training a policy in imagined rollouts.
- Privileged-information training in robotics: a policy trained with full simulator state, then distilled to sensor-only inputs.
