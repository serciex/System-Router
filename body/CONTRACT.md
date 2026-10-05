# Body Contract (v0.3)

The contract is the only interface between the brain and an environment. It is built once and shared by every environment. This document is also the instruction manual for writing adapters, including adapters the model writes itself.

*v0.3 (spec v3): two channels, capability manifest, core actions (think, find, wait), the body slot, surfaces with a shared narrowing library, canonical verbs and roles. The world model's grid levels are gone. v0.2 is kept in [CONTRACT_v0.2.md](CONTRACT_v0.2.md).*

## Principles

1. **Built once.** Prompt layout, questions, core actions, narrowing, IDs, outcomes, timing and checks live in the contract and the protected core, never in adapters.
2. **Two channels.** The sensory channel (what the model sees) is produced by environment integration inside the protected core. The interaction channel (what can be done) is produced by the adapter. An adapter can never alter what the model sees.
3. **Adapters only translate interaction.** They report what is available in a window and execute chosen options through the application's own functions. They never rank, decide, time themselves, or judge completion.
4. **The brain's interface never changes.** Body-specific complexity is solved in adapter code, never by adding questions.

## Sensory channel

Produced by environment integration, read-only to the LLM.

- At least one **screen**, of a declared kind: `flat`, `stereo`, `camera`, `depth`.
- Optional **typed sensors**: `joint_position`, `joint_velocity`, `contact`, `pose` (6DoF), `depth_map`, `force_torque`, `imu`, `scalar` (named, with units), `flag`. New types are added by a person, never by the LLM.
- Encoding in the prompt: scalars and flags as short text lines, joint arrays as compact text, depth maps as an extra image. Only the current screen is an image.

## Capability manifest

| Field | Half | Declared by | Contents |
| --- | --- | --- | --- |
| `body_id`, `version` | Both | Integration | Identity and versioning |
| `screens` | Sensory | Integration | Kind, resolution, field of view, frame rate |
| `sensors` | Sensory | Integration | Name, type, units, rate |
| `pointer_mode` | Interaction | Adapter | `absolute`, `relative` or `none` |
| `actuators` | Interaction | Adapter | `keyboard`, `pointer`, `gamepad`, `head_pose`, `controller`, `hand`, `joint`; limits |
| `verbs` | Interaction | Adapter | Canonical verbs it executes |
| `timing` | Both | Integration | Tick length; `realtime` or `turn_based` |

Privileged engine state is never in the manifest; it lives in the training harness.

## Canonical vocabulary

**Roles:** `button`, `link`, `textbox`, `checkbox`, `radio`, `option`, `menu`, `tab`, `text`, `image`, `icon`, `surface`, `group`, `file`, `symbol`, `entity`, `limb`.

**Verbs:**

| Kind | Verbs |
| --- | --- |
| Pointer and keyboard | `click`, `double_click`, `right_click`, `hover`, `type`, `select`, `scroll_up`, `scroll_down`, `drag`, `press_key` |
| Code workspace | `open`, `read`, `search`, `write`, `run` |
| Embodied | `grab`, `release`, `place`, `use` |

New verbs and roles are added by a person. The contract writes every label as `role: name`; adapters supply the role and a meaningful name ("Submit", not an internal ID).

## Items: what find returns

| Field | Meaning |
| --- | --- |
| `handle` | Adapter's stable handle for the item (the contract assigns the brain-facing ID) |
| `kind` | `element`, `surface` or `group` |
| `role`, `name` | Canonical role and meaningful name |
| `verbs` | Canonical verbs available on it |
| `bbox` | Window-normalized box `(x0, y0, x1, y1)` |
| `container` | Handle of the enclosing group, if any |
| `collapsed` | For groups: number of hidden children (picking the group lets the next find open it) |
| `dims` | For surfaces: 1 or 2 |
| `value` | Current value, for inputs |
| `reversible` | Whether its verbs can be undone; null if unknown |
| `confidence` | 1.0 for structured sources, lower for pixel-inferred items |

Visible items only, grouped by container, large groups collapsed.

## The body slot

- Holds the latest find result: navigation targets (elements, surfaces, groups) and the action verbs available.
- **The core runs find automatically** whenever its change check reports a window change. The model calls find only for a scope (a group or region) or to expand a collapsed group.
- A status line tells the model whether the window changed since the last find.
- Executing a stale option returns `unavailable`.

## Questions and options

| Question | Options |
| --- | --- |
| **Where next?** (navigation) | Body slot targets and surfaces, plus `hold` |
| **What now?** (action) | Verbs of the current target, verbs of `self`, `none`, and the core actions `think`, `find`, `wait` |

- Labels are two-letter codes, never numeric indexes.
- Core actions appear only in the action question and override the navigation answer for that step.
- Execution order: the action applies to the current target, then the body moves to the navigation target.
- Relative-pointer bodies replace positions with movement options: one navigation question per movement axis declared in the manifest, plus a destination.

## Core actions

Provided by the contract and the core, identical in every body.

| Action | Effect |
| --- | --- |
| `think` | Token reasoning: plans, word goals, adapter code. Forced by the core when the top pick's probability is below the current threshold |
| `find` | Refreshes the body slot, optionally scoped |
| `wait` | No inference; the core's change check (pixel difference on watched regions, sensor thresholds, OS or engine events) ends it on a meaningful change or at a time cap |

## Surfaces and the narrowing library

- Adapters declare surfaces (canvas, slider track, map, or the whole window for the fallback) with a box and `dims`.
- Picking a surface opens the **narrowing library**: 9 regions for 2D, 3 for 1D, plus `here`; option text includes the region's coordinates.
- All narrowing picks happen within one step; nothing executes until `here`, which resolves to the region centre.
- `drag` takes two points (from, to); drawing takes a sequence of points.
- The library is part of the contract. Adapters never implement narrowing.
- With several screens, choosing the screen is the coarsest level.

## Outcomes

| Field | Values |
| --- | --- |
| `navigation.status` | `reached`, `moving`, `blocked`, `held` |
| `action.status` | `done`, `failed`, `partial`, `in_progress`, `unavailable`, `none` |
| `text` | One line per part, e.g. "clicked button: Search, results page loaded" |
| `step`, `latency_ms` | From the core, never from the adapter |

Outcome reports go into history only, never into the sensory channel.

## What an adapter provides

| Method | Returns |
| --- | --- |
| `manifest()` | The interaction half of the manifest |
| `find(scope=None)` | Items (above) in the current window, or within the scope |
| `invoke(handle, verb, arg=None)` | Native result, passed through unchanged |
| `act_at(surface, points, verb)` | Native result for a position or point sequence on a surface |
| `move(inputs)` | Relative mode only: one input per declared axis |
| `reset(seed)`, `close()` | Episode lifecycle |

`arg` carries free-form input (text to type, code to write); it comes from a word goal or plan produced by think, never from a pick.

## What the core does for every body

1. Builds the prompt in the fixed layout (instructions, history, screens and sensors, goals, status lines, options, questions).
2. Runs the change check; runs find automatically on window change; drives wait.
3. Assigns brain-facing IDs, keeps them stable across finds, and writes canonical labels.
4. Runs the narrowing library and resolves `here` to points.
5. Revalidates an option before executing it; returns `unavailable` if stale.
6. Measures latency, applies executor limits and irreversibility rules, logs every step.
7. Runs conformance (including the hidden split) and the sandbox for written adapters.

## Adapter rules

An adapter **must**: report only visible items, with canonical roles and verbs and meaningful names; keep handles stable while the window is unchanged; mark pixel-inferred items with confidence below 1.0; execute through the application's own functions; report outcomes honestly.

An adapter **must not**: touch the sensory channel; rank or filter by relevance; decide anything; measure its own latency; judge completion; access the protected core.

## Conformance

Every adapter must pass before use. Part of the suite is a **hidden split**, kept outside any workspace and never shown to the model.

| Check | Passes if |
| --- | --- |
| Manifest | Interaction half valid; verbs canonical; versions match |
| Items | Valid fields, normalized boxes inside the window, canonical roles, meaningful names (no raw IDs) |
| Stable handles | Same handles for an unchanged window across two finds |
| Scope | A scoped find returns only items inside the scope; expanding a collapsed group returns its children |
| Faithful execution | Each declared verb on a known test item produces the expected change |
| Surfaces | `act_at` lands within tolerance of the requested point; `drag` follows from and to |
| Honest outcomes | Reported status matches what the core observes; stale options return `unavailable` |
| Coverage | Items found match a vision detection of the same window above a set fraction |
| Movement (relative) | Each declared input produces an observable change |
| Reset | Same first state for the same seed |

## Versioning

The contract carries a version string. Each adapter declares the contract version it targets; a mismatch fails conformance.
