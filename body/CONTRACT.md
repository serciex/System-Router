# Body Contract (v0.2)

The contract is the only interface between the brain and an environment. It is built once and shared by every environment. Adapters translate between an environment's native interface and this contract; they never reimplement it.

This document is also the instruction manual for writing adapters, including adapters the LLM will write later.

*v0.2: one sub-contract per decision model (navigation, action); the navigation translator is removed.*

## Principles

1. **Built once.** Everything common (targets, IDs, cells, levels, outcomes, timing, checks) lives in the contract, not in adapters.
2. **One sub-contract per decision model.** Navigation and action each have their own options and their own execution rule, so the two decision models run in parallel without depending on each other.
3. **Adapters only translate.** Interaction and movement pass straight through to the environment's own functions. The adapter is the only per-environment piece.
4. **The brain sees only the contract.** It never knows which adapter is running.
5. **The core keeps score.** Latency, completion and limits are measured by the protected core, never reported by adapters.

## Coordinates

- All positions are **window-normalized**, `x` and `y` in `[0, 1]`, origin at the top left of the view.
- Grids are defined by the contract, not by adapters.

| Level | Grid | Placement |
| --- | --- | --- |
| 1 | 3×3 | Centered on the anchor. The 8 outer cells are directions, the center is "here" |
| 2 | 6×6 | Over the whole view |
| 3 | 12×12 | Over the whole view |

## Target

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Stable handle, kept the same across frames and across sources |
| `label` | string | Short description the LLM reads |
| `verbs` | list of strings | What can be done to it, e.g. `press`, `type`, `select`, `grab`, `use` |
| `bbox` | 4 floats | Normalized box `(x0, y0, x1, y1)` |
| `reversible` | bool or null | Whether its verbs can be undone; `null` if unknown |
| `confidence` | float in [0, 1] | 1.0 for structured sources, lower for vision |
| `bearing` | float or null | Relative mode: angle from the anchor's facing direction, in degrees (negative is left). `null` in absolute mode |
| `distance` | `near`, `mid`, `far` or null | Relative mode: rough distance from the anchor. `null` in absolute mode |
| `source` | string | Which adapter produced it. Logged, **never shown to the brain** |

### Targets the contract generates itself

- **`self`:** the anchor. Its verbs are global actions (fire, jump, keyboard shortcuts).
- **Directions:** at level 1, one target per outer cell around the anchor ("left", "up-right").
- **Positional:** a selected cell with no element in it becomes "cell (row, col)", for moving into empty space.
- **Edges:** targets at the four view edges, for exploring off-screen.

## Navigation contract

Used by the navigation decision model. Its job is getting to a target, or moving.

**Pointer mode** is declared by the adapter and decides what the options are:

| Mode | Environments | Options shown to the navigation model | Execution |
| --- | --- | --- | --- |
| `absolute` | Web, desktop, VM | Targets returned by `observe` (including edges, which scroll) | The contract moves the pointer straight to the chosen target |
| `relative` | Games, VR, simulated robots | The environment's own **movement inputs** from the adapter (e.g. `forward`, `back`, `turn left`, `strafe right`, `stop`) | The adapter sends the chosen inputs through unchanged |

- In relative mode, navigation may be **several parallel questions**, one per movement axis declared by the adapter (for example `move` and `turn`).
- In relative mode, navigation keeps a **destination**: the target it is heading toward. A `destination` question (options are the observed targets plus `keep`) runs alongside the movement questions, and the destination's bearing and distance are shown to them. This is what "reached" is measured against. The Doom demo's `target` head plays this role.
- Every navigation question includes a `hold` option, so the model can stay put while acting.
- In relative mode, the level-1 directions map onto the adapter's movement inputs.

## Action contract

Used by the action decision model. Its job is applying a verb to the target you are at.

- **Options:** the verbs of the current target (`at`), plus the verbs of `self`, plus `none`.
- **Execution:** the adapter invokes the environment's own function for that verb on that target, passing `arg` if the verb takes one.
- `arg` carries free-form input such as text to type. It comes from the task list written by System 2, never from System 1.
- Before executing, the contract revalidates the target (still exists, same label and role).

## Executing both in one step

- The navigation and action choices from the same System 1 pass execute **in the same step**, so moving and acting can happen at once (strafe while firing).
- In absolute mode, the action runs on the current target first, then the pointer moves, so the action is never applied to the wrong element.

## Outcome

`act` returns one outcome per decision model, so a failure can always be attributed to navigation or to action.

| Field | Type | Meaning |
| --- | --- | --- |
| `navigation.status` | `reached`, `moving`, `blocked`, `held` | Result of the navigation choice. Absolute mode is `reached` once the pointer lands |
| `navigation.text` | string | One line, e.g. "turned left, door now 5° right, near" |
| `action.status` | `done`, `failed`, `partial`, `in_progress`, `none` | Result of the action choice |
| `action.changed` | bool | Whether the observable state changed |
| `action.text` | string | One line, e.g. "pressed Search, results page loaded" |
| `step` | int | Step counter from the contract |
| `latency_ms` | float | **Measured by the core**, never by the adapter |

## Capabilities

Declared by each adapter so the contract knows what is available.

| Field | Meaning |
| --- | --- |
| `pointer` | `absolute` or `relative` |
| `movement_axes` | Relative mode only: the movement inputs grouped by axis |
| `verbs` | Verbs this adapter can execute |
| `stable_ids` | Whether native handles survive across frames |
| `realtime` | Whether the environment keeps moving between steps |
| `has_source` | Whether `source()` returns anything for the labeler |

## Calls the brain uses

| Call | Returns | Notes |
| --- | --- | --- |
| `observe(level, cells)` | Frame, targets inside the selected cells at that level, the current target (`at`), and the navigation and action options | Only targets inside the cells are returned |
| `act(navigation, action, arg)` | Outcome (navigation and action parts) | `navigation` is one choice per navigation question; `action` is one verb or `none` |
| `reset(seed)` | First observation | Restarts the episode |
| `describe_source()` | Raw DOM, accessibility tree or code | **Training only**, for the labeler |
| `check(condition)` | bool | **Protected core.** Tests a verifiable success condition |

## What an adapter provides

The adapter is the only per-environment piece.

| Method | Returns |
| --- | --- |
| `read()` | The frame, the anchor point, and a list of native elements: handle, box, label, role, native actions |
| `invoke(handle, native_action, arg)` | The native result, passed through unchanged |
| `move(inputs)` | Relative mode only: sends the chosen movement inputs, passed through unchanged |
| `point(x, y)` | Absolute mode only: moves the pointer to a normalized position |
| `source()` | Raw source for the labeler, or nothing |
| `capabilities()` | The capabilities table above |
| `reset(seed)`, `close()` | Episode lifecycle |

## What the contract does for every environment

1. Merges elements from all active adapters, preferring structured sources and filling gaps with vision.
2. Assigns and tracks stable target IDs across frames and sources.
3. Converts boxes to normalized coordinates and filters targets by the selected cells and level.
4. Generates `self`, direction, positional and edge targets, and in relative mode computes each target's bearing and distance from the anchor.
5. Builds the navigation and action options from the two sub-contracts.
6. Revalidates targets, then executes the action and navigation choices in the same step.
7. Measures latency, applies executor limits, and records every step in the log, including the anchor before and after and the native inputs sent.

## Adapter rules

An adapter **must**:

- translate native elements to the fields above, honestly and without editorializing;
- list the environment's real movement inputs (relative mode) and verbs, nothing invented;
- keep native handles stable when the environment allows it;
- set `confidence` below 1.0 for anything inferred from pixels;
- pass interaction and movement through to the environment's own functions.

An adapter **must not**:

- filter or rank targets by relevance, which is the world model's job;
- decide anything, which is the decision models' job;
- measure its own latency, judge completion, or touch the protected core.

## Conformance checks

Every adapter must pass these before the world model uses it.

| Check | Passes if |
| --- | --- |
| Schema | Every target has valid fields and normalized boxes inside the view |
| Stable IDs | The same element keeps its ID across consecutive frames with no change |
| Faithful execution | Each declared verb, invoked on a known test element, produces the expected state change |
| Movement inputs | Relative mode: each declared movement input produces an observable change of the anchor or view. Absolute mode: `point` lands on the requested position |
| Honest outcomes | Both the navigation and action statuses match what `check()` and the anchor position observe |
| Reset | `reset(seed)` returns the same first observation for the same seed |

## Versioning

The contract carries a version string. Each adapter declares the contract version it targets, and a version mismatch fails conformance.
