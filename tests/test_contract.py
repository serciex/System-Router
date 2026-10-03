from body.adapters.degraded import DegradedAdapter
from body.conformance import run_conformance
from body.contract import Body
from body.core import Core
from body.grid import Grid
from body.schema import HOLD, NONE

from .fakes import FakeAdapter, FakePage, FakeRelativeAdapter


def make_body(adapter=None):
    page = FakePage()
    adapter = adapter or FakeAdapter(page)
    grid = Grid({1: 3, 2: 6, 3: 12}, [1, 3])
    return Body([adapter], grid, Core(allowed_verbs=("click", "type", "scroll_up", "scroll_down", "use"))), page


def test_ids_are_stable_across_reads():
    body, _ = make_body()
    body.reset(0)
    first = [(t.id, t.label) for t in body.targets]
    body.sense()
    assert [(t.id, t.label) for t in body.targets] == first


def test_observe_filters_by_cells_and_adds_positional_targets():
    body, _ = make_body()
    body.reset(0)
    submit = next(t for t in body.targets if t.label.startswith("Submit"))
    cell = body.grid.cell_of(3, *submit.center)
    empty = 0 if cell != 0 else 1
    obs = body.observe(3, [cell, empty])
    kinds = {t.kind for t in obs.targets}
    assert submit.id in {t.id for t in obs.targets}
    assert "positional" in kinds
    assert obs.navigation["navigate"][0].key == HOLD
    assert obs.action[0].key == NONE


def test_level_one_returns_directions():
    body, _ = make_body()
    body.reset(0)
    obs = body.observe(1, [3, 5])
    assert {t.id for t in obs.targets} == {"dir:left", "dir:right"}


def test_navigate_then_act_on_current_target():
    body, page = make_body()
    body.reset(0)
    field = next(t for t in body.targets if t.label.startswith("Username"))
    obs = body.observe(3, [body.grid.cell_of(3, *field.center)])
    body.begin_step()
    outcome = body.act({"navigate": field.id}, NONE)
    assert outcome.navigation.status == "reached"
    assert body.at is not None and body.at.id == field.id

    obs = body.observe(3, [body.grid.cell_of(3, *field.center)])
    assert f"type@{field.id}" in {o.key for o in obs.action}
    body.begin_step()
    outcome = body.act({"navigate": HOLD}, f"type@{field.id}", "alice")
    assert outcome.action.status == "done" and page.typed == "alice"
    assert outcome.navigation.status == "held"
    assert body.check({"type": "value_equals", "target": "Username", "value": "alice"})


def test_action_on_missing_target_fails_and_core_blocks_disallowed_verbs():
    body, _ = make_body()
    body.reset(0)
    body.observe(3, [])
    assert body.act({}, "click@t999").action.status == "failed"
    body.core.allowed_verbs = ("click",)
    field = next(t for t in body.targets if t.label.startswith("Username"))
    body.at = field
    body.observe(3, [])
    assert body.act({}, f"type@{field.id}", "x").action.status == "failed"


def test_env_success_is_reported():
    body, _ = make_body()
    body.reset(0)
    submit = next(t for t in body.targets if t.label.startswith("Submit"))
    body.at = submit
    body.observe(3, [])
    body.begin_step()
    outcome = body.act({}, f"click@{submit.id}")
    assert outcome.env_done and outcome.env_success and outcome.latency_ms >= 0
    assert body.check({"type": "env_success"})


def test_relative_mode_has_axes_destination_and_bearings():
    body, _ = make_body(FakeRelativeAdapter())
    body.reset(0)
    door = body.targets[0]
    assert door.bearing is not None and door.bearing > 0 and door.distance == "far"
    obs = body.observe(3, [body.grid.cell_of(3, *door.center)])
    assert set(obs.navigation) == {"move", "turn", "destination"}
    outcome = body.act({"destination": door.id, "turn": "right", "move": "forward"}, NONE)
    assert outcome.navigation.status in ("moving", "reached")
    for _ in range(6):
        body.observe(3, [])
        outcome = body.act({"turn": "right", "move": "forward"}, NONE)
    assert outcome.navigation.status == "reached"


def test_degraded_adapter_keeps_interaction():
    page = FakePage()
    body, _ = make_body(DegradedAdapter(FakeAdapter(page), drop=0.0, jitter=0.0, blank_label=1.0, seed=1))
    body.reset(0)
    assert all(t.label == "" for t in body.targets)
    assert all(t.confidence <= 0.8 for t in body.targets)


def test_fake_adapter_passes_conformance():
    report = run_conformance(FakeAdapter(FakePage()))
    assert report.passed, str(report)
