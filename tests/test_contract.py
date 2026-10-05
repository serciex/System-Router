from body.contract import Body
from body.core import Core
from body.schema import HOLD, NONE
from body.vocab import CORE_ACTIONS

from .fakes import FakePage, FakeRelativeAdapter, FakeWebAdapter, StillScreen


def make_body():
    page = FakePage()
    core = Core(allowed_verbs=("click", "type", "hover", "drag", "scroll_up", "use"))
    return Body(page, FakeWebAdapter(page), core, wait_cap_s=0.2, wait_poll_s=0.05), page


def by_label(body, text):
    return next(t for t in body.slot.targets if text in t.label)


def test_reset_fills_the_slot_with_canonical_labels_and_stable_ids():
    body, _ = make_body()
    body.reset(0)
    labels = [t.label for t in body.slot.targets]
    assert "button: Submit" in labels and "surface: canvas" in labels
    first = [(t.id, t.label) for t in body.slot.targets]
    body.find()
    assert [(t.id, t.label) for t in body.slot.targets] == first


def test_auto_find_runs_only_on_window_change():
    body, page = make_body()
    body.reset(0)
    body.observe()
    assert not body.auto_find()
    page.touch(90)
    body.observe()
    assert body.auto_find()


def test_core_actions_only_in_the_action_question():
    body, _ = make_body()
    body.reset(0)
    navigation, action = body.questions()
    nav_keys = {o.key for o in navigation["navigate"]}
    action_keys = {o.key for o in action}
    assert HOLD in nav_keys and not nav_keys & set(CORE_ACTIONS)
    assert set(CORE_ACTIONS) <= action_keys and NONE in action_keys and "scroll_up@self" in action_keys


def test_navigating_to_a_group_expands_it():
    body, _ = make_body()
    body.reset(0)
    group = by_label(body, "group: List")
    body.begin_step()
    outcome = body.act({"navigate": group.id})
    assert outcome.navigation.status == "reached"
    assert len(body.slot.targets) == 15 and body.slot.scope == group.id


def test_navigate_then_act_on_the_current_target():
    body, page = make_body()
    body.reset(0)
    field = by_label(body, "Username")
    body.begin_step()
    assert body.act({"navigate": field.id}).navigation.status == "reached"
    _, action = body.questions()
    assert f"type@{field.id}" in {o.key for o in action}
    body.begin_step()
    outcome = body.act({"navigate": HOLD}, f"type@{field.id}", "alice")
    assert outcome.action.status == "done" and page.typed == "alice"
    assert body.check({"type": "value_equals", "target": "Username", "value": "alice"})


def test_stale_option_is_unavailable():
    body, page = make_body()
    body.reset(0)
    link = by_label(body, "Help")
    body.at = link
    page.removed.add(3)
    page.touch(77)
    body.observe()
    body.begin_step()
    assert body.act({"navigate": HOLD}, f"click@{link.id}").action.status == "unavailable"


def test_surface_point_and_drag():
    body, page = make_body()
    body.reset(0)
    canvas = by_label(body, "canvas")
    body.begin_step()
    body.act({"navigate": canvas.id}, NONE, point=(0.4, 0.35))
    assert body.at.point == (0.4, 0.35)
    body.begin_step()
    body.act({"navigate": HOLD}, f"drag@{canvas.id}", drag_to=(0.5, 0.4))
    assert page.points[-1] == ("drag", [(0.4, 0.35), (0.5, 0.4)])


def test_success_and_latency_are_reported():
    body, _ = make_body()
    body.reset(0)
    body.at = by_label(body, "Submit")
    body.begin_step()
    outcome = body.act({}, f"click@{body.at.id}")
    assert outcome.env_done and outcome.env_success and outcome.latency_ms >= 0
    assert body.check({"type": "env_success"})


def test_wait_without_change_hits_the_cap():
    body, _ = make_body()
    body.reset(0)
    body.begin_step()
    outcome = body.wait(cap_s=0.1)
    assert outcome.action.status == "none" and "no change" in outcome.action.text


def test_relative_mode_axes_destination_and_reaching():
    integration = StillScreen()
    body = Body(integration, FakeRelativeAdapter(), Core(allowed_verbs=("use",)))
    body.reset(0)
    door = body.slot.targets[0]
    assert door.bearing > 0 and door.distance == "far"
    navigation, _ = body.questions()
    assert set(navigation) == {"move", "turn", "destination"}
    body.begin_step()
    body.act({"destination": door.id, "turn": "right", "move": "forward"})
    for _ in range(6):
        body.begin_step()
        outcome = body.act({"turn": "right", "move": "forward"})
    assert outcome.navigation.status == "reached"
