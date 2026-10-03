import numpy as np

from body.grid import Grid
from brain.goal_context import GoalContext, Subtask
from brain.labeler import Label, Labeler
from common.config import apply_override
from wm.codec import S2, ActionCodec, Decision
from wm.rewards import dec_reward, seg_reward

from .test_contract import make_body

DEC = dict(a=0.2, b=1.0, p=0.5, p_irr=2.0, q=0.5, kappa=0.05)


def test_codec_round_trip_and_shape():
    codec = ActionCodec(Grid({1: 3, 2: 6, 3: 12}, [1, 3]))
    assert codec.shape[:2] == (2, 2) and len(codec.shape) == 2 + 9 + 144
    decision = Decision(route=S2, level=3, cells=(0, 5, 143))
    assert codec.decode(codec.encode(decision)) == decision
    assert codec.space.sample().shape == (codec.dim,)


def test_decision_reward_ordering():
    def r(correct, latency):
        return dec_reward(correct, latency, False, False, 1000.0, **DEC)

    fast_right, slow_right = r(True, 500), r(True, 4000)
    fast_wrong, slow_wrong = r(False, 500), r(False, 4000)
    assert fast_right > slow_right > fast_wrong > slow_wrong
    assert dec_reward(False, 500, False, True, 1000.0, **DEC) < fast_wrong  # irreversible costs more
    assert dec_reward(None, 500, False, False, 1000.0, **DEC) == 0.0


def test_segmentation_reward_prefers_covering_only_what_matters():
    body, _ = make_body()
    body.reset(0)
    submit = next(t for t in body.targets if t.label.startswith("Submit"))
    important = {submit.id: {"level": 3, "verbs": ["click"]}}
    anchor = body.frame.anchor
    tight = body.observe(3, [body.grid.cell_of(3, *submit.center)])
    r_tight, info = seg_reward(important, tight, body.targets, body.grid, anchor, beta=1.0, c=0.1, eps=0.01)
    every = body.observe(3, list(range(144)))
    r_all, _ = seg_reward(important, every, body.targets, body.grid, anchor, beta=1.0, c=0.1, eps=0.01)
    miss = body.observe(3, [0])
    r_miss, _ = seg_reward(important, miss, body.targets, body.grid, anchor, beta=1.0, c=0.1, eps=0.01)
    assert info["recall"] == 1.0
    assert r_tight > r_all and r_tight > r_miss


def test_level_one_only_covers_targets_that_need_a_direction():
    body, _ = make_body()
    body.reset(0)
    submit = next(t for t in body.targets if t.label.startswith("Submit"))
    direction = body.grid.cell_of(1, *submit.center, body.frame.anchor)
    obs = body.observe(1, [direction])
    needs_direction = {submit.id: {"level": 1, "verbs": []}}
    needs_element = {submit.id: {"level": 3, "verbs": ["click"]}}
    _, ok = seg_reward(needs_direction, obs, body.targets, body.grid, body.frame.anchor, 1.0, 0.1, 0.01)
    _, not_ok = seg_reward(needs_element, obs, body.targets, body.grid, body.frame.anchor, 1.0, 0.1, 0.01)
    assert ok["recall"] == 1.0 and not_ok["recall"] == 0.0


def test_labeler_verdict():
    label = Label(valid={"t1", "t2"}, important={"t1": {"level": 3, "verbs": ["click"]}})
    assert Labeler.verdict(label, "t1", "none", None)
    assert Labeler.verdict(label, "hold", "click@t1", "t1")
    assert not Labeler.verdict(label, "t2", "none", None)
    assert not Labeler.verdict(label, "hold", "type@t1", "t1")


def test_goal_context_stable_text_and_stall():
    goal = GoalContext()
    goal.wipe("log in")
    assert "none yet" in goal.stable_text()
    goal.set_plan([Subtask("type the username", arg="alice"), Subtask("click Submit", {"type": "env_success"})])
    assert 'Text to use for this subtask: "alice"' in goal.stable_text()
    for _ in range(3):
        goal.tick(2)
    assert goal.needs_replan
    goal.advance()
    goal.advance()
    assert goal.done


def test_config_override_parses_yaml():
    cfg = {"llm": {"quantization": "none"}}
    apply_override(cfg, "llm.quantization=4bit")
    apply_override(cfg, "env.tasks=[a, b]")
    assert cfg["llm"]["quantization"] == "4bit" and cfg["env"]["tasks"] == ["a", "b"]
    assert np.isfinite(1.0)
