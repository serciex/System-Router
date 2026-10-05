from body.conformance import run_conformance
from body.narrowing import HERE, Narrowing, subregions
from body.prompt import build, options_text
from body.schema import Observation, Option, Screen
from body.vocab import label, meaningful

from .fakes import FakePage, FakeWebAdapter


def test_subregions_cover_the_box():
    boxes = subregions((0.0, 0.0, 0.9, 0.9), 2)
    assert len(boxes) == 9 and boxes[0] == ("top-left", (0.0, 0.0, 0.3, 0.3)) and boxes[8][1][2] == 0.9
    assert [n for n, _ in subregions((0.0, 0.0, 0.9, 0.1), 1)] == ["start", "middle", "end"]


def test_narrowing_picks_then_here():
    n = Narrowing((0.0, 0.0, 0.9, 0.9), 2, size=(900, 900), min_px=1)
    assert n.pick("bottom-right") is None and n.box == (0.6, 0.6, 0.9, 0.9)
    point = n.pick(HERE)
    assert abs(point[0] - 0.75) < 1e-9 and n.depth == 1
    assert n.options()[0].key == HERE and "x " in n.options()[1].text


def test_narrowing_stops_when_small():
    n = Narrowing((0.0, 0.0, 1.0, 1.0), 2, size=(90, 90), min_px=12)
    assert n.pick("centre") is None
    assert n.pick("centre") is not None


def test_vocab_labels_and_names():
    assert label("button", "Submit") == "button: Submit"
    assert meaningful("Submit") and not meaningful("") and not meaningful("4f9a2c1d7e")


def test_prompt_layout_order():
    labeled = {"navigate": [("AA", 1, Option("hold", "stay where you are"))], "action": [("AB", 2, Option("think", "think"))]}
    import numpy as np

    obs = Observation(screens=[Screen("flat", np.zeros((4, 4, 3), dtype=np.uint8))])
    parts = build(["step 1: act"], obs, ["Goal: x"], ["window changed since last find: no"], labeled)
    text = parts.text()
    assert text.index("History") < text.index("[screen image]") < text.index("Goals") < text.index("Status") < text.index("Options")
    assert "AA: stay where you are" in options_text(labeled)


def test_fake_web_adapter_passes_conformance():
    page = FakePage()
    report = run_conformance(FakeWebAdapter(page), page)
    assert report.passed, str(report)
