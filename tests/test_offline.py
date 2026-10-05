import pytest

pytestmark = pytest.mark.skip(reason="v2 world-model design, obsolete under spec v3")

import numpy as np

from body.grid import Grid
from wm.codec import S1, ActionCodec, Decision
from wm.dataset import EpisodeWriter, SequenceSampler, load_episodes
from wm.evaluate import episode_batch, suggest, sweep_cells, sweep_escalation
from wm.hindsight import attach_labels, important_targets, label_episode

GRID = Grid({1: 3, 2: 6, 3: 12}, [1, 3])
CODEC = ActionCodec(GRID)


def record(nav, action, s2_nav, s2_action, nav_status="reached", action_status="none", acted=None):
    return {
        "anchor": [0.5, 0.5],
        "positions": {"t1": [0.2, 0.15], "t2": [0.8, 0.85]},
        "s1": {"navigation": {"navigate": nav}, "action": action, "confidence": 0.9},
        "s2": {"navigation": {"navigate": s2_nav}, "action": s2_action},
        "nav_target": nav if nav in ("t1", "t2") else None,
        "nav_status": nav_status,
        "acted_target": acted,
        "action_status": action_status,
    }


def episode(success=True):
    records = [
        record("t1", "none", "t1", "none"),
        record("hold", "click@t1", "hold", "click@t1", nav_status="held", action_status="done", acted="t1"),
        record("t2", "none", "hold", "none", nav_status="reached"),
    ]
    steps = len(records)
    arrays = {"vis": np.zeros((steps, 4), np.float32), "txt": np.zeros((steps, 3), np.float32),
              "vec": np.zeros((steps, 2), np.float32),
              "action": np.stack([CODEC.encode(Decision(S1, 3, (0,)))] * steps)}
    return {"arrays": arrays, "records": records, "success": success, "meta": {}}


def test_important_targets_come_only_from_successful_episodes():
    assert important_targets(episode()["records"], True) == {"t1", "t2"}
    assert important_targets(episode()["records"], False) == set()


def test_labels_mark_cells_and_disagreement():
    labels = label_episode(episode(), GRID, CODEC, "s2")
    offset = CODEC.cell_counts[0]
    t1_cell = GRID.cell_of(3, 0.2, 0.15)
    assert labels["cell_target"][0, offset + t1_cell] == 1.0
    assert labels["cell_mask"].all()
    assert labels["s1_wrong"][:, 0].tolist() == [0.0, 0.0, 1.0]
    failed = label_episode(episode(success=False), GRID, CODEC, "s2")
    assert not failed["cell_mask"].any() and failed["s1_mask"].all()


def test_writer_loader_and_sampler(tmp_path):
    writer = EpisodeWriter(tmp_path)
    for _ in range(3):
        ep = episode()
        writer.start(task="x", adapters=["dom"], heldout=False)
        for t, rec in enumerate(ep["records"]):
            writer.add({k: ep["arrays"][k][t] for k in ("vis", "txt", "vec")}, ep["arrays"]["action"][t], rec)
        writer.finish(True)
    loaded = attach_labels(load_episodes(tmp_path), GRID, CODEC)
    assert len(loaded) == 3 and loaded[0]["meta"]["adapters"] == ["dom"]
    batch = SequenceSampler(loaded, batch_size=4, length=4).sample()
    assert batch["vis"].shape == (4, 4, 4) and batch["action"].shape == (4, 4, CODEC.dim)
    assert batch["is_first"].shape == (4, 4, 1) and batch["is_first"][:, 0].all()
    assert batch["cell_target"].shape == (4, 4, CODEC.n_cells)
    whole = episode_batch(loaded[0]["arrays"])
    assert whole["action"][0, 0].sum() == 0 and whole["is_first"][0, 0, 0]


def test_sweeps_and_suggestion():
    probs = np.array([[0.9, 0.1, 0.2], [0.8, 0.7, 0.1]])
    targets = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.float32)
    cells = sweep_cells(probs, targets, np.array([True, True]), [0.5, 0.95], min_cells=1, max_cells=3)
    assert cells[0]["recall"] == 1.0 and cells[0]["mean_cells"] == 1.5
    esc = sweep_escalation(np.array([0.9, 0.2, 0.6]), np.array([1.0, 0.0, 0.0]), np.array([True] * 3), [0.5, 0.8])
    assert esc[0]["caught"] == 1.0 and esc[1]["escalation_rate"] == 1 / 3
    best = suggest(cells, esc, target_recall=0.95, target_caught=0.9)
    assert best["cell_threshold"] == 0.5 and best["escalate_threshold"] == 0.8
