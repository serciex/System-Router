"""Stage-1 policy: hand-written rules in place of the world model (no training).

Route to System 1 (System 2 still plans first and catches low-confidence steps through the hard rule),
or to System 2 always for the reasoning-only baseline. Segment at the finest active level, selecting
every cell that contains a target, which maximizes recall and gives the rewards a reference to score.
"""

from __future__ import annotations

from body.contract import Body
from body.grid import Grid

from .codec import S1, S2, Decision


class RulePolicy:
    def __init__(self, grid: Grid, route: str = "s1"):
        if route not in ("s1", "s2"):
            raise ValueError("route must be 's1' or 's2'")
        self.grid = grid
        self.route = S1 if route == "s1" else S2

    def decide(self, body: Body) -> Decision:
        level = max(self.grid.levels)
        anchor = body.frame.anchor
        cells = {self.grid.cell_of(level, *t.center, anchor) for t in body.targets}
        cells.discard(None)
        return Decision(route=self.route, level=level, cells=tuple(sorted(cells)))
