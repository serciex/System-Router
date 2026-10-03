"""Window-normalized grids per level, defined by the contract (never by adapters).

Level 1 is a 3x3 block centered on the anchor: the 8 outer cells are directions, the center is "here".
Higher levels tile the whole view.
"""

from __future__ import annotations

from .schema import BBox

DIRECTIONS = ("up-left", "up", "up-right", "left", "here", "right", "down-left", "down", "down-right")


class Grid:
    def __init__(self, sizes: dict[int, int], levels: list[int], level1_span: float = 0.6):
        self.sizes = {int(k): int(v) for k, v in sizes.items()}
        self.levels = sorted(int(level) for level in levels)
        self.level1_span = float(level1_span)
        for level in self.levels:
            if level not in self.sizes:
                raise ValueError(f"No size configured for level {level}")
        if 1 in self.levels and self.sizes[1] != 3:
            raise ValueError("Level 1 must be a 3x3 block around the anchor")

    def n_cells(self, level: int) -> int:
        return self.sizes[level] ** 2

    def cell_rect(self, level: int, index: int, anchor: tuple[float, float] = (0.5, 0.5)) -> BBox:
        side = self.sizes[level]
        row, col = divmod(index, side)
        if level == 1:
            span = self.level1_span
            x0 = anchor[0] - span / 2 + col * span / 3
            y0 = anchor[1] - span / 2 + row * span / 3
            return _clip((x0, y0, x0 + span / 3, y0 + span / 3))
        size = 1.0 / side
        return (col * size, row * size, (col + 1) * size, (row + 1) * size)

    def cell_of(self, level: int, x: float, y: float, anchor: tuple[float, float] = (0.5, 0.5)) -> int | None:
        """Cell index containing a point, or None if outside the level's area."""
        side = self.sizes[level]
        if level == 1:
            span = self.level1_span
            col = int((x - (anchor[0] - span / 2)) // (span / 3))
            row = int((y - (anchor[1] - span / 2)) // (span / 3))
            # Points beyond the block still belong to the outer direction they lie in.
            col, row = min(max(col, 0), 2), min(max(row, 0), 2)
            return row * 3 + col
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        col = min(int(x * side), side - 1)
        row = min(int(y * side), side - 1)
        return row * side + col

    def direction(self, index: int) -> str:
        return DIRECTIONS[index]

    def cell_name(self, level: int, index: int) -> str:
        if level == 1:
            return self.direction(index)
        row, col = divmod(index, self.sizes[level])
        return f"cell row {row + 1}, col {col + 1}"

    def contains(self, level: int, index: int, x: float, y: float, anchor: tuple[float, float] = (0.5, 0.5)) -> bool:
        return self.cell_of(level, x, y, anchor) == index


def _clip(box: BBox) -> BBox:
    x0, y0, x1, y1 = box
    return (min(max(x0, 0.0), 1.0), min(max(y0, 0.0), 1.0), min(max(x1, 0.0), 1.0), min(max(y1, 0.0), 1.0))


def iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
