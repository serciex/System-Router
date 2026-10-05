"""Shared narrowing library: positions inside a surface, picked region by region within one step."""

from __future__ import annotations

from typing import Optional

from .schema import BBox, Option, Point

HERE = "here"
REGIONS_2D = ("top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right")
REGIONS_1D = ("start", "middle", "end")


def _edges(a: float, b: float) -> list[float]:
    step = (b - a) / 3
    return [a, a + step, a + 2 * step, b]


def subregions(box: BBox, dims: int) -> list[tuple[str, BBox]]:
    x0, y0, x1, y1 = box
    xs, ys = _edges(x0, x1), _edges(y0, y1)
    if dims == 1:
        if (x1 - x0) >= (y1 - y0):
            return [(n, (xs[i], y0, xs[i + 1], y1)) for i, n in enumerate(REGIONS_1D)]
        return [(n, (x0, ys[i], x1, ys[i + 1])) for i, n in enumerate(REGIONS_1D)]
    return [(n, (xs[i % 3], ys[i // 3], xs[i % 3 + 1], ys[i // 3 + 1])) for i, n in enumerate(REGIONS_2D)]


def centre(box: BBox) -> Point:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _span(box: BBox, size: Optional[tuple[int, int]]) -> str:
    if size:
        w, h = size
        return (f"x {round(box[0] * w)} to {round(box[2] * w)}, "
                f"y {round(box[1] * h)} to {round(box[3] * h)}")
    return f"x {box[0]:.2f} to {box[2]:.2f}, y {box[1]:.2f} to {box[3]:.2f}"


class Narrowing:
    """One narrowing session on a surface. `pick` returns a point once `here` is chosen or the region is small."""

    def __init__(self, box: BBox, dims: int, size: Optional[tuple[int, int]] = None, min_px: int = 12,
                 max_depth: int = 6):
        self.box, self.dims, self.size = box, max(1, dims), size
        self.min_px, self.max_depth = min_px, max_depth
        self.depth = 0
        self.path: list[str] = []

    def options(self) -> list[Option]:
        options = [Option(HERE, f"here, centre of {_span(self.box, self.size)}")]
        options += [Option(name, f"{name}, {_span(box, self.size)}") for name, box in subregions(self.box, self.dims)]
        return options

    def _small(self) -> bool:
        if not self.size:
            return (self.box[2] - self.box[0]) < 0.01 and (self.box[3] - self.box[1]) < 0.01
        w, h = self.size
        return (self.box[2] - self.box[0]) * w <= self.min_px and (self.box[3] - self.box[1]) * h <= self.min_px

    def pick(self, key: str) -> Optional[Point]:
        if key == HERE:
            return centre(self.box)
        regions = dict(subregions(self.box, self.dims))
        if key not in regions:
            raise ValueError(f"Unknown region {key!r}")
        self.box = regions[key]
        self.depth += 1
        self.path.append(key)
        if self.depth >= self.max_depth or self._small():
            return centre(self.box)
        return None
