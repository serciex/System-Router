"""Native coordinate grounding, compared against the narrowing library in stage 2."""

from __future__ import annotations

import re
from typing import Optional

from body.schema import BBox, Point

from .llm import LLM, generate

_PAIR = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


def point_native(llm: LLM, image, description: str, box: BBox = (0.0, 0.0, 1.0, 1.0)) -> Optional[Point]:
    """Ask the VLM for one point as (x, y) on a 0 to 1000 scale; returns a window-normalized point inside box."""
    prompt = (f"Point to: {description}\nAnswer with the point only, as (x, y) where x and y run from 0 to 1000 "
              "across the whole image.")
    match = _PAIR.search(generate(llm, prompt, image=image, max_new_tokens=24))
    if not match:
        return None
    x, y = (min(max(float(v) / 1000.0, 0.0), 1.0) for v in match.groups())
    return min(max(x, box[0]), box[2]), min(max(y, box[1]), box[3])
