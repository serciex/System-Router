"""Canonical vocabulary (contract v0.3). Extended only by a person, never by the LLM."""

from __future__ import annotations

import re

ROLES = (
    "button", "link", "textbox", "checkbox", "radio", "option", "menu", "tab", "text", "image", "icon",
    "surface", "group", "file", "symbol", "entity", "limb",
)

POINTER_VERBS = ("click", "double_click", "right_click", "hover", "type", "select", "scroll_up", "scroll_down",
                 "drag", "press_key")
CODE_VERBS = ("open", "read", "search", "write", "run")
EMBODIED_VERBS = ("grab", "release", "place", "use")
VERBS = POINTER_VERBS + CODE_VERBS + EMBODIED_VERBS

THINK, FIND, WAIT = "think", "find", "wait"
CORE_ACTIONS = (THINK, FIND, WAIT)

SCREEN_KINDS = ("flat", "stereo", "camera", "depth")
SENSOR_TYPES = ("joint_position", "joint_velocity", "contact", "pose", "depth_map", "force_torque", "imu",
                "scalar", "flag")
POINTER_MODES = ("absolute", "relative", "none")

_RAW_ID = re.compile(r"^([\W\d_]+|[0-9a-f]{8,}|[0-9a-f-]{32,}|ref\d+|t\d+)$", re.IGNORECASE)


def label(role: str, name: str) -> str:
    """The brain-facing label, always `role: name`."""
    return f"{role}: {name}" if name else f"{role}: (unnamed)"


def meaningful(name: str) -> bool:
    """False for empty names and raw identifiers such as hashes or numeric ids."""
    name = (name or "").strip()
    return bool(name) and not _RAW_ID.match(name)
