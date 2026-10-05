"""Web starter adapter on MiniWoB++'s DOM: find by container, invoke by element ref, act_at on canvases."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from ..schema import InteractionManifest, Item, Point
from ..vocab import meaningful
from .base import Adapter, result

ROLES = {
    "button": "button", "input_submit": "button", "input_button": "button", "a": "link",
    "input_text": "textbox", "input_password": "textbox", "textarea": "textbox", "input_number": "textbox",
    "input_email": "textbox", "input_search": "textbox", "input_tel": "textbox", "input_date": "textbox",
    "input_checkbox": "checkbox", "input_radio": "radio", "option": "option", "select": "option",
    "img": "image", "canvas": "surface",
}
TEXT_TAGS = {"t", "span", "div", "p", "li", "td", "th", "label", "h1", "h2", "h3", "h4", "b", "i", "strong", "em"}
SKIP_IDS = {"wrap", "query", "reward-display", "sync-task-cover"}
TYPEABLE = {"textbox"}


def _scalar(value) -> float:
    try:
        return float(value[0])
    except (TypeError, IndexError):
        return float(value)


class MiniWoBDomAdapter(Adapter):
    name = "web"

    def __init__(self, environment, collapse_over: int = 12):
        self.env = environment
        self.collapse_over = int(collapse_over)

    def manifest(self) -> InteractionManifest:
        return InteractionManifest(pointer_mode="absolute",
                                   verbs=("click", "type", "hover", "select", "drag", "scroll_up", "scroll_down"),
                                   self_verbs=("scroll_up", "scroll_down"),
                                   actuators={"pointer": {}, "keyboard": {}})

    # ------------------------------------------------------------------ find
    def find(self, scope: Any = None) -> list[Item]:
        width, height = self.env.size
        elements = [e for e in self.env.dom_elements if e.get("id") not in SKIP_IDS]
        by_ref = {int(e["ref"]): e for e in elements}
        items: list[Item] = []
        for element in elements:
            item = self._item(element, by_ref, width, height)
            if item is not None:
                items.append(item)
        if scope is not None:
            return [i for i in items if i.container == scope]
        return self._collapse(items, by_ref, width, height)

    def _box(self, element: dict, width: int, height: int):
        left, top = _scalar(element["left"]), _scalar(element["top"])
        w, h = _scalar(element["width"]), _scalar(element["height"])
        if w <= 0 or h <= 0:
            return None
        box = (max(left / width, 0.0), max(top / height, 0.0), min((left + w) / width, 1.0), min((top + h) / height, 1.0))
        return box if box[0] < box[2] and box[1] < box[3] else None

    def _item(self, element: dict, by_ref: dict, width: int, height: int) -> Optional[Item]:
        tag = str(element.get("tag", ""))
        flags = element.get("flags")
        is_leaf = bool(flags[3]) if flags is not None and len(flags) > 3 else True
        text = str(element.get("text") or "").strip()
        value = str(element.get("value") or "").strip()
        role = ROLES.get(tag) or ("text" if tag in TEXT_TAGS and is_leaf and text else None)
        if role is None:
            return None
        box = self._box(element, width, height)
        if box is None:
            return None
        name = text or value or str(element.get("id") or "")
        if not meaningful(name):
            name = f"unnamed {role}"
        if role == "surface":
            return Item(handle=int(element["ref"]), kind="surface", role="surface", name=name or "canvas",
                        verbs=("click", "drag", "hover"), bbox=box, dims=2, container=element.get("parent"))
        verbs = ("click", "type", "hover") if role in TYPEABLE else ("click", "hover")
        if role == "option":
            verbs = ("select", "click", "hover")
        return Item(handle=int(element["ref"]), kind="element", role=role, name=name, verbs=verbs, bbox=box,
                    container=element.get("parent"), value=value)

    def _collapse(self, items: list[Item], by_ref: dict, width: int, height: int) -> list[Item]:
        groups: dict[Any, list[Item]] = defaultdict(list)
        for item in items:
            groups[item.container].append(item)
        out: list[Item] = []
        for container, members in groups.items():
            parent = by_ref.get(int(container)) if container not in (None, 0) else None
            if len(members) <= self.collapse_over or parent is None:
                out.extend(members)
                continue
            box = self._box(parent, width, height) or members[0].bbox
            name = str(parent.get("id") or parent.get("classes") or "list").strip() or "list"
            out.append(Item(handle=int(container), kind="group", role="group", name=name, verbs=(), bbox=box,
                            collapsed=len(members)))
        return out

    # ------------------------------------------------------------------ execute
    def invoke(self, handle, verb: str, arg: Optional[str] = None) -> dict:
        if verb in ("scroll_up", "scroll_down"):
            return {**self.env.scroll(verb.split("_")[1]), "text": verb.replace("_", " ")}
        if handle is None:
            return result(False, f"{verb} needs a target")
        if verb in ("click", "select"):
            return {**self.env.click_ref(handle), "text": "clicked"}
        if verb == "type":
            return {**self.env.type_into_ref(handle, arg or ""), "text": f'typed "{arg or ""}"'}
        if verb == "hover":
            element = next((e for e in self.env.dom_elements if int(e["ref"]) == int(handle)), None)
            if element is None:
                return result(False, "element gone")
            box = self._box(element, *self.env.size)
            return {**self.env.point((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), "text": "pointer moved"}
        return result(False, f"unsupported verb {verb}")

    def act_at(self, handle, points: list[Point], verb: str) -> dict:
        if verb == "click":
            return {**self.env.click_at(*points[0]), "text": "clicked"}
        if verb == "hover":
            return {**self.env.point(*points[0]), "text": "pointer moved"}
        if verb == "drag" and len(points) >= 2:
            return {**self.env.drag(points[0], points[1]), "text": "dragged"}
        return result(False, f"unsupported verb {verb} on a surface")
