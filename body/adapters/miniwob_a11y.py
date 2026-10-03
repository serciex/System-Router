"""Accessibility-tree adapter for MiniWoB++: same page, described by the browser's accessibility tree.

Reads Chrome's accessibility tree through the Chrome DevTools Protocol (Selenium `execute_cdp_cmd`),
so labels are roles and accessible names rather than DOM tags and text. Interaction goes through
coordinates (click at the element's center, then type), not element refs, so it exercises a different
native path from the DOM adapter.

Assumes the MiniWoB page is rendered at the top-left of the viewport, so CSS pixels match screenshot pixels.
"""

from __future__ import annotations

from typing import Optional

from ..schema import Capabilities, Frame, NativeElement
from .base import Adapter, result

INTERACTIVE = {"button", "link", "textbox", "checkbox", "radio", "combobox", "listbox", "option", "menuitem",
               "tab", "searchbox", "spinbutton", "switch", "StaticText"}
TYPEABLE = {"textbox", "searchbox", "combobox", "spinbutton"}


class MiniWoBA11yAdapter(Adapter):
    name = "a11y"

    def __init__(self, environment):
        self.env = environment
        self._boxes: dict[int, tuple[float, float, float, float]] = {}

    def capabilities(self) -> Capabilities:
        return Capabilities(pointer="absolute", verbs=("click", "type", "scroll_up", "scroll_down"),
                            stable_ids=True, realtime=False, has_source=True)

    def reset(self, seed: Optional[int] = None) -> Frame:
        self.env.reset(seed=seed)
        return self.read()

    def _cdp(self, command: str, params: dict) -> dict:
        driver = self.env.driver
        if driver is None or not hasattr(driver, "execute_cdp_cmd"):
            raise RuntimeError("The accessibility adapter needs a Chrome Selenium driver with CDP access")
        return driver.execute_cdp_cmd(command, params)

    def read(self) -> Frame:
        width, height = self.env.size
        nodes = self._cdp("Accessibility.getFullAXTree", {}).get("nodes", [])
        elements, self._boxes = [], {}
        for node in nodes:
            if node.get("ignored"):
                continue
            role = (node.get("role") or {}).get("value", "")
            name = str((node.get("name") or {}).get("value", "")).strip()
            backend = node.get("backendDOMNodeId")
            if role not in INTERACTIVE or backend is None or (role == "StaticText" and not name):
                continue
            box = self._box(backend, width, height)
            if box is None:
                continue
            value = str((node.get("value") or {}).get("value", "")).strip()
            verbs = ("click", "type") if role in TYPEABLE else ("click",)
            self._boxes[backend] = box
            elements.append(NativeElement(handle=backend, bbox=box, label=f"{role}: {name or '(no name)'}",
                                          role=role, native_actions=verbs, value=value, confidence=1.0))
        return Frame(image=self.env.image, anchor=self.env.pointer, elements=elements, text=self.env.utterance,
                     info={"task": self.env.task})

    def _box(self, backend: int, width: int, height: int):
        try:
            model = self._cdp("DOM.getBoxModel", {"backendNodeId": backend}).get("model")
        except Exception:  # noqa: BLE001 - nodes without layout have no box
            return None
        if not model:
            return None
        quad = model.get("border") or model.get("content")
        xs, ys = quad[0::2], quad[1::2]
        box = (max(min(xs) / width, 0.0), max(min(ys) / height, 0.0), min(max(xs) / width, 1.0), min(max(ys) / height, 1.0))
        return box if box[0] < box[2] and box[1] < box[3] else None

    def invoke(self, handle, native_action: str, arg: Optional[str] = None) -> dict:
        if native_action in ("scroll_up", "scroll_down"):
            res = self.env.scroll(native_action.split("_")[1])
            return {**res, "text": native_action.replace("_", " ")}
        box = self._boxes.get(handle)
        if box is None:
            return result(False, "element no longer in the accessibility tree")
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if native_action == "click":
            return {**self.env.click_at(x, y), "text": "clicked"}
        if native_action == "type":
            res = self.env.click_at(x, y)
            if not res.get("ok") or res.get("done"):
                return res
            return {**self.env.type_text(arg or ""), "text": f'typed "{arg or ""}"'}
        return result(False, f"unsupported verb {native_action}")

    def point(self, x: float, y: float) -> dict:
        return {**self.env.point(x, y), "text": "pointer moved"}

    def source(self) -> Optional[str]:
        nodes = self._cdp("Accessibility.getFullAXTree", {}).get("nodes", [])
        lines = []
        for node in nodes:
            if node.get("ignored"):
                continue
            role = (node.get("role") or {}).get("value", "")
            name = (node.get("name") or {}).get("value", "")
            lines.append(f"{node.get('nodeId')}: {role} {name!r}")
        return "\n".join(lines)
