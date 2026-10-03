"""DOM adapter for MiniWoB++: reads the environment's own `dom_elements` and invokes by element ref.

Absolute pointer mode. Structured source, confidence 1.0, stable handles (element refs).
"""

from __future__ import annotations

from typing import Optional

from ..schema import Capabilities, Frame, NativeElement
from .base import Adapter, result

TYPEABLE = {"input_text", "input_password", "textarea", "input_number", "input_email", "input_search", "input_tel"}
CLICKABLE = {"button", "a", "input_checkbox", "input_radio", "input_submit", "input_button", "option", "select", "label", "span", "t", "div", "li", "td", "p", "h1", "h2", "h3"}
SKIP_IDS = {"wrap", "query", "area", "reward-display", "sync-task-cover"}


def _scalar(value) -> float:
    try:
        return float(value[0])
    except (TypeError, IndexError):
        return float(value)


class MiniWoBDomAdapter(Adapter):
    name = "dom"

    def __init__(self, environment):
        self.env = environment

    def capabilities(self) -> Capabilities:
        return Capabilities(pointer="absolute", verbs=("click", "type", "scroll_up", "scroll_down"),
                            stable_ids=True, realtime=False, has_source=True)

    def reset(self, seed: Optional[int] = None) -> Frame:
        self.env.reset(seed=seed)
        return self.read()

    def read(self) -> Frame:
        width, height = self.env.size
        elements = []
        for element in self.env.dom_elements:
            native = self._element(element, width, height)
            if native is not None:
                elements.append(native)
        return Frame(image=self.env.image, anchor=self.env.pointer, elements=elements, text=self.env.utterance,
                     info={"task": self.env.task})

    def _element(self, element: dict, width: int, height: int) -> Optional[NativeElement]:
        tag = str(element.get("tag", ""))
        if element.get("id") in SKIP_IDS or tag in {"body", "html", "form"}:
            return None
        flags = element.get("flags")
        is_leaf = bool(flags[3]) if flags is not None and len(flags) > 3 else True
        text = str(element.get("text") or "").strip()
        value = str(element.get("value") or "").strip()
        if tag in TYPEABLE:
            verbs = ("click", "type")
        elif tag in CLICKABLE and (is_leaf or tag in {"button", "a", "input_checkbox", "input_radio"}):
            if not text and tag not in {"button", "input_checkbox", "input_radio", "input_submit"}:
                return None
            verbs = ("click",)
        else:
            return None
        left, top = _scalar(element["left"]), _scalar(element["top"])
        w, h = _scalar(element["width"]), _scalar(element["height"])
        if w <= 0 or h <= 0:
            return None
        box = (max(left / width, 0.0), max(top / height, 0.0), min((left + w) / width, 1.0), min((top + h) / height, 1.0))
        if box[0] >= box[2] or box[1] >= box[3]:
            return None
        label = text or value or (element.get("id") or tag)
        label = f"{label} ({tag.replace('input_', '')})" if tag not in {"t", "span", "div"} else label
        return NativeElement(handle=int(element["ref"]), bbox=box, label=label, role=tag,
                             native_actions=verbs, value=value, confidence=1.0)

    def invoke(self, handle, native_action: str, arg: Optional[str] = None) -> dict:
        if native_action == "click" and handle is not None:
            return self._wrap(self.env.click_ref(handle), "clicked")
        if native_action == "type" and handle is not None:
            return self._wrap(self.env.type_into_ref(handle, arg or ""), f'typed "{arg or ""}"')
        if native_action in ("scroll_up", "scroll_down"):
            return self._wrap(self.env.scroll(native_action.split("_")[1]), native_action.replace("_", " "))
        return result(False, f"unsupported verb {native_action}")

    def point(self, x: float, y: float) -> dict:
        return self._wrap(self.env.point(x, y), "pointer moved")

    def source(self) -> Optional[str]:
        page = self.env.page_source()
        if page:
            return page
        lines = [f"{e.get('ref')}: <{e.get('tag')} id={e.get('id')!r} class={e.get('classes')!r}> {e.get('text') or ''}"
                 for e in self.env.dom_elements]
        return "\n".join(lines)

    @staticmethod
    def _wrap(res: dict, text: str) -> dict:
        res = dict(res)
        res["text"] = res.get("text") or (text if res.get("ok") else "failed")
        return res
